"""
Adaptive-attack robustness benchmark of IG PATH methods: Blur IG, Guided IG, AGI.
================================================================================

Companion to ExplanationBaselineRobustnessBenchmarkIG.py (the straight-line baselines) and
ExplanationBaselineRobustnessBenchmarkExtraIG.py (max_distance / max_entropy / big). Those vary
the *baseline* on a straight path; this file varies the *integration path* itself. It imports
the base benchmark and reuses its EXACT model build, top-k adaptive attack, similarity metrics,
plots and per-config folder layout, so the numbers are directly comparable; only the explanation
map changes. Results go to a distinct `igpath_*` folder family; `--collate` here ranks the
straight-line (`igbase_*`) and path (`igpath_*`) results together.

Methods (--method)
    blur_ig    Blur Integrated Gradients (Xu, Venugopalan, Sundararajan, CVPR 2020,
               arXiv:2004.03383; code github.com/PAIR-code/saliency). Integrates gradients along
               a path of progressively Gaussian-blurred images:
                   attr = -sum_i (sigma_{i+1}-sigma_i) * d(blur)/d(sigma)|_{sigma_i} * df/dx|_{blur_i}
               over a sigma schedule 0..max_sigma. DIFFERENTIABLE -> attacked FULLY ADAPTIVELY
               (the attack backprops through the whole blur-path IG). Gaussian blur is done in the
               frequency domain (FFT), so cost is independent of sigma; this uses circular (periodic)
               boundaries where the authors use zero-padding -- a negligible interior difference.
    guided_ig  Guided Integrated Gradients (Kapishnikov et al., CVPR 2021, arXiv:2106.09788; code
               github.com/PAIR-code/saliency). Adaptive path: at each of x_steps steps it moves only
               the `fraction` of features with the SMALLEST |gradient|, staying within `max_dist` of
               the straight line. The path construction is a non-differentiable iterative procedure,
               so (as for BIG) the FORWARD/report map uses the authors' exact algorithm, and the
               ATTACK uses a straight-line-IG surrogate gradient (BPDA) that targets this method's own
               top-k pixels. Robustness here is thus a LOWER BOUND on attackability; a method that
               looks robust needs the stronger unrolled attack before any robustness claim.
    agi        Adversarial Gradient Integration (Pan, Li, Zhu, IJCAI 2021, doi:10.24963/ijcai.2021/396;
               code github.com/pd90506/AGI). Baseline-free: for each of `topk` target classes it runs a
               steepest-ascent adversarial search and accumulates  -dP_pred/dx * dx  along the way,
               summed over classes. Ported faithfully from AGI_main.py (including fgsm_step perturbing
               from the ORIGINAL image each iteration). Same adaptive-attack treatment as guided_ig
               (true forward, straight-IG surrogate attack).

All maps are reduced to a sum-to-1 heatmap with the SAME reduction as the baselines (|.| summed over
channels, normalised), so every method is measured identically. The attack (Ghorbani top-k), the
class-preservation constraint, beta-growth, eps, metrics and plots are the base file's.

Results  ->  results/igpath_<model>_<method>_img<N>/   (distinct from igbase_*; nothing overwrites)
    rank everything together:  uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --collate

Usage
-----

    uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --model resnet50 --method blur_ig   --max_images 100 --eps 0.25 0.5
    uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --model resnet50 --method guided_ig --max_images 100 --eps 0.25 0.5
    uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --model resnet50 --method agi       --max_images 100 --eps 0.25 0.5
    uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --collate


Run it (one method per GPU, same ε/images so they land on the same curve):

export CUDA_VISIBLE_DEVICES=4
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --model resnet50 --method blur_ig   --max_images 100 --eps 0.5 1 2 0.25

export CUDA_VISIBLE_DEVICES=5
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --model resnet50 --method guided_ig --max_images 100 --eps 0.5 1 2 0.25

export CUDA_VISIBLE_DEVICES=6
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --model resnet50 --method agi       --max_images 100 --eps 0.5 1 2 0.25


Then rank all fourteen methods (straight-line + path) together:

uv run python ExplanationPathMethodsRobustnessBenchmarkIG.py --collate
"""

import argparse
import csv
import glob
import importlib.util
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "basebench", os.path.join(HERE, "ExplanationBaselineRobustnessBenchmarkIG.py"))
BB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BB)
G = BB.G

PATH_METHODS = ["blur_ig", "guided_ig", "agi"]


# --------------------------------------------------------------------------- #
# shared reduction -- identical to BB.ig_core (|.| over channels, sum-to-1)
# --------------------------------------------------------------------------- #
def _reduce(attr):                                   # attr [1,C,H,W] -> [1,H,W]
    heat = attr.abs().sum(1)
    return heat / heat.flatten(1).sum(1).clamp_min(1e-12).view(-1, 1, 1)


# --------------------------------------------------------------------------- #
# Blur IG -- differentiable, FFT Gaussian blur (cost independent of sigma)
# --------------------------------------------------------------------------- #
def _fft_blur(x, sigma):
    """Circular Gaussian blur via FFT. Frequency response of a spatial Gaussian with std
    sigma (pixels) is exp(-2 pi^2 sigma^2 f^2)."""
    if sigma <= 0:
        return x
    _, _, H, W = x.shape
    fy = torch.fft.fftfreq(H, d=1.0, device=x.device)
    fx = torch.fft.fftfreq(W, d=1.0, device=x.device)
    gy = torch.exp(-2.0 * (math.pi ** 2) * (sigma ** 2) * fy ** 2)
    gx = torch.exp(-2.0 * (math.pi ** 2) * (sigma ** 2) * fx ** 2)
    filt = (gy.view(H, 1) * gx.view(1, W))           # [H,W]
    return torch.fft.ifft2(torch.fft.fft2(x) * filt).real


def _blur_ig_map(model, x, class_idx, args, create_graph=False):
    max_sigma, steps, gstep = args.blurig_max_sigma, args.blurig_steps, args.blurig_grad_step
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        sigmas = [i * max_sigma / steps for i in range(steps + 1)]
        diffs = x.new_tensor([sigmas[i + 1] - sigmas[i] for i in range(steps)]).view(steps, 1, 1, 1)
        xs = torch.cat([_fft_blur(x, sigmas[i]) for i in range(steps)], dim=0)            # [S,C,H,W]
        xs_hi = torch.cat([_fft_blur(x, sigmas[i] + gstep) for i in range(steps)], dim=0)
        ggrad = (xs_hi - xs) / gstep                                                      # d(blur)/d(sigma)
        P = F.softmax(model.model(model.normalize(xs)), dim=1)[:, class_idx].sum()
        grad = torch.autograd.grad(P, xs, create_graph=create_graph)[0]                   # df/dx at each blur
        attr = -(diffs * ggrad * grad).sum(0, keepdim=True)                               # [1,C,H,W]
        heat = _reduce(attr)
    return heat


# --------------------------------------------------------------------------- #
# Guided IG -- faithful port of PAIR-code/saliency guided_ig_impl (numpy, sequential).
# Forward/report only; the attack uses a straight-IG surrogate.
# --------------------------------------------------------------------------- #
def _quantile_lower(a, q):
    try:
        return np.quantile(a, q, method="lower")
    except TypeError:
        return np.quantile(a, q, interpolation="lower")


def _guided_ig_impl(x_input, x_baseline, grad_func, steps, fraction, max_dist):
    EPS = 1e-9
    l1 = lambda a, b: np.abs(a - b).sum()
    to_alpha = lambda xp: np.where(x_input - x_baseline != 0,
                                   (xp - x_baseline) / (x_input - x_baseline), np.nan)
    to_x = lambda al, hi, lo: lo + (hi - lo) * al
    x_input = np.asarray(x_input, dtype=np.float64)
    x_baseline = np.asarray(x_baseline, dtype=np.float64)
    x = x_baseline.copy()
    l1_total = l1(x_input, x_baseline)
    attr = np.zeros_like(x_input)
    if np.abs(x_input - x_baseline).sum() == 0:
        return attr
    for step in range(steps):
        grad_actual = grad_func(x)
        grad = grad_actual.copy()
        alpha = (step + 1.0) / steps
        alpha_min, alpha_max = max(alpha - max_dist, 0.0), min(alpha + max_dist, 1.0)
        x_min = to_x(alpha_min, x_input, x_baseline)
        x_max = to_x(alpha_max, x_input, x_baseline)
        l1_target = l1_total * (1 - (step + 1) / steps)
        gamma = np.inf
        while gamma > 1.0:
            x_old = x.copy()
            x_alpha = to_alpha(x)
            x_alpha[np.isnan(x_alpha)] = alpha_max
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]
            l1_current = l1(x, x_input)
            if math.isclose(l1_target, l1_current, rel_tol=EPS, abs_tol=EPS):
                attr += (x - x_old) * grad_actual
                break
            grad[x == x_max] = np.inf
            threshold = _quantile_lower(np.abs(grad), fraction)
            s = np.logical_and(np.abs(grad) <= threshold, grad != np.inf)
            l1_s = (np.abs(x - x_max) * s).sum()
            gamma = (l1_current - l1_target) / l1_s if l1_s > 0 else np.inf
            if gamma > 1.0:
                x[s] = x_max[s]
            else:
                x[s] = to_x(gamma, x_max, x)[s]
            attr += (x - x_old) * grad_actual
    return attr


def _guided_ig_map(model, x, class_idx, args):
    device = x.device
    ci = int(class_idx.item())

    def grad_func(x_np):
        xt = torch.tensor(x_np, dtype=torch.float32, device=device).unsqueeze(0).requires_grad_(True)
        p = F.softmax(model.model(model.normalize(xt.clamp(0, 1))), dim=1)[0, ci]
        g = torch.autograd.grad(p, xt)[0][0]
        return g.detach().cpu().numpy().astype(np.float64)

    x_np = x[0].detach().cpu().numpy().astype(np.float64)
    attr = _guided_ig_impl(x_np, np.zeros_like(x_np), grad_func,
                           args.guided_steps, args.guided_fraction, args.guided_max_dist)
    attr_t = torch.from_numpy(attr).to(device).unsqueeze(0).float()
    return _reduce(attr_t).detach()


# --------------------------------------------------------------------------- #
# AGI -- faithful port of pd90506/AGI (AGI_main.py). Forward/report only.
# --------------------------------------------------------------------------- #
def _agi_pgd(model, x, init_pred, targeted, eps, max_iter):
    """One steepest-ascent search toward `targeted`; accumulates -dP_init/dx * dx.
    fgsm_step perturbs from the ORIGINAL image each iteration (as in the authors' code)."""
    perturbed = x.clone()
    c_delta = torch.zeros_like(x)
    for _ in range(max_iter):
        perturbed = perturbed.detach().requires_grad_(True)
        out = model(perturbed)                                   # NormalizedModel -> logits
        if out.argmax(1).item() == targeted:
            break
        p = F.softmax(out, dim=1)
        g_adv = torch.autograd.grad(p[0, targeted], perturbed, retain_graph=True)[0]
        g_lab = torch.autograd.grad(p[0, init_pred], perturbed)[0]
        perturbed_rect = (x + eps * g_adv.sign()).clamp(0, 1)    # from ORIGINAL x
        d = perturbed_rect - x
        c_delta = c_delta + (-g_lab * d)
        perturbed = perturbed_rect
    return c_delta


def _agi_map(model, x, class_idx, args):
    topk, eps, max_iter = args.agi_topk, args.agi_eps, args.agi_iter
    selected = range(0, 999, max(1, int(1000 / topk)))
    init = int(class_idx.item())
    step_grad = torch.zeros_like(x)
    with torch.enable_grad():
        for l in selected:
            if l == init:
                continue
            step_grad = step_grad + _agi_pgd(model, x, init, l, eps, max_iter)
    return _reduce(step_grad).detach()


# --------------------------------------------------------------------------- #
# Dispatch: replace BB.ig_core so reporting (h_o/h_a) and the blur_ig attack use the
# right map; replace BB.topk_attack so guided_ig/agi use the surrogate attack.
# --------------------------------------------------------------------------- #
_orig_ig_core = BB.ig_core
_orig_topk_attack = BB.topk_attack
_orig_build_cache = BB.build_cache


def ig_core_ext(model, x, method, cache, class_idx, args, create_graph=False):
    if method == "blur_ig":
        return _blur_ig_map(model, x, class_idx, args, create_graph)
    if method == "guided_ig":
        return _guided_ig_map(model, x, class_idx, args)         # true map (report / mask)
    if method == "agi":
        return _agi_map(model, x, class_idx, args)               # true map (report / mask)
    return _orig_ig_core(model, x, method, cache, class_idx, args, create_graph)


def _surrogate_topk_attack(model, x, method, cache, args, beta_modules, eps, lr):
    """Top-k attack whose TARGET is this method's own top-k pixels (true map), optimised with a
    straight-line-IG (black baseline) surrogate gradient. BPDA-style for the non-differentiable path."""
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0
    G.change_beta(beta_modules, 1000.0 if beta_growth else None)
    with torch.no_grad():
        logits0 = model.model(model.normalize(x))
        org_acc = F.softmax(logits0, dim=1)
        org_idx = logits0.argmax(dim=1)

    h0 = BB.ig_core(model, x, method, cache, org_idx, args, create_graph=False).detach()   # TRUE map
    mask = torch.zeros_like(h0).flatten(1)
    mask.scatter_(1, h0.flatten(1).topk(args.topk, dim=1).indices, 1.0)
    mask = mask.view_as(h0)
    org_mass = (h0 * mask).flatten(1).sum(1).clamp_min(1e-12)

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    best_adv = x.clone()
    best_loss = torch.full((x.shape[0],), float("inf"), device=x.device)

    for step in range(args.steps + 1):
        if beta_growth:
            G.change_beta(beta_modules, G.get_beta(step, args.steps))
        xin = x + delta
        h_surr = _orig_ig_core(model, xin, "black", {}, org_idx, args, create_graph=True)  # surrogate
        logits = model.model(model.normalize(xin))
        adv_acc = F.softmax(logits, dim=1)
        cls = logits.argmax(dim=1)
        loss_expl = (h_surr * mask).flatten(1).sum(1) / org_mass
        loss_output = ((adv_acc - org_acc) ** 2).mean(1)
        loss = args.pref_expl * loss_expl + args.pref_out * loss_output
        with torch.no_grad():
            keep = (cls == org_idx) & (loss_expl < best_loss)
            best_adv[keep] = xin[keep].detach()
            best_loss[keep] = loss_expl[keep]
        if step == args.steps:
            break
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x + delta, 0, 1) - x)

    G.change_beta(beta_modules, None)
    return best_adv, org_idx


def topk_attack_ext(model, x, method, cache, args, beta_modules, eps, lr):
    if method in ("guided_ig", "agi"):
        return _surrogate_topk_attack(model, x, method, cache, args, beta_modules, eps, lr)
    # blur_ig (differentiable, via ig_core_ext) and any baseline -> original attack
    return _orig_topk_attack(model, x, method, cache, args, beta_modules, eps, lr)


def build_cache_ext(method, x, pool, args, gen):
    if method in PATH_METHODS:
        return {}
    return _orig_build_cache(method, x, pool, args, gen)


BB.ig_core = ig_core_ext
BB.topk_attack = topk_attack_ext
BB.build_cache = build_cache_ext


# --------------------------------------------------------------------------- #
# Collate straight-line (igbase_*) + path (igpath_*) together.
# --------------------------------------------------------------------------- #
def collate_all(results_root="results"):
    files = sorted(glob.glob(os.path.join(results_root, "igbase_*", "summary.csv")) +
                   glob.glob(os.path.join(results_root, "igpath_*", "summary.csv")))
    if not files:
        print(f"No results under '{results_root}'.")
        return
    rows = []
    for f in files:
        with open(f, newline="") as fh:
            rows.extend(list(csv.DictReader(fh)))
    for r in rows:
        for k, v in r.items():
            if k not in ("baseline", "model"):
                try:
                    r[k] = float(v)
                except (ValueError, TypeError):
                    pass
    out = os.path.join(results_root, "igall_RANKING.csv")
    keys = ["baseline", "model", "eps_x255", "n_images", "decision_unchanged",
            "median_adv_vs_orig_pcc", "median_adv_vs_orig_spearman",
            "median_adv_vs_orig_ssim", "median_adv_vs_orig_topk",
            "mean_adv_vs_orig_pcc", "mean_adv_vs_orig_topk", "median_img_ssim"]
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r.get("model", ""), r.get("eps_x255", 0),
                                             r.get("median_adv_vs_orig_pcc", 1e9))):
            w.writerow(r)
    print(f"\n{'=' * 82}\nIG robustness ranking: straight-line + path methods "
          f"(lower PCC/top-k = less robust)\n{'=' * 82}")
    for model in sorted(set(r["model"] for r in rows)):
        for eps in sorted(set(r["eps_x255"] for r in rows if r["model"] == model)):
            sub = [r for r in rows if r["model"] == model and r["eps_x255"] == eps]
            sub.sort(key=lambda r: r["median_adv_vs_orig_pcc"], reverse=True)
            print(f"\n  {model} | eps={eps:g}/255   (MOST -> LEAST robust by median PCC)")
            print(f"    {'method':>14}{'PCC':>8}{'Spear':>8}{'SSIM':>8}{'top-k':>8}{'unchanged':>11}")
            for r in sub:
                print(f"    {r['baseline']:>14}{r['median_adv_vs_orig_pcc']:>8.3f}"
                      f"{r['median_adv_vs_orig_spearman']:>8.3f}{r['median_adv_vs_orig_ssim']:>8.3f}"
                      f"{r['median_adv_vs_orig_topk']:>8.3f}{r['decision_unchanged']:>11.3f}")
    print(f"\nRanking table -> {out}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collate", action="store_true",
                   help="rank igbase_* + igpath_* results together")
    p.add_argument("--results_root", default="results")
    p.add_argument("--model", default="resnet50",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--method", default="blur_ig", choices=PATH_METHODS)
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default=None,
                   help="default results/igpath_<model>_<method>_img<N>")
    p.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.5], help="L_inf in 1/255")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1)
    p.add_argument("--num_summands", type=int, default=30, help="surrogate straight-IG steps (guided/agi)")
    p.add_argument("--pref_expl", type=float, default=100.0)
    p.add_argument("--pref_out", type=float, default=1e6)
    p.add_argument("--topk", type=int, default=1000)
    # blur_ig
    p.add_argument("--blurig_max_sigma", type=float, default=50.0, help="max Gaussian sigma (Blur IG)")
    p.add_argument("--blurig_steps", type=int, default=30, help="blur-path steps (authors use 100)")
    p.add_argument("--blurig_grad_step", type=float, default=0.01, help="finite-diff step for d(blur)/d(sigma)")
    # guided_ig
    p.add_argument("--guided_steps", type=int, default=128, help="Guided IG path steps (authors use 200)")
    p.add_argument("--guided_fraction", type=float, default=0.25, help="fraction of features moved per step")
    p.add_argument("--guided_max_dist", type=float, default=0.02, help="max L1 deviation from straight line")
    # agi
    p.add_argument("--agi_eps", type=float, default=0.05, help="AGI adversarial step size")
    p.add_argument("--agi_iter", type=int, default=15, help="AGI max adversarial-search iterations")
    p.add_argument("--agi_topk", type=int, default=15, help="AGI number of target classes")
    # unused by path methods but read by BB.run
    p.add_argument("--blur_sigma", type=float, default=5.0)
    p.add_argument("--gauss_std", type=float, default=0.3)
    p.add_argument("--gauss_k", type=int, default=1)
    p.add_argument("--eg_k", type=int, default=8)
    p.add_argument("--eg_summands", type=int, default=8)
    p.add_argument("--eg_pool", type=int, default=64)
    # bookkeeping / plotting
    p.add_argument("--max_images", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_qual", type=int, default=10)
    p.add_argument("--grid_rows", type=int, default=5)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--no_beta_growth", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.collate:
        collate_all(args.results_root)
        return

    # distinct igpath_ folder; map BB.run's internal "baseline" field to our method name
    args.baseline = args.method
    n = args.max_images if args.max_images else "all"
    if args.out_dir is None:
        args.out_dir = os.path.join("results", f"igpath_{args.model}_{args.method}_img{n}")
    BB.run(args)


if __name__ == "__main__":
    main()
