"""
Adaptive-attack robustness benchmark of Geodesic Integrated Gradients (GIG).
============================================================================

Companion to the straight-line (igbase_*) and path-method (igpath_*) benchmarks. GIG replaces IG's
straight line with a GEODESIC under a model-induced Riemannian metric: the metric is the inner product
of the model's output gradient, <grad F, grad F>, so "arc length" is large where the model output
changes fast (near decision boundaries) and small in flat regions. A geodesic is the path of LEAST
such arc length -- it bends around the high-gradient regions that make IG noisy and fragile.

Based on: Salek et al., "Using the Path of Least Resistance to Explain Deep Networks" (Geodesic IG;
code github.com/sina-salek/geodesic-ig, arXiv:2502.12108). That repo ships two implementations that do
NOT port cleanly to an ImageNet adaptive benchmark:
  - geodesic_knn_ig.py: a kNN graph over a data pool + A* shortest path. Discrete / non-differentiable,
    needs a pool, and a kNN graph in 150k-dim pixel space is dominated by the curse of dimensionality.
  - geodesic_svi_ig.py: a continuous path optimised under <grad F, grad F>, but built on Pyro SVI and
    with an energy that MAXIMISES the gradient norm (opposite sign to the paper's "least resistance").
This file implements the paper's PRINCIPLE directly, in clean torch (no Pyro, no data pool), so it is
comparable to your other methods and runs at ImageNet scale:

  1. Initialise the straight path   p_k = baseline + (k/(S-1)) (x - baseline),  k = 0..S-1  (baseline=black).
  2. Optimise the INTERIOR points (endpoints fixed) with Adam to minimise
         E = beta * sum_k || grad F(p_k) ||   +   lam_dist * sum_k || p_k - p_k^straight ||^2
     i.e. pull the path through low-gradient regions (geodesic / least resistance) while staying near
     the straight line. This differentiates ||grad F|| w.r.t. the path (a double-backward per step).
  3. Attribute along the optimised path:  attr = sum_k grad F(p_k) ( p_{k+1} - p_k ),  reduced to a
     sum-to-1 heatmap the same way as every other method.

Adaptivity: the path optimisation is a non-differentiable-to-attack inner loop, so (as for guided_ig /
agi / big) the FORWARD/report map is the true geodesic-path IG, and the ATTACK uses a straight-line-IG
(black-baseline) surrogate gradient that targets GIG's own top-k pixels (BPDA). The reported robustness
is therefore a LOWER BOUND on attackability -- strengthen with an unrolled attack before claiming GIG is
robust. Baseline is black for both GIG and its surrogate, so GIG vs the `black` baseline isolates the
PURE effect of the geodesic path (same endpoints, straight vs geodesic).

NOTE: unlike the ported methods, GIG's path-optimisation hyper-parameters (beta, lam_dist, geo_iters, lr)
have no reference values here -- tune them so the geodesic map actually differs from straight-IG but stays
stable. Quick check: if GIG's map equals the `black`/`imagenet_mean` straight-IG map, raise --gig_beta or
lower --gig_lam_dist.

Results  ->  results/igpath_<model>_gig_img<N>/   (ranks together with igbase_* and igpath_*)
    uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --collate

Usage
-----
    uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --model resnet50 --max_images 100 --eps 0.25 0.5
    uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --collate
    (GIG is expensive -- a path optimisation per image -- start with --max_images 30.)


cd /data1/chethan/robustExAI && source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --model resnet50 --max_images 100 --eps 0.25 0.5
uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --collate   # igbase_ + igpath_ ranked together


export CUDA_VISIBLE_DEVICES=7
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate  
uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --model resnet50 --max_images 100 --eps 0.5 1 2 0.25

uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --model vit_b_16 --max_images 100 --eps 0.5 1 2 0.25



uv run python ExplanationGeodesicIGRobustnessBenchmarkIG.py --collate   # igbase_ + igpath_ ranked together

"""

import argparse
import csv
import glob
import importlib.util
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "basebench", os.path.join(HERE, "ExplanationBaselineRobustnessBenchmarkIG.py"))
BB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BB)
G = BB.G

METHOD = "gig"


def _reduce(attr):                                   # [1,C,H,W] -> [1,H,W], same as BB.ig_core
    heat = attr.abs().sum(1)
    return heat / heat.flatten(1).sum(1).clamp_min(1e-12).view(-1, 1, 1)


# --------------------------------------------------------------------------- #
# Geodesic path optimisation + attribution (true map; used for report + attack mask)
# --------------------------------------------------------------------------- #
def _gig_map(model, x, class_idx, args):
    S = args.gig_steps
    ci = int(class_idx.item())
    x = x.detach()
    baseline = torch.zeros_like(x)                                   # black baseline
    alphas = torch.linspace(0.0, 1.0, S, device=x.device).view(S, 1, 1, 1)
    path_init = baseline + alphas * (x - baseline)                   # [S,C,H,W]

    def grad_F(p, create_graph):                                     # grad of softmax_c(F) at each path point
        Fout = F.softmax(model.model(model.normalize(p)), dim=1)[:, ci]
        return torch.autograd.grad(Fout.sum(), p, create_graph=create_graph)[0]

    # optimise the interior points to minimise  beta*sum||gradF|| + lam*sum||p - p_straight||^2
    delta = torch.zeros(S - 2, *x.shape[1:], device=x.device, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=args.gig_lr)
    for _ in range(args.gig_geo_iters):
        opt.zero_grad(set_to_none=True)
        p = torch.cat([path_init[:1], path_init[1:-1] + delta, path_init[-1:]], dim=0)
        gF = grad_F(p, create_graph=True)                          # double-backward (differentiate ||gradF|| w.r.t. delta)
        curv = gF.flatten(1).norm(dim=1).sum()
        dist = (p - path_init).flatten(1).pow(2).sum()
        (args.gig_beta * curv + args.gig_lam_dist * dist).backward()
        opt.step()

    # attribute along the optimised path (first-order grads)
    with torch.enable_grad():
        p = torch.cat([path_init[:1], path_init[1:-1] + delta.detach(), path_init[-1:]], dim=0)
        p = p.detach().requires_grad_(True)
        gF = grad_F(p, create_graph=False)                         # first-order, p is a leaf
        seg = p[1:] - p[:-1]                                        # [S-1,C,H,W]
        attr = (gF[:-1] * seg).sum(0, keepdim=True)                 # sum_k gradF(p_k)(p_{k+1}-p_k)
    return _reduce(attr).detach()


# --------------------------------------------------------------------------- #
# Dispatch + BPDA surrogate attack (straight-IG from black), identical pattern to the path file
# --------------------------------------------------------------------------- #
_orig_ig_core = BB.ig_core
_orig_topk_attack = BB.topk_attack
_orig_build_cache = BB.build_cache


def ig_core_ext(model, x, method, cache, class_idx, args, create_graph=False):
    if method == METHOD:
        return _gig_map(model, x, class_idx, args)
    return _orig_ig_core(model, x, method, cache, class_idx, args, create_graph)


def _surrogate_topk_attack(model, x, method, cache, args, beta_modules, eps, lr):
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0
    G.change_beta(beta_modules, 1000.0 if beta_growth else None)
    with torch.no_grad():
        logits0 = model.model(model.normalize(x))
        org_acc = F.softmax(logits0, dim=1)
        org_idx = logits0.argmax(dim=1)

    h0 = BB.ig_core(model, x, method, cache, org_idx, args, create_graph=False).detach()   # TRUE geodesic map
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
    if method == METHOD:
        return _surrogate_topk_attack(model, x, method, cache, args, beta_modules, eps, lr)
    return _orig_topk_attack(model, x, method, cache, args, beta_modules, eps, lr)


def build_cache_ext(method, x, pool, args, gen):
    if method == METHOD:
        return {}
    return _orig_build_cache(method, x, pool, args, gen)


BB.ig_core = ig_core_ext
BB.topk_attack = topk_attack_ext
BB.build_cache = build_cache_ext


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
    print(f"\n{'=' * 82}\nIG robustness ranking: straight-line + path + geodesic "
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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collate", action="store_true")
    p.add_argument("--results_root", default="results")
    p.add_argument("--model", default="resnet50",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default=None, help="default results/igpath_<model>_gig_img<N>")
    p.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.5], help="L_inf in 1/255")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1)
    p.add_argument("--num_summands", type=int, default=30, help="surrogate straight-IG steps")
    p.add_argument("--pref_expl", type=float, default=100.0)
    p.add_argument("--pref_out", type=float, default=1e6)
    p.add_argument("--topk", type=int, default=1000)
    # GIG geodesic knobs (no reference values -- tune so the path differs from straight but stays stable)
    p.add_argument("--gig_steps", type=int, default=20, help="path points S")
    p.add_argument("--gig_geo_iters", type=int, default=50, help="path-optimisation Adam steps")
    p.add_argument("--gig_beta", type=float, default=1.0, help="weight on sum||grad F|| (least resistance)")
    p.add_argument("--gig_lam_dist", type=float, default=0.1, help="weight on stay-near-straight")
    p.add_argument("--gig_lr", type=float, default=0.01, help="path-optimisation lr")
    # unused by GIG but read by BB.run
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

    args.baseline = METHOD                                           # BB.run labels rows with this
    n = args.max_images if args.max_images else "all"
    if args.out_dir is None:
        args.out_dir = os.path.join("results", f"igpath_{args.model}_{METHOD}_img{n}")
    BB.run(args)


if __name__ == "__main__":
    main()
