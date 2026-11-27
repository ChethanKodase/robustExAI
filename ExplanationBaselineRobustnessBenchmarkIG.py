"""

export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate    
uv run python ExplanationBaselineRobustnessBenchmarkIG.py --model resnet50 --baseline imagenet_mean --max_images 10 --eps 8 --steps 100

# GPU 0 — ResNet-50
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate    
CUDA_VISIBLE_DEVICES=1 bash -c 'for BL in black white mean blur gaussian uniform eg imagenet_mean; do
  uv run python ExplanationBaselineRobustnessBenchmarkIG.py --model resnet50 --baseline $BL --max_images 100 --eps 0.5 1 2 0.25; done'

# GPU 1 — ViT-B/16   (if eg OOMs: add --eg_k 4 --eg_summands 6)
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate    
CUDA_VISIBLE_DEVICES=2 bash -c 'for BL in black white mean blur gaussian uniform eg imagenet_mean; do
  uv run python ExplanationBaselineRobustnessBenchmarkIG.py --model vit_b_16 --baseline $BL --max_images 100 --eps 0.5 1 2 0.25; done'

# GPU 2 — ConvNeXt-Base  (slowest; skip if you want results sooner)
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate    
CUDA_VISIBLE_DEVICES=3 bash -c 'for BL in black white mean blur gaussian uniform eg imagenet_mean; do
  uv run python ExplanationBaselineRobustnessBenchmarkIG.py --model convnext_base --baseline $BL --max_images 100 --eps 0.5 1 2 0.25; done'

  
uv run python ExplanationBaselineRobustnessBenchmarkIG.py --collate  

"""

import argparse
import csv
import glob
import importlib.util
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))

# Shared machinery (build_model, metrics, plots, data, beta helpers) imported from the GRILL-IG
# file so the IG recipe, similarity metrics and figures are IDENTICAL -> fair head-to-head.
_spec = importlib.util.spec_from_file_location(
    "grill", os.path.join(HERE, "ExplanationGrillAttackonIG.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

BASELINES = ["black", "white", "mean", "blur", "gaussian", "uniform", "eg", "imagenet_mean"]
INPUT_DERIVED = {"mean", "blur"}          # recomputed from the current input every call (and attacked through)
STOCHASTIC = {"gaussian", "uniform", "eg"}  # sampled once per image, then held fixed across clean/attack/adv


# --------------------------------------------------------------------------- #
# Separable Gaussian blur (depthwise) -- differentiable, used for the 'blur' baseline
# --------------------------------------------------------------------------- #
def gaussian_blur(x, sigma):
    if sigma is None or sigma <= 0:
        return x
    r = max(1, int(round(3.0 * sigma)))
    ks = 2 * r + 1
    coords = torch.arange(ks, device=x.device, dtype=x.dtype) - r
    g = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g = g / g.sum()
    C = x.shape[1]
    kx = g.view(1, 1, 1, ks).expand(C, 1, 1, ks)
    ky = g.view(1, 1, ks, 1).expand(C, 1, ks, 1)
    x = F.conv2d(x, kx, padding=(0, r), groups=C)
    x = F.conv2d(x, ky, padding=(r, 0), groups=C)
    return x


# --------------------------------------------------------------------------- #
# Per-image baseline cache (built ONCE per image so stochastic baselines stay fixed
# across the clean map, every attack step, and the final adversarial map).
# --------------------------------------------------------------------------- #
def build_cache(method, x, pool, args, gen):
    """x: [1,3,H,W] pixel. pool: [P,3,H,W] pixel (for eg) or None. gen: torch.Generator."""
    cache = {}
    _, C, H, W = x.shape
    if method == "blur":
        cache["blur_sigma"] = args.blur_sigma
    elif method == "gaussian":
        b = 0.5 + args.gauss_std * torch.randn(args.gauss_k, C, H, W, generator=gen, device=x.device)
        cache["baselines"] = b.clamp(0.0, 1.0)
    elif method == "uniform":
        cache["baselines"] = torch.rand(args.gauss_k, C, H, W, generator=gen, device=x.device)
    elif method == "eg":
        P = pool.shape[0]
        idx = torch.randint(0, P, (args.eg_k,), generator=gen, device=pool.device)
        cache["baselines"] = pool[idx].to(x.device)
    return cache


def resolve_baselines(model, method, x, cache):
    """Return the baseline tensor(s) [K,3,H,W] in PIXEL space for the current input x."""
    if method == "black":
        return torch.zeros_like(x)
    if method == "white":
        return torch.ones_like(x)
    if method == "imagenet_mean":                 # z = 0  <=>  x = channel mean  (the default IG)
        return model.mean.expand_as(x)
    if method == "mean":                          # per-image mean colour (input-derived)
        return x.mean(dim=(2, 3), keepdim=True).expand_as(x)
    if method == "blur":                          # input-derived; gradient flows through the blur
        return gaussian_blur(x, cache["blur_sigma"])
    if method in STOCHASTIC:                       # fixed per image
        return cache["baselines"]
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# Generalised Integrated Gradients in normalised z-space (same recipe / space as
# G.get_expl_ig; with baseline = 'imagenet_mean' this reproduces G.get_expl_ig exactly).
#   IG = mean_k  (z - z_b^k) (.) mean_alpha dP_c/d(path)      ;  path = z_b^k + alpha (z - z_b^k)
# Returns a sum-to-1 heatmap [1,H,W].  Differentiable w.r.t. x when create_graph=True
# (the attacker then backprops through the baseline construction too -> adaptive).
# --------------------------------------------------------------------------- #
def ig_core(model, x, method, cache, class_idx, args, create_graph=False):
    S = args.eg_summands if method == "eg" else args.num_summands
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        z = model.normalize(x)                                    # [1,3,H,W]
        bl = resolve_baselines(model, method, x, cache)           # [K,3,H,W] pixel
        zb = model.normalize(bl)                                  # [K,3,H,W]
        K = zb.shape[0]
        diff = z - zb                                             # [K,3,H,W]
        alphas = z.new_tensor([s / S for s in range(1, S + 1)])   # {1/S..1}
        path = zb.unsqueeze(1) + alphas.view(1, S, 1, 1, 1) * diff.unsqueeze(1)   # [K,S,3,H,W]
        path = path.reshape(K * S, *z.shape[1:])
        logits = model.model(path)                                # path is already in z-space
        P = F.softmax(logits, dim=1)
        sel = P[:, class_idx].sum()
        g = torch.autograd.grad(sel, path, create_graph=create_graph)[0]   # [K*S,3,H,W]
        g = g.view(K, S, *z.shape[1:]).mean(1)                    # mean path gradient per baseline
        ig = (diff * g).mean(0, keepdim=True)                     # [1,3,H,W]  (EG average over K)
        heat = ig.abs().sum(1)                                    # [1,H,W]
        heat = heat / heat.flatten(1).sum(1).view(-1, 1, 1).clamp_min(1e-12)
    return heat


# --------------------------------------------------------------------------- #
# Adaptive Ghorbani top-k attack THROUGH the chosen baseline's IG path.
# Identical objective / selection / beta schedule to ExplanationBaselineAttacksonIG topk,
# but get_expl_ig is replaced by ig_core(method).
# --------------------------------------------------------------------------- #
def topk_attack(model, x, method, cache, args, beta_modules, eps, lr):
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0
    G.change_beta(beta_modules, 1000.0 if beta_growth else None)

    z0 = model.normalize(x)
    with torch.no_grad():
        logits0 = model.model(z0)
        org_acc = F.softmax(logits0, dim=1)
        org_idx = logits0.argmax(dim=1)

    # clean map (for the top-k mask) -- computed at beta=1000 like the baseline file
    h0 = ig_core(model, x, method, cache, org_idx, args, create_graph=False).detach()
    mask = torch.zeros_like(h0).flatten(1)
    mask.scatter_(1, h0.flatten(1).topk(args.topk, dim=1).indices, 1.0)
    mask = mask.view_as(h0)
    org_mass = (h0 * mask).flatten(1).sum(1).clamp_min(1e-12)

    delta = torch.zeros_like(x, requires_grad=True)               # top-k: gradient != 0 at delta=0
    opt = torch.optim.Adam([delta], lr=lr)
    best_adv = x.clone()
    best_loss = torch.full((x.shape[0],), float("inf"), device=x.device)

    for step in range(args.steps + 1):
        if beta_growth:
            G.change_beta(beta_modules, G.get_beta(step, args.steps))
        xin = x + delta
        h_adv = ig_core(model, xin, method, cache, org_idx, args, create_graph=True)
        logits = model.model(model.normalize(xin))
        adv_acc = F.softmax(logits, dim=1)
        cls = logits.argmax(dim=1)

        loss_expl = (h_adv * mask).flatten(1).sum(1) / org_mass           # minimise clean-top-k mass
        loss_output = ((adv_acc - org_acc) ** 2).mean(1)                  # keep the decision
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


# --------------------------------------------------------------------------- #
# EG data pool (training-free: just a handful of real eval images as baselines)
# --------------------------------------------------------------------------- #
def build_pool(dataset, n, device):
    n = min(n, len(dataset))
    imgs = torch.stack([dataset[i][0] for i in range(n)], dim=0).to(device)
    return imgs


# --------------------------------------------------------------------------- #
# One (model, baseline) run over all eps -- reporting/plots identical to G files
# --------------------------------------------------------------------------- #
def run(args):
    G.set_seed(args.seed)
    device = torch.device(args.device)
    model, pixel_tf, categories, beta_modules, crop = G.build_model(
        args.model, device, check_mha=(args.model == "vit_b_16"))
    dataset = G.LabelledImages(args.data_dir, G.load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) < 1:
        raise SystemExit("No labelled images - check --data_dir and --labels.")
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    pool = build_pool(dataset, args.eg_pool, device) if args.baseline == "eg" else None

    n_img = len(dataset)
    out_dir = args.out_dir or os.path.join("results", f"igbase_{args.model}_{args.baseline}_img{n_img}")
    os.makedirs(out_dir, exist_ok=True)
    mnames = ["mse", "pcc", "ssim", "spearman", "topk"]
    qual_dir, quant_dir = os.path.join(out_dir, "qualitative"), os.path.join(out_dir, "quantitative")
    import random as _random
    n_qual = 0 if args.no_plots else min(args.n_qual, n_img)
    qual_names = set(dataset.files[i] for i in _random.Random(args.seed).sample(range(n_img), n_qual))

    print(f"\n########## baseline={args.baseline} | model={args.model} | images={n_img} | "
          f"eps={args.eps} | steps={args.steps} | summands={args.num_summands}"
          f"{' | eg_k=%d eg_summands=%d' % (args.eg_k, args.eg_summands) if args.baseline == 'eg' else ''} "
          f"| beta {'on' if (not args.no_beta_growth) and beta_modules else 'off'} | out={out_dir} ##########")

    summary = []
    for eps_255 in args.eps:
        eps = eps_255 / 255.0
        lr = args.lr_factor * eps
        print(f"\n{'=' * 70}\n{args.baseline} | eps = {eps_255}/255   (Adam lr = {lr:.6f})\n{'=' * 70}")
        rows, qual_samples, cols = [], [], {}
        n = n_clean = n_unchanged = 0
        tag = f"eps{eps_255:g}".replace(".", "p")
        t0 = time.time()

        for bi, (x, y, names) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            gen = torch.Generator(device=device).manual_seed(args.seed * 100000 + bi)
            cache = build_cache(args.baseline, x, pool, args, gen)

            x_adv, org_idx = topk_attack(model, x, args.baseline, cache, args, beta_modules, eps, lr)

            # reported maps: TRUE IG (beta=None already restored inside topk_attack), same cache
            h_o = ig_core(model, x, args.baseline, cache, org_idx, args, create_graph=False).detach()
            with torch.no_grad():
                z_o, z_a = model.normalize(x), model.normalize(x_adv)
                acc_o = F.softmax(model.model(z_o), dim=1)
                logits_a = model.model(z_a)
                acc_a = F.softmax(logits_a, dim=1)
                pred_o, pred_a = org_idx, logits_a.argmax(dim=1)
            h_a = ig_core(model, x_adv, args.baseline, cache, org_idx, args, create_graph=False).detach()

            with torch.no_grad():
                unchanged = pred_a == pred_o
                out_change = (acc_a - acc_o).norm(dim=1)
                linf = (x_adv - x).abs().flatten(1).max(1).values * 255
                img_mse = ((x_adv - x) ** 2).flatten(1).mean(1)
                img_ssim = G.ssim(x_adv, x)
                res = G.expl_metrics(h_a, h_o, args.topk)

            n += len(y)
            n_clean += (pred_o == y).sum().item()
            n_unchanged += unchanged.sum().item()

            for i in range(len(y)):
                row = {"filename": names[i], "baseline": args.baseline, "model": args.model,
                       "true_idx": y[i].item(), "clean_pred_idx": pred_o[i].item(),
                       "adv_pred_idx": pred_a[i].item(),
                       "clean_correct": int(pred_o[i].item() == y[i].item()),
                       "label_unchanged": int(unchanged[i].item()),
                       "output_change_l2": out_change[i].item(),
                       "linf_x255": linf[i].item(), "img_mse": img_mse[i].item(),
                       "img_ssim": img_ssim[i].item()}
                for m in mnames:
                    row[f"adv_vs_orig_{m}"] = res[m][i].item()
                rows.append(row)
                for kname in ["output_change_l2", "linf_x255", "img_mse", "img_ssim"] + \
                             [f"adv_vs_orig_{m}" for m in mnames]:
                    cols.setdefault(kname, []).append(row[kname])

                if names[i] in qual_names:
                    stem = os.path.splitext(names[i])[0]
                    G.save_qualitative(
                        os.path.join(qual_dir, tag, stem), x[i], x_adv[i], h_o[i], h_a[i],
                        caption=f"{names[i]} | baseline={args.baseline} | eps={eps_255}/255 | {args.model} | "
                                f"pred {categories[pred_o[i].item()]} -> {categories[pred_a[i].item()]} | "
                                f"PCC(clean,pert)={row['adv_vs_orig_pcc']:.3f}  "
                                f"top-k={row['adv_vs_orig_topk']:.3f}")
                    qual_samples.append({"name": names[i], "x": x[i].cpu(), "x_adv": x_adv[i].cpu(),
                                         "h_o": h_o[i].cpu(), "h_a": h_a[i].cpu(),
                                         "pred_o": categories[pred_o[i].item()],
                                         "pred_a": categories[pred_a[i].item()],
                                         "pcc_ao": row["adv_vs_orig_pcc"]})

            print(f"  img {bi + 1:3d}/{n_img} | unchanged {n_unchanged / n:.3f} | "
                  f"median PCC(adv,orig) {np.median(cols['adv_vs_orig_pcc']):.3f} | "
                  f"median top-k {np.median(cols['adv_vs_orig_topk']):.3f}")

        elapsed = time.time() - t0
        med = {k: float(np.median(v)) for k, v in cols.items()}
        mean = {k: float(np.mean(v)) for k, v in cols.items()}
        print(f"\n  Images {n} ({elapsed:.1f}s) | clean acc {n_clean / n:.4f} | decision unchanged "
              f"{n_unchanged / n:.4f}")
        print(f"  adv vs orig IG : PCC {med['adv_vs_orig_pcc']:.3f}  SSIM {med['adv_vs_orig_ssim']:.3f}  "
              f"Spearman {med['adv_vs_orig_spearman']:.3f}  top-k {med['adv_vs_orig_topk']:.3f}")

        if not args.no_plots:
            G.save_quantitative_eps(os.path.join(quant_dir, tag), cols, eps_255, mnames)
            G.save_qualitative_grid(os.path.join(qual_dir, tag, "grid"), qual_samples, args.grid_rows)
        with open(os.path.join(out_dir, f"per_image_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        summary.append({"baseline": args.baseline, "model": args.model, "eps_x255": eps_255,
                        "n_images": n, "clean_acc": n_clean / n, "decision_unchanged": n_unchanged / n,
                        "seconds": round(elapsed, 1),
                        **{f"median_{k}": v for k, v in med.items()},
                        **{f"mean_{k}": v for k, v in mean.items()}})

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2, default=str)
    if not args.no_plots:
        G.save_quantitative_summary(quant_dir, summary)

    print(f"\n{args.baseline}/{args.model} ->")
    print(f"{'eps(/255)':>10}{'unchanged':>11}{'PCC a-o':>9}{'Spear a-o':>11}{'top-k a-o':>11}{'img SSIM':>10}")
    for s in summary:
        print(f"{s['eps_x255']:>10g}{s['decision_unchanged']:>11.3f}{s['median_adv_vs_orig_pcc']:>9.3f}"
              f"{s['median_adv_vs_orig_spearman']:>11.3f}{s['median_adv_vs_orig_topk']:>11.3f}"
              f"{s['median_img_ssim']:>10.4f}")
    print(f"\nResults -> {out_dir}")
    return summary


# --------------------------------------------------------------------------- #
# Collate: gather every results/igbase_*/summary.csv into one ranking table.
# Most robust baseline = HIGHEST post-attack median PCC / top-k (per model, eps).
# --------------------------------------------------------------------------- #
def collate(results_root="results"):
    files = sorted(glob.glob(os.path.join(results_root, "igbase_*", "summary.csv")))
    if not files:
        print(f"No results/igbase_*/summary.csv under '{results_root}'. Run some configs first.")
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
    keys = ["baseline", "model", "eps_x255", "n_images", "decision_unchanged",
            "median_adv_vs_orig_pcc", "median_adv_vs_orig_spearman",
            "median_adv_vs_orig_ssim", "median_adv_vs_orig_topk",
            "mean_adv_vs_orig_pcc", "mean_adv_vs_orig_topk", "median_img_ssim"]
    out = os.path.join(results_root, "igbase_RANKING.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r.get("model", ""), r.get("eps_x255", 0),
                                             r.get("median_adv_vs_orig_pcc", 1e9))):
            w.writerow(r)

    print(f"\n{'=' * 78}\nIG baseline robustness ranking  (lower PCC/top-k = less robust)\n{'=' * 78}")
    for model in sorted(set(r["model"] for r in rows)):
        for eps in sorted(set(r["eps_x255"] for r in rows if r["model"] == model)):
            sub = [r for r in rows if r["model"] == model and r["eps_x255"] == eps]
            sub.sort(key=lambda r: r["median_adv_vs_orig_pcc"], reverse=True)   # most robust first
            print(f"\n  {model} | eps={eps:g}/255   (ranked MOST -> LEAST robust by median PCC)")
            print(f"    {'baseline':>14}{'PCC':>8}{'Spear':>8}{'SSIM':>8}{'top-k':>8}{'unchanged':>11}")
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
                   help="ignore all other args; gather results/igbase_*/summary.csv into a ranking")
    p.add_argument("--results_root", default="results")
    p.add_argument("--model", default="resnet50",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--baseline", default="black", choices=BASELINES)
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default=None,
                   help="default results/igbase_<model>_<baseline>_img<N>")
    p.add_argument("--eps", type=float, nargs="+", default=[4.0, 8.0], help="L_inf in 1/255")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1, help="Adam lr = lr_factor * eps")
    p.add_argument("--num_summands", type=int, default=30, help="IG path points (non-eg baselines)")
    p.add_argument("--pref_expl", type=float, default=100.0, help="top-k loss weight")
    p.add_argument("--pref_out", type=float, default=1e6, help="output-preservation weight")
    p.add_argument("--topk", type=int, default=1000, help="k for the top-k attack & metric")
    # baseline-specific knobs
    p.add_argument("--blur_sigma", type=float, default=5.0, help="'blur' baseline Gaussian sigma (px)")
    p.add_argument("--gauss_std", type=float, default=0.3, help="'gaussian' baseline noise std in [0,1]")
    p.add_argument("--gauss_k", type=int, default=1, help="# noise baselines for gaussian/uniform (avg)")
    p.add_argument("--eg_k", type=int, default=8, help="# data baselines for Expected Gradients")
    p.add_argument("--eg_summands", type=int, default=8, help="IG path points per eg baseline")
    p.add_argument("--eg_pool", type=int, default=64, help="# eval images used as the eg baseline pool")
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
        collate(args.results_root)
        return
    run(args)


if __name__ == "__main__":
    main()
