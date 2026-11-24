"""
Baseline untargeted IG explanation attacks, MULTI-MODEL (ConvNeXt-Base / ViT-B/16 / ResNet-50),
in ONE runner -- so the four terminal-only baselines can be compared against GRILL-IG on the SAME
models at matched settings.

    *** New file. Does NOT edit the originals. The four single-model baseline scripts
        (ExplanationNotargetAttackonIG / ExplanationUntargetedDrombowskiAttackonIG /
        ExplanationMasscenterAttackonIG / ExplanationCosineAttackonIG) are ConvNeXt-only;
        this runner reproduces their losses exactly but uses the multi-model build_model
        (with ProbeMHA for ViT), get_expl_ig, metrics and plots IMPORTED from
        ExplanationGrillAttackonIG.py, so every number here is directly comparable to the
        ig_grill_attack_<model> results. ***

Methods (--method, one or more):
    topk        Ghorbani IFIA top-k: minimise the clean-top-k IG mass.  loss = sum_{top-k}(h_adv)/mass
                -> delta starts at 0 (gradient non-zero there).  prefactors [100, 1e6].
    dombrowski  reversed-Dombrowski: push the IG map away from its clean self (max MSE).
                loss = -||h_adv - h_clean||^2.  random start.  prefactors [1e11, 1e6].
    masscenter  Ghorbani mass-centre: displace the IG centre of mass.
                loss = -||c(h_adv) - c(h_clean)||^2 on [0,1] coords.  random start.  prefactors [1e4, 1e6].
    cosine      Wang & Kong cosine (~ Kendall): minimise cos(h_adv, h_clean).
                loss = cos(h_adv, h_clean).  random start.  prefactors [100, 1e6].
Every method: frozen model, white-box, input-only L_inf attack, projected Adam, beta growth on ReLU
(ResNet), hard class-preserving constraint, best iterate = min(own loss_expl) among class-preserving.
All defaults match GRILL-IG and the originals (steps 500, num_summands 30, lr = 0.1*eps).

Outputs -> results/baseline_<method>_<model>/  (distinct from the originals' dirs; no overlap):
    per_image_<eps>.csv, summary.csv, qualitative/, quantitative/  (same format as ig_grill_attack_*).

Usage
-----
    # one method on one model (put each on its own GPU to parallelise):
    uv run python ExplanationBaselineAttacksonIG.py --method topk       --model vit_b_16 --max_images 100 --eps 4 8 --n_qual 10
    uv run python ExplanationBaselineAttacksonIG.py --method dombrowski  --model vit_b_16 --max_images 100 --eps 4 8
    uv run python ExplanationBaselineAttacksonIG.py --method masscenter  --model resnet50 --max_images 100 --eps 4 8
    uv run python ExplanationBaselineAttacksonIG.py --method cosine      --model resnet50 --max_images 100 --eps 4 8
    # or all four methods in one process (slow): omit --method.
    (eps in units of 1/255. Match the GRILL-IG runs: --eps 4 8 --max_images 100, default steps/summands.)



# ========== topk (Ghorbani IFIA) ==========
export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationBaselineAttacksonIG.py --method topk       --model vit_b_16      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method topk       --model resnet50      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method topk       --model convnext_base --max_images 100 --eps 4 8 --n_qual 10

# ========== dombrowski (reversed-Dombrowski) ==========
export CUDA_VISIBLE_DEVICES=1
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationBaselineAttacksonIG.py --method dombrowski --model vit_b_16      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method dombrowski --model resnet50      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method dombrowski --model convnext_base --max_images 100 --eps 4 8 --n_qual 10

# ========== masscenter (Ghorbani mass-centre) ==========
export CUDA_VISIBLE_DEVICES=2
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationBaselineAttacksonIG.py --method masscenter --model vit_b_16      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method masscenter --model resnet50      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method masscenter --model convnext_base --max_images 100 --eps 4 8 --n_qual 10

# ========== cosine (Wang & Kong cosine ~ Kendall) ==========
export CUDA_VISIBLE_DEVICES=4
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationBaselineAttacksonIG.py --method cosine     --model vit_b_16      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method cosine     --model resnet50      --max_images 100 --eps 4 8 --n_qual 10
uv run python ExplanationBaselineAttacksonIG.py --method cosine     --model convnext_base --max_images 100 --eps 4 8 --n_qual 10

"""

import argparse
import csv
import importlib.util
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))

# Shared machinery (build_model, get_expl_ig, metrics, plots, data) imported from the GRILL-IG file
# so the IG computation, similarity metrics and figures are IDENTICAL -> fair head-to-head.
_spec = importlib.util.spec_from_file_location("grill", os.path.join(HERE, "ExplanationGrillAttackonIG.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)

METHODS = ["topk", "dombrowski", "masscenter", "cosine"]
DEFAULT_PREFACTORS = {
    "topk": [100.0, 1e6],
    "dombrowski": [1e11, 1e6],
    "masscenter": [1e4, 1e6],
    "cosine": [100.0, 1e6],
}


# --------------------------------------------------------------------------- #
# The baseline attack (all four losses; faithful to the single-model originals)
# --------------------------------------------------------------------------- #
def baseline_attack(method, model, x, eps, steps, lr, prefactors, num_summands,
                    beta_modules, beta_growth, topk):
    G.change_beta(beta_modules, 1000.0 if beta_growth else None)
    org_expl, org_acc, org_idx = G.get_expl_ig(model, x, None, num_summands)
    org_expl, org_acc = org_expl.detach(), org_acc.detach()
    B = x.shape[0]

    # method-specific references + start
    if method == "topk":
        mask = torch.zeros_like(org_expl).flatten(1)
        mask.scatter_(1, org_expl.flatten(1).topk(topk, dim=1).indices, 1.0)
        mask = mask.view_as(org_expl)
        org_topk_mass = (org_expl * mask).flatten(1).sum(1).clamp_min(1e-12)   # [B]
        delta = torch.zeros_like(x, requires_grad=True)                        # grad != 0 at delta=0
    else:
        delta = (torch.rand_like(x) * 2 - 1) * eps                             # random start in eps-ball
        delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    if method == "masscenter":
        H, W = org_expl.shape[-2], org_expl.shape[-1]
        ys = torch.linspace(0, 1, H, device=x.device).view(1, H, 1)
        xs = torch.linspace(0, 1, W, device=x.device).view(1, 1, W)
        org_center = torch.stack([(org_expl * ys).flatten(1).sum(1),
                                  (org_expl * xs).flatten(1).sum(1)], dim=1)    # [B,2]
    if method == "cosine":
        org_flat = org_expl.flatten(1)

    opt = torch.optim.Adam([delta], lr=lr)
    best_adv = x.clone()
    best_loss = torch.full((B,), float("inf"), device=x.device)

    for step in range(steps + 1):
        if beta_growth:
            G.change_beta(beta_modules, G.get_beta(step, steps))
        adv_expl, adv_acc, class_idx = G.get_expl_ig(model, x + delta, org_idx,
                                                     num_summands, create_graph=True)
        if method == "topk":
            loss_expl = (adv_expl * mask).flatten(1).sum(1) / org_topk_mass
        elif method == "dombrowski":
            loss_expl = -((adv_expl - org_expl) ** 2).flatten(1).mean(1)
        elif method == "masscenter":
            adv_center = torch.stack([(adv_expl * ys).flatten(1).sum(1),
                                      (adv_expl * xs).flatten(1).sum(1)], dim=1)
            loss_expl = -((adv_center - org_center) ** 2).sum(1)
        else:  # cosine
            loss_expl = F.cosine_similarity(adv_expl.flatten(1), org_flat, dim=1)
        loss_output = ((adv_acc - org_acc) ** 2).mean(1)
        loss = prefactors[0] * loss_expl + prefactors[1] * loss_output         # minimised

        with torch.no_grad():
            keep = (class_idx == org_idx) & (loss_expl < best_loss)
            best_adv[keep] = (x + delta)[keep].detach()
            best_loss[keep] = loss_expl[keep]

        if step == steps:
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
# One (method, model) run over all eps -- same reporting/plotting as GRILL-IG
# --------------------------------------------------------------------------- #
def run_one(method, args, model, pixel_tf, categories, beta_modules, dataset, device):
    prefactors = args.prefactors if args.prefactors is not None else DEFAULT_PREFACTORS[method]
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0
    out_dir = args.out_dir or f"results/baseline_{method}_{args.model}"
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    mnames = ["mse", "pcc", "ssim", "spearman", "topk"]
    qual_dir = os.path.join(out_dir, "qualitative")
    quant_dir = os.path.join(out_dir, "quantitative")
    n_qual = 0 if args.no_plots else min(args.n_qual, len(dataset))
    import random as _random
    qual_names = set(dataset.files[i] for i in _random.Random(args.seed).sample(range(len(dataset)), n_qual))

    print(f"\n########## method={method} | model={args.model} | prefactors={prefactors} | "
          f"beta growth {'on' if beta_growth else 'off'} | out={out_dir} ##########")

    summary = []
    for eps_255 in args.eps:
        eps = eps_255 / 255.0
        lr = args.lr_factor * eps
        print(f"\n{'=' * 70}\n{method} | eps = {eps_255}/255   (Adam lr = {lr:.6f})\n{'=' * 70}")
        rows, store, qual_samples, cols = [], {}, [], {}
        n = n_clean = n_unchanged = 0
        tag = f"eps{eps_255:g}".replace(".", "p")
        t0 = time.time()

        for b, (x, y, names) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            x_adv, org_idx = baseline_attack(method, model, x, eps, args.steps, lr, prefactors,
                                             args.num_summands, beta_modules, beta_growth, args.topk)
            h_o, acc_o, pred_o = G.get_expl_ig(model, x, num_summands=args.num_summands)
            h_a, acc_a, pred_a = G.get_expl_ig(model, x_adv, num_summands=args.num_summands)
            h_o, h_a = h_o.detach(), h_a.detach()
            with torch.no_grad():
                unchanged = pred_a == pred_o
                out_change = (acc_a - acc_o).norm(dim=1)
                linf = (x_adv - x).abs().flatten(1).max(1).values * 255
                img_mse = ((x_adv - x) ** 2).flatten(1).mean(1)
                img_ssim = G.ssim(x_adv, x)
                res = G.expl_metrics(h_a, h_o, args.topk)

            n += len(y); n_clean += (pred_o == y).sum().item(); n_unchanged += unchanged.sum().item()
            for i in range(len(y)):
                row = {"filename": names[i], "true_idx": y[i].item(),
                       "clean_pred_idx": pred_o[i].item(), "adv_pred_idx": pred_a[i].item(),
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
                if args.save_expl:
                    store[names[i]] = {"x_adv": x_adv[i].cpu().half(),
                                       "h_orig": h_o[i].cpu(), "h_adv": h_a[i].cpu()}
                if names[i] in qual_names:
                    stem = os.path.splitext(names[i])[0]
                    G.save_qualitative(os.path.join(qual_dir, tag, stem), x[i], x_adv[i], h_o[i], h_a[i],
                                       caption=f"{names[i]} | {method} | eps={eps_255}/255 | {args.model} | "
                                               f"pred {categories[pred_o[i].item()]} -> "
                                               f"{categories[pred_a[i].item()]} | "
                                               f"PCC(clean,pert)={row['adv_vs_orig_pcc']:.3f}")
                    qual_samples.append({"name": names[i], "x": x[i].cpu(), "x_adv": x_adv[i].cpu(),
                                         "h_o": h_o[i].cpu(), "h_a": h_a[i].cpu(),
                                         "pred_o": categories[pred_o[i].item()],
                                         "pred_a": categories[pred_a[i].item()],
                                         "pcc_ao": row["adv_vs_orig_pcc"]})
            print(f"  batch {b + 1:3d}/{len(loader)} | label unchanged {n_unchanged / n:.3f} | "
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
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if args.save_expl:
            torch.save(store, os.path.join(out_dir, f"baseline_{method}_{tag}.pt"))
        summary.append({"method": method, "model": args.model, "eps_x255": eps_255, "n_images": n,
                        "clean_acc": n_clean / n, "decision_unchanged": n_unchanged / n,
                        "seconds": round(elapsed, 1),
                        **{f"median_{k}": v for k, v in med.items()},
                        **{f"mean_{k}": v for k, v in mean.items()}})

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    if not args.no_plots:
        G.save_quantitative_summary(quant_dir, summary)
    print(f"\n{method}/{args.model} ->")
    print(f"{'eps(/255)':>10}{'unchanged':>11}{'PCC a-o':>9}{'Spear a-o':>11}{'top-k a-o':>11}")
    for s in summary:
        print(f"{s['eps_x255']:>10g}{s['decision_unchanged']:>11.3f}{s['median_adv_vs_orig_pcc']:>9.3f}"
              f"{s['median_adv_vs_orig_spearman']:>11.3f}{s['median_adv_vs_orig_topk']:>11.3f}")
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", nargs="+", default=METHODS, choices=METHODS,
                   help="one or more of: topk dombrowski masscenter cosine (default all four)")
    p.add_argument("--model", default="convnext_base",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default=None, help="default results/baseline_<method>_<model>")
    p.add_argument("--eps", type=float, nargs="+", default=[4.0, 8.0], help="L_inf in 1/255")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1, help="Adam lr = lr_factor * eps")
    p.add_argument("--prefactors", nargs=2, type=float, default=None,
                   help="override (expl, output) weights; default is per-method (see --help)")
    p.add_argument("--num_summands", type=int, default=30)
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_expl", action="store_true")
    p.add_argument("--n_qual", type=int, default=10)
    p.add_argument("--grid_rows", type=int, default=5)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--no_beta_growth", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    G.set_seed(args.seed)
    device = torch.device(args.device)
    model, pixel_tf, categories, beta_modules, crop = G.build_model(
        args.model, device, check_mha=(args.model == "vit_b_16"))
    dataset = G.LabelledImages(args.data_dir, G.load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) < 1:
        raise SystemExit("No labelled images - check --data_dir and --labels.")
    print(f"Model: {args.model} | device: {device} | images: {len(dataset)} | methods: {args.method} | "
          f"steps: {args.steps} | summands: {args.num_summands} | eps: {args.eps}")

    all_summ = []
    for method in args.method:
        if args.out_dir and len(args.method) > 1:
            raise SystemExit("--out_dir with multiple methods would collide; run one --method at a time.")
        all_summ += run_one(method, args, model, pixel_tf, categories, beta_modules, dataset, device)

    if len(args.method) > 1:
        os.makedirs("results", exist_ok=True)
        combo = os.path.join("results", f"baseline_ALL_{args.model}_summary.csv")
        with open(combo, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_summ[0].keys())); w.writeheader(); w.writerows(all_summ)
        print(f"\nCombined summary -> {combo}")


if __name__ == "__main__":
    main()
