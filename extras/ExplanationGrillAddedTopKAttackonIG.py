"""
GRILL-augmented top-k attack on Integrated Gradients (ViT-B/16 / ResNet-50).

Idea (the repaired GRILL-IG): take the STRONGEST terminal baseline -- Ghorbani IFIA top-k on the
INPUT-level IG map -- and ADD the advanced-GRILL coupling of intermediate Layer-IG distortions as an
auxiliary term weighted by lambda. The top-k term directly spoils the measured (input) explanation;
the GRILL term injects the extra, less-degraded intermediate-layer gradient signal. lambda=0
reproduces the pure top-k baseline EXACTLY (same start, same objective, same best-iterate rule), so
this is a clean nested ablation: sweep lambda and see whether any lambda>0 beats lambda=0.

    minimise   pref[0] * L_topk(input IG)  -  lambda * L_couple(intermediate Layer-IG)  +  pref[1]*MSE(out)
      L_topk    = sum_{clean top-k}(h_adv) / clean_top-k_mass        (>=0, minimise -> drain top-k mass)
      L_couple  = sum_{i<j} d_i d_j,  d_k=||LayerIG_k(x_adv)-LayerIG_k(x)||_2   (advanced GRILL; maximise)
      start: delta = 0 (top-k gradient is non-zero there; GRILL term switches on once delta!=0)
      best iterate: min L_topk among class-preserving  (so lambda only changes the trajectory, and the
                    comparison is judged on the measured target -> lambda=0 == top-k baseline)

Why small eps: at eps 4/8 the top-k baseline already saturates (IG PCC ~0, top-k intersection 0), so
there's no headroom for lambda to show a gain. At small eps (1,2,4) the baseline has NOT saturated,
so a real lambda>0 benefit -- if it exists on these (undefended) models -- can appear. Judge on PCC /
Spearman (not top-k intersection, which saturates first).

Reuses build_model (multi-model, ProbeMHA), get_expl_ig, get_layer_ig, metrics and plots from
ExplanationGrillAttackonIG.py, so results are directly comparable to results/ig_grill_attack_<model>/
and results/baseline_topk_<model>/. ConvNeXt is intentionally dropped (slow); --model still allows it.

Outputs -> results/grilltopk_<model>_lam<lambda>/  (lambda in the folder name -> a sweep never
    overwrites, and nothing clashes with existing dirs). Same files/format as the other attacks.

Usage
-----
    # best baseline (top-k) at small eps, for reference  (lambda = 0):
    uv run python ExplanationGrillAddedTopKAttackonIG.py --model vit_b_16 --lambda_grill 0 --eps 1 2 4 --max_images 100 --n_qual 10
    # augmented, one lambda per run (sweep log-spaced; put each on its own GPU):
    for L in 0.001 0.01 0.1 1 10; do
      uv run python ExplanationGrillAddedTopKAttackonIG.py --model vit_b_16 --lambda_grill $L --eps 1 2 4 --max_images 100
    done
    #   repeat with --model resnet50.  Also run the OTHER baselines at matched eps for the full comparison:
    #   uv run python ExplanationBaselineAttacksonIG.py --method cosine dombrowski masscenter --model vit_b_16 --eps 1 2 4 --max_images 100
    (eps in units of 1/255. L_topk ~ O(1), L_couple ~ O(1-10): sweep lambda log-spaced, e.g. 1e-3..1e2.)


# --- ResNet-50 first (cheaper). lambda=0 is the top-k baseline reference at small eps ---

export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model resnet50 --lambda_grill 0    --eps 0.5 1 2 --max_images 100 --n_qual 10

export CUDA_VISIBLE_DEVICES=1
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model resnet50 --lambda_grill 0.01 --eps 0.5 1 2 --max_images 100

export CUDA_VISIBLE_DEVICES=2
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model resnet50 --lambda_grill 0.1  --eps 0.5 1 2 --max_images 100

export CUDA_VISIBLE_DEVICES=3
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model resnet50 --lambda_grill 1    --eps 0.5 1 2 --max_images 100
# (add --lambda_grill 10 if a GPU is free)

# --- ViT-B/16 (expensive: ~2x baseline/step). Trim to keep it tractable for a first pass ---
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model vit_b_16 --lambda_grill 0    --eps 0.5 1 2 --max_images 50

export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model vit_b_16 --lambda_grill 0.1  --eps 0.5 1 2 --max_images 50

export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model vit_b_16 --lambda_grill 0.01  --eps 0.5 1 2 --max_images 50

export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationGrillAddedTopKAttackonIG.py --model vit_b_16 --lambda_grill 1    --eps 0.5 1 2 --max_images 50

# --- optional: the other three baselines at small eps, for the full comparison table ---
uv run python ExplanationBaselineAttacksonIG.py --method cosine dombrowski masscenter --model resnet50 --eps 1 2 4 --max_images 100

"""

import argparse
import csv
import importlib.util
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("grill", os.path.join(HERE, "ExplanationGrillAttackonIG.py"))
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def fmt_lambda(l):
    return f"{l:g}".replace("-", "m").replace(".", "p").replace("+", "")


# --------------------------------------------------------------------------- #
# top-k (input IG)  +  lambda * advanced-GRILL (intermediate Layer-IG)
# --------------------------------------------------------------------------- #
def grill_topk_attack(model, x, eps, steps, lr, prefactors, lambda_grill, num_summands,
                      coupling_layers, recorders, baseline_acts, beta_modules, beta_growth, topk):
    G.change_beta(beta_modules, 1000.0 if beta_growth else None)
    names = [n for n, *_ in coupling_layers]
    use_grill = lambda_grill != 0.0

    org_expl, org_acc, org_idx = G.get_expl_ig(model, x, None, num_summands)   # input-level IG
    org_expl, org_acc = org_expl.detach(), org_acc.detach()
    B = x.shape[0]

    mask = torch.zeros_like(org_expl).flatten(1)
    mask.scatter_(1, org_expl.flatten(1).topk(topk, dim=1).indices, 1.0)
    mask = mask.view_as(org_expl)
    org_topk_mass = (org_expl * mask).flatten(1).sum(1).clamp_min(1e-12)        # [B]

    if use_grill:
        clean_ligs, _, _ = G.get_layer_ig(model, x, None, coupling_layers, recorders,
                                          baseline_acts, num_summands, False)
        clean_ligs = {k: v.detach() for k, v in clean_ligs.items()}

    delta = torch.zeros_like(x, requires_grad=True)                            # top-k start (== baseline)
    opt = torch.optim.Adam([delta], lr=lr)
    best_adv = x.clone()
    best_loss = torch.full((B,), float("inf"), device=x.device)                # min L_topk (the target)

    for step in range(steps + 1):
        if beta_growth:
            G.change_beta(beta_modules, G.get_beta(step, steps))
        adv_expl, adv_acc, class_idx = G.get_expl_ig(model, x + delta, org_idx,
                                                     num_summands, create_graph=True)
        L_topk = (adv_expl * mask).flatten(1).sum(1) / org_topk_mass           # [B], minimise
        loss = prefactors[0] * L_topk + prefactors[1] * ((adv_acc - org_acc) ** 2).mean(1)
        if use_grill:
            adv_ligs, _, _ = G.get_layer_ig(model, x + delta, org_idx, coupling_layers, recorders,
                                            baseline_acts, num_summands, create_graph=True)
            D = torch.stack([((adv_ligs[n] - clean_ligs[n]) ** 2).flatten(1).sum(1).clamp_min(1e-12).sqrt()
                             for n in names], dim=0)                            # [K, B]
            L_couple = 0.5 * (D.sum(0) ** 2 - (D ** 2).sum(0))                  # advanced GRILL, maximise
            loss = loss - lambda_grill * L_couple

        with torch.no_grad():
            keep = (class_idx == org_idx) & (L_topk < best_loss)
            best_adv[keep] = (x + delta)[keep].detach()
            best_loss[keep] = L_topk[keep]

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


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_b_16",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--lambda_grill", type=float, default=0.0,
                   help="weight of the advanced-GRILL term (0 == pure top-k baseline)")
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default=None,
                   help="default results/grilltopk_<model>_lam<lambda>")
    p.add_argument("--eps", type=float, nargs="+", default=[1.0, 2.0, 4.0], help="L_inf in 1/255")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1)
    p.add_argument("--prefactors", nargs=2, type=float, default=[100.0, 1e6],
                   help="(top-k weight, output-MSE weight); top-k baseline default [100, 1e6]")
    p.add_argument("--num_summands", type=int, default=30)
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--layers", nargs="+", default=None, help="subset of coupling-layer names")
    p.add_argument("--no_beta_growth", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_expl", action="store_true")
    p.add_argument("--n_qual", type=int, default=10)
    p.add_argument("--grid_rows", type=int, default=5)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    G.set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir or f"results/grilltopk_{args.model}_lam{fmt_lambda(args.lambda_grill)}"
    os.makedirs(out_dir, exist_ok=True)

    model, pixel_tf, categories, beta_modules, crop = G.build_model(
        args.model, device, check_mha=(args.model == "vit_b_16"))
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0

    coupling_layers = G.select_coupling_layers(args.model, model.model)
    if args.layers:
        wanted = set(args.layers)
        coupling_layers = [c for c in coupling_layers if c[0] in wanted]
    if args.lambda_grill != 0.0 and len(coupling_layers) < 2:
        raise SystemExit(f"advanced-GRILL coupling needs >=2 layers (got {len(coupling_layers)}); "
                         f"broaden --layers or use lambda_grill 0.")
    recorders, handles = G.attach_recorders(coupling_layers, retain=False)
    baseline_acts = G.precompute_baseline_acts(model, coupling_layers, recorders, device, crop)

    dataset = G.LabelledImages(args.data_dir, G.load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) < 1:
        raise SystemExit("No labelled images - check --data_dir and --labels.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    print(f"Model: {args.model} | lambda_grill: {args.lambda_grill} "
          f"({'top-k + GRILL' if args.lambda_grill != 0 else 'PURE top-k baseline'}) | "
          f"device: {device} | images: {len(dataset)} | eps: {args.eps} | steps: {args.steps} | "
          f"prefactors: {args.prefactors} | beta growth: {'on' if beta_growth else 'off'}")
    print(f"Coupling layers ({len(coupling_layers)}): {[c[0] for c in coupling_layers]}")
    print(f"out_dir: {out_dir}")

    mnames = ["mse", "pcc", "ssim", "spearman", "topk"]
    qual_dir, quant_dir = os.path.join(out_dir, "qualitative"), os.path.join(out_dir, "quantitative")
    import random as _random
    n_qual = 0 if args.no_plots else min(args.n_qual, len(dataset))
    qual_names = set(dataset.files[i] for i in _random.Random(args.seed).sample(range(len(dataset)), n_qual))

    summary = []
    for eps_255 in args.eps:
        eps = eps_255 / 255.0
        lr = args.lr_factor * eps
        print(f"\n{'=' * 70}\nlambda={args.lambda_grill} | eps = {eps_255}/255   (lr={lr:.6f})\n{'=' * 70}")
        rows, store, qual_samples, cols = [], {}, [], {}
        n = n_clean = n_unchanged = 0
        tag = f"eps{eps_255:g}".replace(".", "p")
        t0 = time.time()

        for b, (x, y, names_) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            x_adv, org_idx = grill_topk_attack(
                model, x, eps, args.steps, lr, args.prefactors, args.lambda_grill, args.num_summands,
                coupling_layers, recorders, baseline_acts, beta_modules, beta_growth, args.topk)
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
                row = {"filename": names_[i], "true_idx": y[i].item(),
                       "clean_pred_idx": pred_o[i].item(), "adv_pred_idx": pred_a[i].item(),
                       "clean_correct": int(pred_o[i].item() == y[i].item()),
                       "label_unchanged": int(unchanged[i].item()),
                       "output_change_l2": out_change[i].item(), "linf_x255": linf[i].item(),
                       "img_mse": img_mse[i].item(), "img_ssim": img_ssim[i].item()}
                for m in mnames:
                    row[f"adv_vs_orig_{m}"] = res[m][i].item()
                rows.append(row)
                for kname in ["output_change_l2", "linf_x255", "img_mse", "img_ssim"] + \
                             [f"adv_vs_orig_{m}" for m in mnames]:
                    cols.setdefault(kname, []).append(row[kname])
                if args.save_expl:
                    store[names_[i]] = {"x_adv": x_adv[i].cpu().half(),
                                        "h_orig": h_o[i].cpu(), "h_adv": h_a[i].cpu()}
                if names_[i] in qual_names:
                    stem = os.path.splitext(names_[i])[0]
                    G.save_qualitative(os.path.join(qual_dir, tag, stem), x[i], x_adv[i], h_o[i], h_a[i],
                                       caption=f"{names_[i]} | top-k+{args.lambda_grill}*GRILL | "
                                               f"eps={eps_255}/255 | {args.model} | "
                                               f"PCC(clean,pert)={row['adv_vs_orig_pcc']:.3f}")
                    qual_samples.append({"name": names_[i], "x": x[i].cpu(), "x_adv": x_adv[i].cpu(),
                                         "h_o": h_o[i].cpu(), "h_a": h_a[i].cpu(),
                                         "pred_o": categories[pred_o[i].item()],
                                         "pred_a": categories[pred_a[i].item()],
                                         "pcc_ao": row["adv_vs_orig_pcc"]})
            print(f"  batch {b + 1:3d}/{len(loader)} | unchanged {n_unchanged / n:.3f} | "
                  f"median PCC(adv,orig) {np.median(cols['adv_vs_orig_pcc']):.3f} | "
                  f"median top-k {np.median(cols['adv_vs_orig_topk']):.3f}")

        elapsed = time.time() - t0
        med = {k: float(np.median(v)) for k, v in cols.items()}
        mean = {k: float(np.mean(v)) for k, v in cols.items()}
        print(f"\n  Images {n} ({elapsed:.1f}s) | clean acc {n_clean / n:.4f} | unchanged {n_unchanged / n:.4f}")
        print(f"  adv vs orig IG : PCC {med['adv_vs_orig_pcc']:.3f}  Spearman {med['adv_vs_orig_spearman']:.3f}  "
              f"top-k {med['adv_vs_orig_topk']:.3f}")
        if not args.no_plots:
            G.save_quantitative_eps(os.path.join(quant_dir, tag), cols, eps_255, mnames)
            G.save_qualitative_grid(os.path.join(qual_dir, tag, "grid"), qual_samples, args.grid_rows)
        with open(os.path.join(out_dir, f"per_image_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if args.save_expl:
            torch.save(store, os.path.join(out_dir, f"grilltopk_{tag}.pt"))
        summary.append({"model": args.model, "lambda_grill": args.lambda_grill, "eps_x255": eps_255,
                        "n_images": n, "clean_acc": n_clean / n, "decision_unchanged": n_unchanged / n,
                        "seconds": round(elapsed, 1),
                        **{f"median_{k}": v for k, v in med.items()},
                        **{f"mean_{k}": v for k, v in mean.items()}})

    for h in handles:
        h.remove()
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    if not args.no_plots:
        G.save_quantitative_summary(quant_dir, summary)
    print(f"\n{args.model} | lambda={args.lambda_grill} ->")
    print(f"{'eps(/255)':>10}{'unchanged':>11}{'PCC a-o':>9}{'Spear a-o':>11}{'top-k a-o':>11}")
    for s in summary:
        print(f"{s['eps_x255']:>10g}{s['decision_unchanged']:>11.3f}{s['median_adv_vs_orig_pcc']:>9.3f}"
              f"{s['median_adv_vs_orig_spearman']:>11.3f}{s['median_adv_vs_orig_topk']:>11.3f}")
    print(f"\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
