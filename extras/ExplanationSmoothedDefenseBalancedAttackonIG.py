"""
GRILL-augmented top-k attack vs the SMOOTHED-IG DEFENSE, with GRADIENT BALANCING (ViT / ResNet).

Why this file: the fixed-lambda sweep (ExplanationSmoothedDefenseAttackonIG.py) showed no effect, but
that was a SCALE artifact -- the advanced-GRILL term L_couple = sum_{i<j} d_i d_j is a pairwise product
of small sum-to-1 map distances, so its gradient is ~1e6x smaller than the top-k(+output) gradient.
lambda <= 50 was 4-6 orders of magnitude too weak to register. This file removes the scale confound:

    --grill_balance c   (per step) scale the GRILL gradient to magnitude c * ||g_topk||, i.e. the
                         effective update is   g = g_topk  -  c * (||g_topk|| / ||g_couple||) * g_couple.
                         c=1 makes GRILL's push exactly as strong as the top-k push; c=2 twice as strong.
                         c=0 reproduces the pure top-k baseline EXACTLY (nested).
    --lambda_grill L     (kept for reference) old fixed-weight behaviour: g = g_topk - L * g_couple.

This directly tests whether GRILL's DIRECTION helps, independent of its raw magnitude. If even a balanced
GRILL (its step as large as top-k's) fails to lower the smoothed-IG PCC below the c=0 baseline, that is
the definitive null and it isolates the proxy-misalignment question (GRILL optimises UNsmoothed
intermediate Layer-IG; the metric/selection is the SMOOTHED input IG) from the scaling artifact.

Everything else matches ExplanationSmoothedDefenseAttackonIG.py: SmoothGrad defense (mean of N IG over
Gaussian noise sigma, fixed draw), top-k on the smoothed input IG, advanced-GRILL on the unsmoothed
intermediate Layer-IG, hard class-preserving constraint, best iterate = min L_topk among class-preserving
(so c=0 == pure top-k). The per-batch line prints ||g_topk||, ||g_couple|| and the applied scale so you
can watch GRILL actually competing. Reuses build_model/get_expl_ig/get_layer_ig/metrics/plots from
ExplanationGrillAttackonIG.py; smoothed IG batches the N samples into one forward.

Outputs -> results/smoothdefbal_<model>_{bal<c>|lam<L>}_sig<sigma>_N<n_smooth>_img<#samples>/
    (distinct prefix 'smoothdefbal_' + the mode in the name -> never overwrites the lambda-only runs).

Usage (ResNet, sigma=0.1 is the biting-but-open regime; small eps; 100 samples)
-----
    # c=0 is the pure top-k baseline; then sweep the balance factor:
    uv run python ExplanationSmoothedDefenseBalancedAttackonIG.py --model resnet50 --grill_balance 0    --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100
    for C in 0.25 0.5 1 2; do
      uv run python ExplanationSmoothedDefenseBalancedAttackonIG.py --model resnet50 --grill_balance $C --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100
    done
    # win = median smoothed-IG PCC clearly below the c=0 value, class kept. (eps in 1/255; sigma in [0,1].)


# c=0 is the pure top-k baseline (reproduces your ~0.978); then the balanced sweep:

export CUDA_VISIBLE_DEVICES=2
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationSmoothedDefenseBalancedAttackonIG.py --model resnet50 --grill_balance 0 --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100

export CUDA_VISIBLE_DEVICES=3
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationSmoothedDefenseBalancedAttackonIG.py --model resnet50 --grill_balance 0.5 --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100

export CUDA_VISIBLE_DEVICES=4
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationSmoothedDefenseBalancedAttackonIG.py --model resnet50 --grill_balance 1 --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100

export CUDA_VISIBLE_DEVICES=5
cd /data1/chethan/robustExAI
source .venv/bin/activate
uv run python ExplanationSmoothedDefenseBalancedAttackonIG.py --model resnet50 --grill_balance 2 --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100
    
    

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


def fmt(v):
    return f"{v:g}".replace("-", "m").replace(".", "p").replace("+", "")


def get_expl_ig_smooth(model, x, desired_index, num_summands, noise, create_graph=False):
    """SmoothGrad-defended IG: mean over N fixed Gaussian-noised copies (batched). [B,*] sum-to-1."""
    N, B = noise.shape[0], x.shape[0]
    xn = (x.unsqueeze(0) + noise).reshape(N * B, *x.shape[1:])
    di = None if desired_index is None else desired_index.repeat(N)
    h, _, _ = G.get_expl_ig(model, xn, di, num_summands, create_graph=create_graph)
    return h.view(N, B, *h.shape[1:]).mean(0)


def balanced_attack(model, x, eps, steps, lr, prefactors, lambda_grill, grill_balance, num_summands,
                    noise, coupling_layers, recorders, baseline_acts, beta_modules, beta_growth, topk):
    """Returns best_adv, org_idx, org_heat(smoothed), (last ||g_topk||, ||g_couple||, applied scale)."""
    G.change_beta(beta_modules, 1000.0 if beta_growth else None)
    names = [n for n, *_ in coupling_layers]
    balance = grill_balance != 0.0
    use_grill = balance or (lambda_grill != 0.0)
    B = x.shape[0]

    with torch.no_grad():
        logits0 = model(x)
        org_acc, org_idx = torch.softmax(logits0, dim=1), logits0.argmax(dim=1)
    org_heat = get_expl_ig_smooth(model, x, org_idx, num_summands, noise, create_graph=False).detach()
    mask = torch.zeros_like(org_heat).flatten(1)
    mask.scatter_(1, org_heat.flatten(1).topk(topk, dim=1).indices, 1.0)
    mask = mask.view_as(org_heat)
    org_topk_mass = (org_heat * mask).flatten(1).sum(1).clamp_min(1e-12)
    if use_grill:
        clean_ligs, _, _ = G.get_layer_ig(model, x, org_idx, coupling_layers, recorders,
                                          baseline_acts, num_summands, False)
        clean_ligs = {k: v.detach() for k, v in clean_ligs.items()}

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    best_adv = x.clone()
    best_loss = torch.full((B,), float("inf"), device=x.device)
    gt_n = gc_n = sc_n = 0.0

    for step in range(steps + 1):
        if beta_growth:
            G.change_beta(beta_modules, G.get_beta(step, steps))

        adv_heat = get_expl_ig_smooth(model, x + delta, org_idx, num_summands, noise, create_graph=True)
        L_topk = (adv_heat * mask).flatten(1).sum(1) / org_topk_mass
        logits = model(x + delta)
        adv_acc, class_idx = torch.softmax(logits, dim=1), logits.argmax(dim=1)
        topk_loss = prefactors[0] * L_topk + prefactors[1] * ((adv_acc - org_acc) ** 2).mean(1)

        with torch.no_grad():
            keep = (class_idx == org_idx) & (L_topk < best_loss)
            best_adv[keep] = (x + delta)[keep].detach()
            best_loss[keep] = L_topk[keep]
        if step == steps:
            break

        g_topk = torch.autograd.grad(topk_loss.sum(), delta, retain_graph=False)[0]
        if use_grill:
            adv_ligs, _, _ = G.get_layer_ig(model, x + delta, org_idx, coupling_layers, recorders,
                                            baseline_acts, num_summands, create_graph=True)
            D = torch.stack([((adv_ligs[n] - clean_ligs[n]) ** 2).flatten(1).sum(1).clamp_min(1e-12).sqrt()
                             for n in names], dim=0)
            L_couple = 0.5 * (D.sum(0) ** 2 - (D ** 2).sum(0))
            g_couple = torch.autograd.grad(L_couple.sum(), delta)[0]
            if balance:
                scale = grill_balance * (g_topk.norm() / (g_couple.norm() + 1e-12))
            else:
                scale = torch.as_tensor(float(lambda_grill), device=x.device)
            eff = g_topk - scale * g_couple               # minimise topk_loss, maximise L_couple
            gt_n, gc_n, sc_n = float(g_topk.norm()), float(g_couple.norm()), float(scale)
        else:
            eff = g_topk
            gt_n = float(g_topk.norm())

        opt.zero_grad(set_to_none=True)
        delta.grad = eff
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x + delta, 0, 1) - x)

    G.change_beta(beta_modules, None)
    return best_adv, org_idx, org_heat, (gt_n, gc_n, sc_n)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet50", choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--grill_balance", type=float, default=0.0,
                   help="per-step GRILL gradient magnitude as a fraction of ||g_topk|| (0 = pure top-k). "
                        "Overrides --lambda_grill when nonzero.")
    p.add_argument("--lambda_grill", type=float, default=0.0, help="old fixed-weight mode (used only if "
                                                                    "--grill_balance 0)")
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n_smooth", type=int, default=8)
    p.add_argument("--noise_seed", type=int, default=1234)
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--eps", type=float, nargs="+", default=[1.0])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1)
    p.add_argument("--prefactors", nargs=2, type=float, default=[100.0, 1e6])
    p.add_argument("--num_summands", type=int, default=30)
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--layers", nargs="+", default=None)
    p.add_argument("--no_beta_growth", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=100,
                   help="#samples (default 100; the count is written into the folder name as img<count>).")
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
    model, pixel_tf, categories, beta_modules, crop = G.build_model(
        args.model, device, check_mha=(args.model == "vit_b_16"))
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0
    coupling_layers = G.select_coupling_layers(args.model, model.model)
    if args.layers:
        wanted = set(args.layers)
        coupling_layers = [c for c in coupling_layers if c[0] in wanted]
    balance = args.grill_balance != 0.0
    if (balance or args.lambda_grill != 0.0) and len(coupling_layers) < 2:
        raise SystemExit(f"advanced-GRILL coupling needs >=2 layers (got {len(coupling_layers)}).")
    recorders, handles = G.attach_recorders(coupling_layers, retain=False)
    baseline_acts = G.precompute_baseline_acts(model, coupling_layers, recorders, device, crop)

    dataset = G.LabelledImages(args.data_dir, G.load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) < 1:
        raise SystemExit("No labelled images - check --data_dir and --labels.")
    mode_tag = f"bal{fmt(args.grill_balance)}" if balance else f"lam{fmt(args.lambda_grill)}"
    out_dir = args.out_dir or (f"results/smoothdefbal_{args.model}_{mode_tag}"
                               f"_sig{fmt(args.sigma)}_N{args.n_smooth}_img{len(dataset)}")
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    noise_gen = torch.Generator(device=device).manual_seed(args.noise_seed)

    mode_desc = (f"GRADIENT-BALANCED c={args.grill_balance}" if balance else
                 (f"fixed lambda={args.lambda_grill}" if args.lambda_grill else "PURE top-k (no GRILL)"))
    print(f"Model: {args.model} | SMOOTHED-IG DEFENSE sigma={args.sigma} N={args.n_smooth} | {mode_desc} | "
          f"images: {len(dataset)} | eps: {args.eps} | steps: {args.steps} | beta growth: "
          f"{'on' if beta_growth else 'off'}")
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
        print(f"\n{'=' * 70}\n{mode_desc} sigma={args.sigma} | eps={eps_255}/255 (lr={lr:.6f})\n{'=' * 70}")
        rows, store, qual_samples, cols = [], {}, [], {}
        n = n_clean = n_unchanged = 0
        tag = f"eps{eps_255:g}".replace(".", "p")
        t0 = time.time()
        last_gn = (0.0, 0.0, 0.0)

        for b, (x, y, names_) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            noise = torch.randn((args.n_smooth,) + tuple(x.shape), device=device,
                                generator=noise_gen) * args.sigma
            x_adv, org_idx, h_o, gn = balanced_attack(
                model, x, eps, args.steps, lr, args.prefactors, args.lambda_grill, args.grill_balance,
                args.num_summands, noise, coupling_layers, recorders, baseline_acts,
                beta_modules, beta_growth, args.topk)
            last_gn = gn
            h_a = get_expl_ig_smooth(model, x_adv, org_idx, args.num_summands, noise, create_graph=False).detach()
            h_o = h_o.detach()
            with torch.no_grad():
                l_o, l_a = model(x), model(x_adv)
                pred_o, pred_a = l_o.argmax(1), l_a.argmax(1)
                unchanged = pred_a == pred_o
                out_change = (torch.softmax(l_a, 1) - torch.softmax(l_o, 1)).norm(dim=1)
                linf = (x_adv - x).abs().flatten(1).max(1).values * 255
                img_ssim = G.ssim(x_adv, x)
                res = G.expl_metrics(h_a, h_o, args.topk)
            n += len(y); n_clean += (pred_o == y).sum().item(); n_unchanged += unchanged.sum().item()
            for i in range(len(y)):
                row = {"filename": names_[i], "true_idx": y[i].item(),
                       "clean_pred_idx": pred_o[i].item(), "adv_pred_idx": pred_a[i].item(),
                       "clean_correct": int(pred_o[i].item() == y[i].item()),
                       "label_unchanged": int(unchanged[i].item()),
                       "output_change_l2": out_change[i].item(), "linf_x255": linf[i].item(),
                       "img_ssim": img_ssim[i].item()}
                for m in mnames:
                    row[f"adv_vs_orig_{m}"] = res[m][i].item()
                rows.append(row)
                for kname in ["output_change_l2", "linf_x255", "img_ssim"] + [f"adv_vs_orig_{m}" for m in mnames]:
                    cols.setdefault(kname, []).append(row[kname])
                if args.save_expl:
                    store[names_[i]] = {"x_adv": x_adv[i].cpu().half(), "h_orig": h_o[i].cpu(), "h_adv": h_a[i].cpu()}
                if names_[i] in qual_names:
                    stem = os.path.splitext(names_[i])[0]
                    G.save_qualitative(os.path.join(qual_dir, tag, stem), x[i], x_adv[i], h_o[i], h_a[i],
                                       caption=f"{names_[i]} | {mode_desc} sig{args.sigma} | eps={eps_255}/255 | "
                                               f"PCC(smoothed clean,pert)={row['adv_vs_orig_pcc']:.3f}")
                    qual_samples.append({"name": names_[i], "x": x[i].cpu(), "x_adv": x_adv[i].cpu(),
                                         "h_o": h_o[i].cpu(), "h_a": h_a[i].cpu(),
                                         "pred_o": categories[pred_o[i].item()],
                                         "pred_a": categories[pred_a[i].item()], "pcc_ao": row["adv_vs_orig_pcc"]})
            gt, gc, sc = last_gn
            print(f"  batch {b + 1:3d}/{len(loader)} | unchanged {n_unchanged / n:.3f} | "
                  f"median PCC {np.median(cols['adv_vs_orig_pcc']):.3f} | top-k {np.median(cols['adv_vs_orig_topk']):.3f}"
                  f" | ||g_topk||={gt:.2e} ||g_couple||={gc:.2e} scale={sc:.2e}")

        elapsed = time.time() - t0
        med = {k: float(np.median(v)) for k, v in cols.items()}
        mean = {k: float(np.mean(v)) for k, v in cols.items()}
        print(f"\n  Images {n} ({elapsed:.1f}s) | clean acc {n_clean / n:.4f} | unchanged {n_unchanged / n:.4f}")
        print(f"  SMOOTHED adv vs clean IG : PCC {med['adv_vs_orig_pcc']:.3f}  Spearman "
              f"{med['adv_vs_orig_spearman']:.3f}  top-k {med['adv_vs_orig_topk']:.3f}")
        print(f"  last-step grads: ||g_topk||={last_gn[0]:.3e}  ||g_couple||={last_gn[1]:.3e}  "
              f"applied scale={last_gn[2]:.3e}  (balanced: ||scale*g_couple||/||g_topk|| = "
              f"{(last_gn[2] * last_gn[1] / (last_gn[0] + 1e-12)):.3f})")
        if not args.no_plots:
            G.save_quantitative_eps(os.path.join(quant_dir, tag), cols, eps_255, mnames)
            G.save_qualitative_grid(os.path.join(qual_dir, tag, "grid"), qual_samples, args.grid_rows)
        with open(os.path.join(out_dir, f"per_image_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if args.save_expl:
            torch.save(store, os.path.join(out_dir, f"smoothdefbal_{tag}.pt"))
        summary.append({"model": args.model, "grill_balance": args.grill_balance,
                        "lambda_grill": args.lambda_grill, "sigma": args.sigma, "n_smooth": args.n_smooth,
                        "eps_x255": eps_255, "n_images": n, "clean_acc": n_clean / n,
                        "decision_unchanged": n_unchanged / n, "seconds": round(elapsed, 1),
                        "last_g_topk": last_gn[0], "last_g_couple": last_gn[1], "last_scale": last_gn[2],
                        **{f"median_{k}": v for k, v in med.items()},
                        **{f"mean_{k}": v for k, v in mean.items()}})

    for h in handles:
        h.remove()
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys())); w.writeheader(); w.writerows(summary)
    if not args.no_plots:
        G.save_quantitative_summary(quant_dir, summary)
    print(f"\n{args.model} | {mode_desc} sigma={args.sigma} ->")
    print(f"{'eps(/255)':>10}{'unchanged':>11}{'PCC a-o':>9}{'Spear a-o':>11}{'top-k a-o':>11}")
    for s in summary:
        print(f"{s['eps_x255']:>10g}{s['decision_unchanged']:>11.3f}{s['median_adv_vs_orig_pcc']:>9.3f}"
              f"{s['median_adv_vs_orig_spearman']:>11.3f}{s['median_adv_vs_orig_topk']:>11.3f}")
    print(f"\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
