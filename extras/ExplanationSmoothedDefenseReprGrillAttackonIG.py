"""
GRILL-NATIVE (representation) augmented top-k attack vs the SMOOTHED-IG DEFENSE (ViT / ResNet).

Unlike ExplanationSmoothedDefenseBalancedAttackonIG.py (which coupled intermediate ATTRIBUTION / Layer-IG
distortions and was rejected -- flat under lambda, worse under balancing), this file uses GRILL as in the
paper: couple distortions of the intermediate REPRESENTATIONS (activations a_k), nothing to do with
attributions.

    d_k = || a_k(x_adv) - a_k(x) || / || a_k(x) ||          (relative activation distortion at layer k)
    L_couple = sum_{i<j} d_i d_j                            (advanced-GRILL dense pairwise coupling)
    minimise  prefactors[0]*L_topk(smoothed input IG)  -  grill_sign * (scale) * L_couple  +  pref[1]*MSE(out)

Why this is the principled version:
  - FIRST-ORDER, healthy gradient. d_k uses activations directly (one forward), so dL_couple/ddelta is
    O(1) -- not the ~1e-5 double-backward of the attribution version that the lambda sweep couldn't move.
  - GRILL's native mechanism: restore attack EXPLORATION through the ill-conditioned early layers
    (ResNet layer1 are the near-null ones) when the primary (smoothed, starved) top-k gradient has stalled.
  - Cheap: the activations come for free from the model(x+delta) forward already done for the class term.

Open questions it still faces (be honest when reading results):
  - The metric & best-iterate selection are the smoothed INPUT IG; representation disruption helps only if
    it steers toward lower input-IG top-k mass (or jolts a stalled attack into a better basin). Still a proxy.
  - Representations drive the logits, so maximising feature change pushes the class to flip -> tension with
    the class-preserving constraint (the pref[1]*MSE(out) term fights it). A positive result means GRILL found
    class-preserving feature disruptions that also distort the IG.

Modes (same machinery as the balanced file):
    --grill_balance c   per-step GRILL gradient scaled to c*||g_topk|| (c=0 -> pure top-k baseline, nested).
    --grill_sign  +1/-1 +1 maximise representation distortion (GRILL as intended); -1 minimise (preserve).
    --lambda_grill L    fixed-weight mode (used only if --grill_balance 0); healthy here, so may actually bite.

Outputs -> results/smoothdefrepr_<model>_{bal<c>|lam<L>}[neg]_sig<sigma>_N<n_smooth>_img<#samples>/
Usage
-----
    uv run python ExplanationSmoothedDefenseReprGrillAttackonIG.py --model resnet50 --grill_balance 0   --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100
    for C in 0.5 1 2; do
      uv run python ExplanationSmoothedDefenseReprGrillAttackonIG.py --model resnet50 --grill_balance $C --sigma 0.1 --n_smooth 8 --eps 1 --max_images 100
    done
    # win = median smoothed-IG PCC below the c=0 baseline (~0.98 median) with class kept. (eps in 1/255.)


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python ExplanationSmoothedDefenseReprGrillAttackonIG.py --model resnet50 --grill_balance 0.5 --sigma 0.1 --n_smooth 4 --eps 1 --max_images 100


export CUDA_VISIBLE_DEVICES=2
cd /data1/chethan/robustExAI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python ExplanationSmoothedDefenseReprGrillAttackonIG.py --model resnet50 --grill_balance 0   --sigma 0.1 --n_smooth 4 --eps 1 --max_images 100


export CUDA_VISIBLE_DEVICES=3
cd /data1/chethan/robustExAI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python ExplanationSmoothedDefenseReprGrillAttackonIG.py --model resnet50 --grill_balance 0.5 --sigma 0.1 --n_smooth 4 --eps 1 --max_images 100


export CUDA_VISIBLE_DEVICES=4
cd /data1/chethan/robustExAI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python ExplanationSmoothedDefenseReprGrillAttackonIG.py --model resnet50 --grill_balance 1   --sigma 0.1 --n_smooth 4 --eps 1 --max_images 100


export CUDA_VISIBLE_DEVICES=5
cd /data1/chethan/robustExAI
source .venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv run python ExplanationSmoothedDefenseReprGrillAttackonIG.py --model resnet50 --grill_balance 2   --sigma 0.1 --n_smooth 4 --eps 1 --max_images 100


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
    N, B = noise.shape[0], x.shape[0]
    xn = (x.unsqueeze(0) + noise).reshape(N * B, *x.shape[1:])
    di = None if desired_index is None else desired_index.repeat(N)
    h, _, _ = G.get_expl_ig(model, xn, di, num_summands, create_graph=create_graph)
    return h.view(N, B, *h.shape[1:]).mean(0)


def repr_grill_attack(model, x, eps, steps, lr, prefactors, lambda_grill, grill_balance, grill_sign,
                      num_summands, noise, coupling_layers, recorders, beta_modules, beta_growth, topk):
    """GRILL on intermediate REPRESENTATIONS. Returns best_adv, org_idx, org_heat, (||g_topk||,||g_couple||,scale)."""
    G.change_beta(beta_modules, 1000.0 if beta_growth else None)
    names = [n for n, *_ in coupling_layers]
    balance = grill_balance != 0.0
    use_grill = balance or (lambda_grill != 0.0)
    B = x.shape[0]

    with torch.no_grad():
        logits0 = model(x)
        org_acc, org_idx = torch.softmax(logits0, dim=1), logits0.argmax(dim=1)
    # clean activations a_k(x) (detached), captured once
    for r in recorders.values():
        r.reset()
    with torch.no_grad():
        model(x)
    clean_act = {n: recorders[n].out.detach().clone() for n in names}

    org_heat = get_expl_ig_smooth(model, x, org_idx, num_summands, noise, create_graph=False).detach()
    mask = torch.zeros_like(org_heat).flatten(1)
    mask.scatter_(1, org_heat.flatten(1).topk(topk, dim=1).indices, 1.0)
    mask = mask.view_as(org_heat)
    org_topk_mass = (org_heat * mask).flatten(1).sum(1).clamp_min(1e-12)

    delta = torch.zeros_like(x, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    best_adv = x.clone()
    best_loss = torch.full((B,), float("inf"), device=x.device)
    gt_n = gc_n = sc_n = 0.0

    for step in range(steps + 1):
        if beta_growth:
            G.change_beta(beta_modules, G.get_beta(step, steps))

        # smoothed IG FIRST (heavy batch N*summands graph), nothing else held during it (matches the
        # balanced file's ordering / peak memory)
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

        # free the heavy smoothed-IG graph BEFORE the coupling forward (peak memory = one IG graph)
        g_topk = torch.autograd.grad(topk_loss.sum(), delta, retain_graph=False)[0]
        if use_grill:
            for r in recorders.values():                   # cheap separate forward for a_k(x+delta)
                r.reset()
            model(x + delta)
            adv_act = {n: recorders[n].out for n in names}
            D = torch.stack([(adv_act[n] - clean_act[n]).flatten(1).norm(dim=1)
                             / (clean_act[n].flatten(1).norm(dim=1) + 1e-12) for n in names], dim=0)  # [K,B]
            L_couple = 0.5 * (D.sum(0) ** 2 - (D ** 2).sum(0))
            g_couple = torch.autograd.grad(L_couple.sum(), delta)[0]
            if balance:
                scale = grill_balance * (g_topk.norm() / (g_couple.norm() + 1e-12))
            else:
                scale = torch.as_tensor(float(lambda_grill), device=x.device)
            eff = g_topk - grill_sign * scale * g_couple
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
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet50", choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--grill_balance", type=float, default=0.0)
    p.add_argument("--grill_sign", type=float, default=1.0, choices=[1.0, -1.0])
    p.add_argument("--lambda_grill", type=float, default=0.0)
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
    p.add_argument("--layers", nargs="+", default=None, help="subset of coupling-layer names (default all)")
    p.add_argument("--no_beta_growth", action="store_true")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=100,
                   help="#samples (default 100; count written into the folder name as img<count>).")
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

    dataset = G.LabelledImages(args.data_dir, G.load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) < 1:
        raise SystemExit("No labelled images - check --data_dir and --labels.")
    sgn = "neg" if args.grill_sign < 0 else ""
    mode_tag = f"bal{sgn}{fmt(args.grill_balance)}" if balance else f"lam{sgn}{fmt(args.lambda_grill)}"
    out_dir = args.out_dir or (f"results/smoothdefrepr_{args.model}_{mode_tag}"
                               f"_sig{fmt(args.sigma)}_N{args.n_smooth}_img{len(dataset)}")
    os.makedirs(out_dir, exist_ok=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    noise_gen = torch.Generator(device=device).manual_seed(args.noise_seed)

    sdir = "minimise-repr" if args.grill_sign < 0 else "maximise-repr"
    mode_desc = (f"REPR-GRILL balanced c={args.grill_balance} [{sdir}]" if balance else
                 (f"REPR-GRILL lambda={args.lambda_grill} [{sdir}]" if args.lambda_grill else "PURE top-k"))
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
            x_adv, org_idx, h_o, gn = repr_grill_attack(
                model, x, eps, args.steps, lr, args.prefactors, args.lambda_grill, args.grill_balance,
                args.grill_sign, args.num_summands, noise, coupling_layers, recorders,
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
              f"scale={last_gn[2]:.3e}")
        if not args.no_plots:
            G.save_quantitative_eps(os.path.join(quant_dir, tag), cols, eps_255, mnames)
            G.save_qualitative_grid(os.path.join(qual_dir, tag, "grid"), qual_samples, args.grid_rows)
        with open(os.path.join(out_dir, f"per_image_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        if args.save_expl:
            torch.save(store, os.path.join(out_dir, f"smoothdefrepr_{tag}.pt"))
        summary.append({"model": args.model, "grill_balance": args.grill_balance,
                        "grill_sign": args.grill_sign, "lambda_grill": args.lambda_grill,
                        "sigma": args.sigma, "n_smooth": args.n_smooth, "eps_x255": eps_255, "n_images": n,
                        "clean_acc": n_clean / n, "decision_unchanged": n_unchanged / n,
                        "seconds": round(elapsed, 1), "last_g_topk": last_gn[0], "last_g_couple": last_gn[1],
                        "last_scale": last_gn[2],
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
