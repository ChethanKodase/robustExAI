"""
Gradient-degradation diagnostic for IG explanation attacks (the GRILL-IG go/no-go test).

Question: do the EXISTING (terminal-only) IG attacks suffer the gradient degradation the GRILL
papers describe -- i.e. does the second-order optimisation gradient of the terminal input-IG
distortion loss collapse while crossing the ill-conditioned (near-null) layers, so that adding
intermediate Layer-IG distortions (GRILL-IG) would inject signal the baselines miss?

We measure, at a random start in the eps-ball (where the distortion losses are already non-zero),
for each model (--model convnext_base | vit_b_16 | resnet50) and each coupling layer k:

  (A) BACKWARD PROFILE.  d L_term / d a_k , where L_term = || IG(x+delta) - IG(x) ||^2 is the
      terminal (reversed-Dombrowski) input-IG distortion and a_k is the output of layer k along
      the IG path (captured by a retain_grad forward hook). We report the per-element RMS of this
      second-order attack signal per layer, plus the signal that reaches the INPUT (d L_term/d delta).
      If the RMS collapses at / after the near-null layers, the terminal attack signal is dying there.

  (B) GRILL RATIO (the decisive number).  || d (d_k^2) / d delta ||  /  || d L_term / d delta || ,
      where d_k = || LayerIG_k(x+delta) - LayerIG_k(x) ||_2 is the layer-k Layer-IG distortion.
      This is how much larger the INPUT-space attack gradient is when it comes from a single
      intermediate layer's IG distortion (short, less-degraded backward path) than from the terminal
      input-IG distortion. Ratio >> 1 at the near-null layers  =>  GRILL-IG will pay off.

Prediction (from spectralSpectrumResults/): the near-null layers, and therefore a large GRILL ratio,
are ABSENT on ConvNeXt-Base (0% near-null, negative control), present and EARLY on ResNet-50
(13%), and present in every attention output projection on ViT-B/16 (26%). So GRILL-IG should beat
the terminal-only baselines more on ViT/ResNet than on ConvNeXt.

Reuses the GRILL-IG attack module unchanged (get_layer_ig / get_expl_ig / build_model /
select_coupling_layers), so the diagnostic measures EXACTLY the attack's Layer-IG.

Outputs (--out_dir, default gradientDegradationResults/<model>/):
    per_layer.csv, summary_by_group.csv, figures/ (backward_profile, grill_ratio, ratio_vs_sigmamin),
    config.txt

Usage
-----
    uv run python gradientDegradationProbeIG.py --model vit_b_16 \
        --data_dir data/imagenet_val_1000 --labels data/imagenet_val_1000/labels.csv \
        --eps 4 --num_images 20

    uv run python gradientDegradationProbeIG.py --model resnet50      --eps 4 --num_images 20
    uv run python gradientDegradationProbeIG.py --model convnext_base --eps 4 --num_images 20
    (eps in units of 1/255. Lower --num_summands for ViT if memory is tight.)

    

export CUDA_VISIBLE_DEVICES=3
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python gradientDegradationProbeIG.py --model vit_b_16      --eps 4 --num_images 20
uv run python gradientDegradationProbeIG.py --model resnet50      --eps 4 --num_images 20
uv run python gradientDegradationProbeIG.py --model convnext_base --eps 4 --num_images 20
    

"""

import argparse
import contextlib
import csv
import importlib.util
import io
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


# --------------------------------------------------------------------------- #
# Spectral helper: sigma_min / sigma_max / condition number of a coupling layer
# --------------------------------------------------------------------------- #
@torch.no_grad()
def weight_spectrum(module, device):
    if isinstance(module, nn.Conv2d):
        W = module.weight.reshape(module.out_channels, -1)
    elif isinstance(module, nn.Linear):
        W = module.weight
    else:
        return None
    s = torch.linalg.svdvals(W.to(torch.float32).to(device))
    smin, smax = float(s.min()), float(s.max())
    return {"sigma_min": smin, "sigma_max": smax, "cond": smax / (smin + 1e-30),
            "near_null": int(smin < 1e-3), "shape": tuple(W.shape)}


def rms(t):
    return float(t.detach().float().pow(2).mean().sqrt()) if t is not None else float("nan")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def numeric_key(f):
    m = re.match(r"(\d+)", f)
    return (int(m.group(1)) if m else float("inf"), f)


def load_labels(path):
    labels = {}
    with open(path, newline="") as fh:
        first = fh.readline(); fh.seek(0)
        if "," in first:
            for row in csv.reader(fh):
                if not row or row[0].strip().lower() in ("filename", "file", "image"):
                    continue
                labels[row[0].strip()] = int(row[1])
        else:
            for line in fh:
                p = line.split()
                if len(p) >= 2:
                    labels[p[0]] = int(p[1])
    return labels


def list_images(data_dir, labels, n, seed):
    files = [f for f in os.listdir(data_dir)
             if f.lower().endswith(IMG_EXTENSIONS) and f in labels]
    files = sorted(files, key=numeric_key)
    if n and n < len(files):
        files = [files[i] for i in sorted(random.Random(seed).sample(range(len(files)), n))]
    return files


def load_img(data_dir, f, tf, device):
    return tf(Image.open(os.path.join(data_dir, f)).convert("RGB")).unsqueeze(0).to(device)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vit_b_16",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default="gradientDegradationResults")
    p.add_argument("--eps", type=float, default=4.0, help="L_inf of the random start, in 1/255")
    p.add_argument("--num_images", type=int, default=20)
    p.add_argument("--num_summands", type=int, default=30)
    p.add_argument("--layers", nargs="+", default=None,
                   help="subset of coupling-layer names (default: all)")
    p.add_argument("--topk", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    eps = args.eps / 255.0
    out_dir = os.path.join(args.out_dir, args.model)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    G = load_module("grill", "ExplanationGrillAttackonIG.py")

    print(f"Building {args.model} ...")
    model, pixel_tf, categories, beta_modules, crop = G.build_model(
        args.model, device, check_mha=(args.model == "vit_b_16"))
    coupling_layers = G.select_coupling_layers(args.model, model.model)
    if args.layers:
        wanted = set(args.layers)
        coupling_layers = [c for c in coupling_layers if c[0] in wanted]
    names = [n for n, *_ in coupling_layers]

    # spectra of the coupling layers
    spec = {name: weight_spectrum(mod, device) for name, mod, kind in coupling_layers}

    # retain_grad recorders (so we can read d L_term / d a_k off the hooked outputs)
    recorders, handles = G.attach_recorders(coupling_layers, retain=True)
    baseline_acts = G.precompute_baseline_acts(model, coupling_layers, recorders, device, crop)

    # ResNet uses ReLU: smooth it (softplus) at a fixed beta so we isolate WEIGHT ill-conditioning
    # from ReLU's zero second derivative (Dombrowski's beta fix). GELU models: no-op.
    if beta_modules:
        G.change_beta(beta_modules, 50.0)

    print(f"Model: {args.model} | device: {device} | images: {args.num_images} | "
          f"eps(start): {args.eps}/255 | summands: {args.num_summands} | layers: {len(coupling_layers)}")
    for name, mod, kind in coupling_layers:
        s = spec[name]
        print(f"  {name:22s} {kind:12s} {s['shape']}  sigma_min={s['sigma_min']:.2e} "
              f"{'* NEAR-NULL' if s['near_null'] else ''}")

    labels = load_labels(args.labels)
    files = list_images(args.data_dir, labels, args.num_images, args.seed)
    if not files:
        raise SystemExit("No labelled images - check --data_dir and --labels.")

    rows = []
    t0 = time.time()
    for idx, fname in enumerate(files, 1):
        x = load_img(args.data_dir, fname, pixel_tf, device)

        # clean references (detached)
        clean_ig = G.get_expl_ig(model, x, None, args.num_summands, create_graph=False)[0].detach()
        clean_ligs, _, pred_clean = G.get_layer_ig(
            model, x, None, coupling_layers, recorders, baseline_acts, args.num_summands, False)
        clean_ligs = {k: v.detach() for k, v in clean_ligs.items()}

        # random start in the eps-ball
        d0 = (torch.rand_like(x) * 2 - 1) * eps
        x_adv0 = torch.clamp(x + d0, 0, 1)

        # ---------- (B) GRILL ratio: input-space gradients ----------
        delta = (x_adv0 - x).detach().requires_grad_(True)
        adv_ig = G.get_expl_ig(model, x + delta, pred_clean, args.num_summands, create_graph=True)[0]
        L_term = ((adv_ig - clean_ig) ** 2).flatten(1).sum(1).sum()
        g_term = torch.autograd.grad(L_term, delta, retain_graph=False)[0]
        gterm_norm = float(g_term.norm())

        adv_ligs, adv_acc, adv_pred = G.get_layer_ig(
            model, x + delta, pred_clean, coupling_layers, recorders, baseline_acts,
            args.num_summands, create_graph=True)
        grill_ratio = {}
        for n in names:
            d2 = ((adv_ligs[n] - clean_ligs[n]) ** 2).flatten(1).sum(1).sum()  # d_n^2
            g_n = torch.autograd.grad(d2, delta, retain_graph=True)[0]
            grill_ratio[n] = float(g_n.norm()) / (gterm_norm + 1e-12)

        label_pres = int(adv_pred.item() == pred_clean.item())
        linf = float((x + delta - x).abs().max() * 255)
        ig_pcc = float(G.pcc(adv_ig.detach(), clean_ig).mean())
        ig_topk = float(G.topk_intersection(adv_ig.detach(), clean_ig, args.topk).mean())

        # ---------- (A) backward profile: d L_term / d a_k ----------
        for r in recorders.values():
            r.reset()
        delta2 = (x_adv0 - x).detach().requires_grad_(True)
        adv_ig2 = G.get_expl_ig(model, x + delta2, pred_clean, args.num_summands, create_graph=True)[0]
        L_term2 = ((adv_ig2 - clean_ig) ** 2).flatten(1).sum(1).sum()
        L_term2.backward()
        input_rms = rms(delta2.grad)
        layer_grad_rms = {n: rms(recorders[n].out.grad) for n in names}

        for name, mod, kind in coupling_layers:
            s = spec[name]
            rows.append({
                "image": fname, "model": args.model, "layer": name, "kind": kind,
                "sigma_min": s["sigma_min"], "sigma_max": s["sigma_max"], "cond": s["cond"],
                "near_null": s["near_null"], "eps_x255": args.eps, "linf_x255": round(linf, 4),
                "label_preserved": label_pres, "ig_pcc_start": ig_pcc, "ig_topk_start": ig_topk,
                "gradA_rms": layer_grad_rms[name], "input_gradA_rms": input_rms,
                "gradA_over_input": layer_grad_rms[name] / (input_rms + 1e-30),
                "grill_ratio": grill_ratio[name], "gterm_norm": gterm_norm,
            })
        if idx % 5 == 0 or idx == len(files):
            print(f"  [{idx:4d}/{len(files)}]  ({time.time() - t0:.0f}s)")

    for h in handles:
        h.remove()

    with open(os.path.join(out_dir, "per_layer.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- group near-null vs well-conditioned ----
    def agg(sub, key):
        v = np.array([r[key] for r in sub], float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    groups = [("NEAR-NULL (sigma_min<1e-3)", lambda r: r["near_null"] == 1),
              ("well-conditioned", lambda r: r["near_null"] == 0)]
    srows = []
    print(f"\n{'=' * 84}\n{args.model}: gradient-degradation diagnostic (eps_start={args.eps}/255)\n{'=' * 84}")
    for gname, pred in groups:
        grp = [r for r in rows if pred(r)]
        if not grp:
            continue
        nlayers = len(set(r["layer"] for r in grp))
        med_ratio = agg(grp, "grill_ratio")
        med_gradA = agg(grp, "gradA_rms")
        med_goi = agg(grp, "gradA_over_input")
        med_sig = agg(grp, "sigma_min")
        srows.append({"group": gname, "n_layers": nlayers, "median_sigma_min": med_sig,
                      "median_grill_ratio": med_ratio, "median_gradA_rms": med_gradA,
                      "median_gradA_over_input": med_goi})
        print(f"\n  [{gname}]  ({nlayers} layers)")
        print(f"    median GRILL ratio ||d(d_k^2)/ddelta|| / ||dL_term/ddelta|| : {med_ratio:.3f}")
        print(f"    median backward RMS d L_term/d a_k                         : {med_gradA:.3e}")
        print(f"    median (layer backward RMS) / (input backward RMS)         : {med_goi:.3f}")
    print("\nKey test: on ViT/ResNet the NEAR-NULL group should show GRILL ratio >> 1 (intermediate "
          "Layer-IG distortions give a much larger input gradient than the terminal IG loss). On "
          "ConvNeXt (no near-null layers) the ratio should stay ~1  =>  GRILL-IG helps model-dependently.")

    with open(os.path.join(out_dir, "summary_by_group.csv"), "w", newline="") as fh:
        if srows:
            w = csv.DictWriter(fh, fieldnames=list(srows[0].keys()))
            w.writeheader(); w.writerows(srows)

    try:
        make_figures(rows, names, spec, args, fig_dir)
    except Exception as e:
        print("(figure step skipped:", e, ")")
    with open(os.path.join(out_dir, "config.txt"), "w") as fh:
        fh.write(str(vars(args)) + "\nlayers: " + ", ".join(names) + "\n")
    print(f"\nResults -> {out_dir}")


def make_figures(rows, names, spec, args, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 8, "axes.spines.top": False,
                                "axes.spines.right": False, "pdf.fonttype": 42})
    show = names if len(names) <= 16 else names[::max(1, len(names) // 14)]
    x = np.arange(len(show))
    nn_mask = [spec[n]["near_null"] == 1 for n in show]

    def med(metric, lname):
        v = [r[metric] for r in rows if r["layer"] == lname]
        v = [t for t in v if np.isfinite(t)]
        return float(np.median(v)) if v else 0.0

    # (A) backward-gradient RMS profile
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    vals = [med("gradA_rms", n) for n in show]
    ax.bar(x, vals, color=["#D55E00" if m else "#0072B2" for m in nn_mask])
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=40, ha="right", fontsize=5)
    ax.set_ylabel(r"median RMS  $\partial L_{term}/\partial a_k$")
    ax.set_title("backward attack-signal per layer (orange = near-null)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "backward_profile.pdf"))
    fig.savefig(os.path.join(fig_dir, "backward_profile.png"), dpi=300); plt.close(fig)

    # (B) GRILL ratio per layer
    fig, ax = plt.subplots(figsize=(7.16, 2.8))
    vals = [med("grill_ratio", n) for n in show]
    ax.bar(x, vals, color=["#D55E00" if m else "#0072B2" for m in nn_mask])
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(show, rotation=40, ha="right", fontsize=5)
    ax.set_ylabel(r"median $\|\partial d_k^2/\partial\delta\| / \|\partial L_{term}/\partial\delta\|$")
    ax.set_title("GRILL ratio: layer-IG vs terminal input gradient (orange = near-null)", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "grill_ratio.pdf"))
    fig.savefig(os.path.join(fig_dir, "grill_ratio.png"), dpi=300); plt.close(fig)

    # (C) ratio vs sigma_min scatter
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    xs = [spec[n]["sigma_min"] for n in names]
    ys = [med("grill_ratio", n) for n in names]
    ax.scatter(xs, ys, s=12, color="#0072B2")
    ax.set_xscale("log"); ax.set_xlabel(r"$\sigma_{\min}(W_k)$")
    ax.set_ylabel("median GRILL ratio")
    ax.axhline(1.0, color="black", ls="--", lw=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(fig_dir, "ratio_vs_sigmamin.pdf"))
    fig.savefig(os.path.join(fig_dir, "ratio_vs_sigmamin.png"), dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
