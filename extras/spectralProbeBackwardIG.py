"""
Backward-side spectral probe (diagnostic B): does the explanation attack's change in the
BACKWARD signal (the gradient that builds the IG map) concentrate in the bottom LEFT
singular subspaces of ConvNeXt-Base's intermediate linear layers?

Why a second probe. The forward probe (spectralProbeIG.py) tested only the FORWARD side:
it aligned the representation change dz = z(x+delta) - z(x) at a layer's INPUT with the
RIGHT singular vectors V of W, and found no signal (attenuation identical across sources;
the "bottom" there is the trivial geometric null space of the rectangular 4C->C map).
IG is a BACKWARD quantity, so the more natural object is the gradient. For y = W z
(W = U S V^T):
    forward :  dz lives in input space (basis V),  W dz = U S (V^T dz)
    backward:  g_y = d(logit_c)/dy lives in OUTPUT space (basis U);
               the gradient at the input is g_z = W^T g_y = V S (U^T g_y).
So the symmetric backward test aligns the change of the output gradient
    dg_y = g_y(x+delta) - g_y(x)
with the LEFT singular vectors U. Unlike the forward "bottom" (V), the bottom-s columns
of U are NOT a null space: all min(m,n) left singular vectors have non-zero sigma, so
this probes genuine small-sigma (near-null) OUTPUT directions.

Hypothesis (H1-backward): the prediction-preserving IG attacks steer dg_y into bottom-U
(small-sigma output directions) more than the classification attack or random noise.
Those directions are produced weakly by the forward map, so changing the gradient there
is "cheap" and need not move the forward activations (hence the decision).

Backward signal used. g_y is the gradient of the explained-class LOGIT w.r.t. the layer
output, at the actual input x (the plain gradient / saliency backward signal). This is a
tractable proxy for the IG backward signal (IG is the path-average of this gradient);
--ig_path averages it over the num_summands interpolation points instead (slower, closer
to IG). The explained class is the CLEAN prediction, used for both clean and adv (the
attacks preserve it), so dg_y reflects the attack, not a class change.

This script reuses the existing attack files unchanged, exactly like spectralProbeIG.py,
and reports BOTH sides per layer and source:
    forward : psi_top/bottom of dz in V, attenuation |W dz|/(|W||dz|)
    backward: psi_top/bottom of dg_y in U, gy change norm

Sources: ig_untar_topk, ig_untar_revdomb, clf_untargeted, random (as before).

Outputs (folder created at runtime; --out_dir, default spectralProbeBackwardResults/):
    per_token_alignment.csv   one row per (image, source, layer) with fwd_* and bwd_*
    summary_by_source.csv / summary_overall.csv
    figures/                  bwd psi_bottom per layer; fwd-vs-bwd bottom/top ratio
    config.txt

Usage
-----
    uv run python spectralProbeBackwardIG.py \
        --data_dir data/imagenet_val_1000 --labels data/imagenet_val_1000/labels.csv \
        --eps 4 --steps 200 --num_images 20 --subspace_s 64 \
        --sources ig_untar_topk ig_untar_revdomb clf_untargeted random \
        --out_dir spectralProbeBackwardResults


export CUDA_VISIBLE_DEVICES=2
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python spectralProbeBackwardIG.py --eps 4 --steps 200 --num_images 20 --subspace_s 64
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

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
# Model + probed layers
# --------------------------------------------------------------------------- #
class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.model(self.normalize(x))


def build_model(device):
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    tf = weights.transforms()
    base = convnext_base(weights=weights).eval()
    model = NormalizedModel(base, tf.mean, tf.std).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pixel_tf = transforms.Compose([
        transforms.Resize(tf.resize_size, interpolation=tf.interpolation, antialias=True),
        transforms.CenterCrop(tf.crop_size),
        transforms.ToTensor(),
    ])
    return model, pixel_tf, weights.meta["categories"]


def select_probe_layers(base):
    """MLP down-projection (pwconv2, Linear 4C->C) of the last CNBlock per stage,
    plus the classifier head. Same layers as the forward probe."""
    layers = []
    for si, stage in enumerate(base.features):
        blocks = [m for m in stage.modules() if type(m).__name__ == "CNBlock"]
        if not blocks:
            continue
        lin = [m for m in blocks[-1].modules() if isinstance(m, nn.Linear)]
        if len(lin) >= 2:
            layers.append((f"stage{si}_pwconv2", lin[1]))
    layers.append(("classifier_head", base.classifier[2]))
    return layers


class Recorder:
    """Forward hook: store the layer INPUT z (detached) and OUTPUT y (grad retained)."""
    def __init__(self):
        self.z = None
        self.y = None

    def __call__(self, module, inputs, output):
        a = inputs[0]
        self.z = a.reshape(-1, a.shape[-1]).detach()
        if output.requires_grad:
            output.retain_grad()
        self.y = output


# --------------------------------------------------------------------------- #
# Spectral helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def layer_svd(W, device):
    """Return U [m,m], V [n,n] (columns = singular vectors), s (min(m,n))."""
    U, s, Vh = torch.linalg.svd(W.to(torch.float32).to(device), full_matrices=True)
    return U.contiguous(), Vh.transpose(0, 1).contiguous(), s


@torch.no_grad()
def alignment(vec, Basis, s_dim):
    """
    vec: [tokens, d] change; Basis: [d, d] orthonormal columns ordered by descending sigma.
    Returns psi_top (energy in top-s columns), psi_bottom (bottom-s columns), mean norm.
    """
    d = vec.shape[1]
    norm = vec.norm(dim=1, keepdim=True)
    keep = (norm.squeeze(1) > 1e-12)
    if keep.sum() == 0:
        return None
    u = vec[keep] / norm[keep]
    top = (u @ Basis[:, :s_dim]).pow(2).sum(1)
    bot = (u @ Basis[:, d - s_dim:]).pow(2).sum(1)
    return {"psi_top": top.mean().item(), "psi_bottom": bot.mean().item(),
            "norm": norm[keep].mean().item(), "n_tokens": int(keep.sum())}


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
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default="spectralProbeBackwardResults")
    p.add_argument("--eps", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--num_images", type=int, default=20)
    p.add_argument("--subspace_s", type=int, default=64)
    p.add_argument("--num_summands", type=int, default=30)
    p.add_argument("--topk_attack", type=int, default=1000)
    p.add_argument("--ig_path", action="store_true",
                   help="average the backward signal over the IG interpolation path "
                        "(slower, closer to the true IG gradient) instead of the plain gradient at x")
    p.add_argument("--sources", nargs="+",
                   default=["ig_untar_topk", "ig_untar_revdomb", "clf_untargeted", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    eps = args.eps / 255.0
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    IGt = load_module("ig_targeted", "ExplanationAttackonIG.py")
    IGk = load_module("ig_notarget", "ExplanationNotargetAttackonIG.py")
    IGr = load_module("ig_revdomb", "ExplanationUntargetedDrombowskiAttackonIG.py")
    CLF = load_module("clf_attack", "adversarialAttack.py")

    model, pixel_tf, categories = build_model(device)
    base = model.model
    probe_layers = select_probe_layers(base)
    s_dim = args.subspace_s

    recorders, Us, Vs, svals, Wnorms = {}, {}, {}, {}, {}
    handles = []
    for name, mod in probe_layers:
        rec = Recorder()
        handles.append(mod.register_forward_hook(rec))
        recorders[name] = rec
        U, V, s = layer_svd(mod.weight.data, device)
        Us[name], Vs[name], svals[name], Wnorms[name] = U, V, s, float(s.max())

    print(f"Model: ConvNeXt-Base | device: {device} | images: {args.num_images} | "
          f"eps: {args.eps}/255 | steps: {args.steps} | s: {s_dim} | "
          f"backward signal: {'IG-path avg' if args.ig_path else 'plain gradient'}")
    print("Probed layers (output dim m -> left basis U is m x m, all sigma non-zero):")
    for name, mod in probe_layers:
        m_, n_ = mod.weight.shape
        print(f"  {name:22s} m={m_:5d}  n={n_:5d}  sigma_min(U side)={svals[name].min():.2e}")

    labels = load_labels(args.labels)
    files = list_images(args.data_dir, labels, args.num_images, args.seed)
    if not files:
        raise SystemExit("No labelled images found - check --data_dir and --labels.")
    target_files = [files[(i + len(files) // 2) % len(files)] for i in range(len(files))]
    prefac = [1e11, 1e6]

    def record(xin, class_idx):
        """
        Grad-enabled forward (+ backward of the explained-class logit) that fills the
        recorders. Returns z_dict {name:[tokens,n]} and gy_dict {name:[tokens,m]} where
        gy = d(logit_c)/d(layer output). With --ig_path, gy is averaged over the IG
        interpolation path (mean of d(logit_c)/d(output) at t/num_summands * z_norm).
        """
        model.zero_grad(set_to_none=True)

        if not args.ig_path:
            x = xin.clone().detach().requires_grad_(True)
            logits = model(x)
            sel = logits.gather(1, class_idx.view(-1, 1)).sum()
            sel.backward()
            z_dict = {n_: recorders[n_].z.clone() for n_, _ in probe_layers}
            gy_dict = {n_: recorders[n_].y.grad.reshape(-1, recorders[n_].y.shape[-1]).detach().clone()
                       for n_, _ in probe_layers}
            return z_dict, gy_dict

        # IG-path averaged backward signal (baseline 0 in normalised space)
        z_in = model.normalize(xin).detach()
        N = args.num_summands
        gy_sum = {n_: 0.0 for n_, _ in probe_layers}
        z_dict = None
        for k in range(1, N + 1):
            model.zero_grad(set_to_none=True)
            zk = (float(k) / N * z_in).detach().requires_grad_(True)
            logits = base(zk)
            sel = logits.gather(1, class_idx.view(-1, 1)).sum()
            sel.backward()
            for n_, _ in probe_layers:
                gy_sum[n_] = gy_sum[n_] + recorders[n_].y.grad.reshape(
                    -1, recorders[n_].y.shape[-1]).detach()
            if k == N:
                z_dict = {n_: recorders[n_].z.clone() for n_, _ in probe_layers}
        gy_dict = {n_: gy_sum[n_] / N for n_, _ in probe_layers}
        return z_dict, gy_dict

    def run_source(src, x, x_t, name_cur):
        if src == "random":
            d = (torch.rand_like(x) * 2 - 1) * eps
            return torch.clamp(x + d, 0, 1)
        if src == "ig_targeted":
            xa, _ = quiet(IGt.ig_explanation_attack, model, x, x_t, eps, args.steps,
                          0.1 * eps, prefac, args.num_summands, [], False)
            return xa
        if src == "ig_untar_topk":
            xa, _ = quiet(IGk.ig_explanation_attack, model, x, x_t, eps, args.steps,
                          0.1 * eps, [100.0, 1e6], args.num_summands, [], False, args.topk_attack)
            return xa
        if src == "ig_untar_revdomb":
            xa, _ = quiet(IGr.ig_explanation_attack, model, x, x_t, eps, args.steps,
                          0.1 * eps, prefac, args.num_summands, [], False)
            return xa
        if src == "clf_untargeted":
            y = torch.tensor([labels[name_cur]], device=device)
            xa, _ = quiet(CLF.adam_linf_attack, model, x, y, eps, args.steps,
                          0.25 * eps, "margin", 1)
            return xa
        raise ValueError(src)

    rows = []
    t0 = time.time()
    for idx, (fname, tname) in enumerate(zip(files, target_files), 1):
        x = load_img(args.data_dir, fname, pixel_tf, device)
        x_t = load_img(args.data_dir, tname, pixel_tf, device)

        h_clean, _, pred_clean = quiet(IGt.get_expl_ig, model, x, None, args.num_summands)
        h_clean = h_clean.detach()
        cidx = pred_clean.view(1).to(device)
        z_clean, gy_clean = record(x, cidx)

        for src in args.sources:
            x_adv = run_source(src, x, x_t, fname).detach()
            h_adv, _, pred_adv = quiet(IGt.get_expl_ig, model, x_adv, None, args.num_summands)
            h_adv = h_adv.detach()
            z_adv, gy_adv = record(x_adv, cidx)  # same explained class

            linf = float((x_adv - x).abs().max() * 255)
            label_pres = int(pred_adv.item() == pred_clean.item())
            ig_pcc = float(IGt.pcc(h_adv, h_clean).mean())
            ig_topk = float(IGt.topk_intersection(h_adv, h_clean, args.topk_attack).mean())

            for name, mod in probe_layers:
                dz = z_adv[name] - z_clean[name]
                dgy = gy_adv[name] - gy_clean[name]
                fwd = alignment(dz, Vs[name], s_dim)      # forward: input change in V
                bwd = alignment(dgy, Us[name], s_dim)     # backward: output-grad change in U
                if fwd is None:
                    continue
                with torch.no_grad():
                    Wdz = F.linear(dz, mod.weight.data.to(dz.dtype))
                    atten = float((Wdz.norm(dim=1) / (Wnorms[name] * dz.norm(dim=1) + 1e-12)).mean())
                row = {"image": fname, "source": src, "layer": name,
                       "eps_x255": args.eps, "linf_x255": round(linf, 4),
                       "label_preserved": label_pres, "ig_pcc": ig_pcc, "ig_topk": ig_topk,
                       "fwd_psi_top": fwd["psi_top"], "fwd_psi_bottom": fwd["psi_bottom"],
                       "fwd_ratio_bot_top": fwd["psi_bottom"] / (fwd["psi_top"] + 1e-12),
                       "attenuation": atten, "dz_norm": fwd["norm"]}
                if bwd is None:  # e.g. classifier head: dgy ~ 0 (one-hot cancels)
                    row.update({"bwd_psi_top": float("nan"), "bwd_psi_bottom": float("nan"),
                                "bwd_ratio_bot_top": float("nan"), "dgy_norm": 0.0,
                                "bwd_valid": 0})
                else:
                    row.update({"bwd_psi_top": bwd["psi_top"], "bwd_psi_bottom": bwd["psi_bottom"],
                                "bwd_ratio_bot_top": bwd["psi_bottom"] / (bwd["psi_top"] + 1e-12),
                                "dgy_norm": bwd["norm"], "bwd_valid": 1})
                rows.append(row)
        if idx % 5 == 0 or idx == len(files):
            print(f"  [{idx:4d}/{len(files)}]  ({time.time() - t0:.0f}s)")

    for h in handles:
        h.remove()

    cols = list(rows[0].keys())
    with open(os.path.join(args.out_dir, "per_token_alignment.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    metric_keys = ["fwd_psi_top", "fwd_psi_bottom", "fwd_ratio_bot_top", "attenuation",
                   "dz_norm", "bwd_psi_top", "bwd_psi_bottom", "bwd_ratio_bot_top",
                   "dgy_norm", "label_preserved", "ig_pcc", "ig_topk"]

    def agg(subset):
        out = {}
        for k in metric_keys:
            v = np.array([r[k] for r in subset], dtype=float)
            v = v[~np.isnan(v)]
            out["median_" + k] = float(np.median(v)) if len(v) else float("nan")
            out["mean_" + k] = float(np.mean(v)) if len(v) else float("nan")
        out["n"] = len(subset)
        return out

    layer_names = [n for n, _ in probe_layers]
    with open(os.path.join(args.out_dir, "summary_by_source.csv"), "w", newline="") as fh:
        w = None
        for src in args.sources:
            for lname in layer_names:
                sub = [r for r in rows if r["source"] == src and r["layer"] == lname]
                if not sub:
                    continue
                rec = {"source": src, "layer": lname, **agg(sub)}
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(rec.keys())); w.writeheader()
                w.writerow(rec)

    overall = []
    with open(os.path.join(args.out_dir, "summary_overall.csv"), "w", newline="") as fh:
        w = None
        for src in args.sources:
            sub = [r for r in rows if r["source"] == src]
            rec = {"source": src, **agg(sub)}
            overall.append(rec)
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(rec.keys())); w.writeheader()
            w.writerow(rec)

    print(f"\n{'=' * 86}\nBackward probe summary  (eps={args.eps}/255, s={s_dim})\n{'=' * 86}")
    print(f"{'source':18s}{'bwd_Psi_bot':>12}{'bwd_Psi_top':>12}{'bwd_bot/top':>12}"
          f"{'fwd_bot/top':>12}{'label_kept':>11}{'IG PCC':>9}")
    for rec in overall:
        print(f"{rec['source']:18s}{rec['median_bwd_psi_bottom']:>12.4f}"
              f"{rec['median_bwd_psi_top']:>12.4f}{rec['median_bwd_ratio_bot_top']:>12.2f}"
              f"{rec['median_fwd_ratio_bot_top']:>12.2f}{rec['median_label_preserved']:>11.2f}"
              f"{rec['median_ig_pcc']:>9.3f}")
    print("\nExpected if H1-backward holds: the IG attacks (label_kept ~ 1) show HIGHER "
          "bwd_Psi_bottom / bwd_bot-top ratio than clf_untargeted and random.")

    try:
        make_figures(rows, overall, layer_names, args, fig_dir)
    except Exception as e:
        print("(figure step skipped:", e, ")")

    with open(os.path.join(args.out_dir, "config.txt"), "w") as fh:
        fh.write(str(vars(args)) + "\n")
        fh.write("probe_layers: " + ", ".join(layer_names) + "\n")
    print(f"\nResults -> {args.out_dir}")


def make_figures(rows, overall, layer_names, args, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 8, "axes.spines.top": False,
                                "axes.spines.right": False, "pdf.fonttype": 42})
    colors = {"ig_targeted": "#0072B2", "ig_untar_topk": "#009E73",
              "ig_untar_revdomb": "#56B4E9", "clf_untargeted": "#D55E00", "random": "#999999"}

    # backward psi_bottom per layer, grouped by source
    x = np.arange(len(layer_names))
    width = 0.8 / max(len(args.sources), 1)
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    for i, src in enumerate(args.sources):
        vals = []
        for lname in layer_names:
            sub = [r["bwd_psi_bottom"] for r in rows
                   if r["source"] == src and r["layer"] == lname and not np.isnan(r["bwd_psi_bottom"])]
            vals.append(np.median(sub) if sub else 0)
        ax.bar(x + i * width, vals, width, label=src, color=colors.get(src))
    ax.set_xticks(x + width * (len(args.sources) - 1) / 2)
    ax.set_xticklabels(layer_names, rotation=30, ha="right")
    ax.set_ylabel(r"median backward $\Psi_{\mathrm{bottom}}$ (left subspace)")
    ax.legend(ncol=len(args.sources), fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bwd_psi_bottom_by_layer.pdf"))
    fig.savefig(os.path.join(fig_dir, "bwd_psi_bottom_by_layer.png"), dpi=300)
    plt.close(fig)

    # forward vs backward bottom/top ratio per source (overall)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    srcs = [r["source"] for r in overall]
    xb = np.arange(len(srcs))
    ax.bar(xb - 0.2, [r["median_fwd_ratio_bot_top"] for r in overall], 0.4, label="forward (V)",
           color="#999999")
    ax.bar(xb + 0.2, [r["median_bwd_ratio_bot_top"] for r in overall], 0.4, label="backward (U)",
           color="#0072B2")
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xticks(xb); ax.set_xticklabels(srcs, rotation=30, ha="right")
    ax.set_ylabel(r"median $\Psi_{\mathrm{bottom}}/\Psi_{\mathrm{top}}$")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "fwd_vs_bwd_ratio.pdf"))
    fig.savefig(os.path.join(fig_dir, "fwd_vs_bwd_ratio.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
