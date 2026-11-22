"""
H1 spectral probe: do explanation attacks on Integrated Gradients hide their
representation changes in the BOTTOM (near-null) singular subspaces of ConvNeXt-Base's
intermediate linear layers, while a classification attack uses the TOP subspaces?

Idea (from "On Adversarial Vulnerability of VLMs through Intermediate Spectral
Subspaces", SSGRA, turned around for the IG explanation threat model):
  For a linear layer y = W z (W = U S V^T), a change dz = z(x+delta) - z(x) that lies
  in the bottom-s right-singular subspace of W is attenuated by W (|W dz| ~ 0), so the
  forward signal / decision barely moves, but the gradient (hence IG) still changes
  because it depends on the pre-activation states. So a prediction-preserving
  explanation attack can "hide" its dz in the bottom subspace; a classification
  attack, which must move the output, should use the top subspace.

What this script does (NO new attack is proposed here - this only PROBES existing ones):
  * loads ConvNeXt-Base (frozen) with normalisation inside the model ([0,1] pixel space);
  * registers forward hooks on the INPUTS z_m of selected linear layers
    (the stage MLP down-projections pwconv2, i.e. Linear(4C -> C), plus the classifier
    head) and records the SVD (right singular vectors V) of each W once;
  * for each image, builds adversarial examples with several perturbation sources at a
    common L_inf budget:
        ig_targeted      - Dombrowski targeted IG attack      (ExplanationAttackonIG)
        ig_untar_topk    - Ghorbani top-k untargeted IG attack (ExplanationNotargetAttackonIG)
        ig_untar_revdomb - reversed-Dombrowski untargeted IG   (ExplanationUntargetedDrombowskiAttackonIG)
        clf_untargeted   - untargeted classification attack    (adversarialAttack)
        random           - uniform noise in the L_inf ball     (control)
  * measures, per layer m and per source, the spectral alignment of the normalised
    representation change  dz_m = z_m(x+delta) - z_m(x)  with the top-s and bottom-s
    right singular subspaces of W_m, using the paper's alignment measure
        Psi_s(dz, V_s) = mean_token  sum_{j in subspace} ( v_j^T dz / |dz| )^2 ,
    plus the attenuation ratio  |W_m dz| / |W_m| |dz|  (how much the layer passes through);
  * also records whether the predicted class was preserved and the IG-map change
    (PCC / top-k) so each perturbation source can be read against its effect.

Outputs (folder created at runtime; see --out_dir, default spectralProbeResults/):
  per_token_alignment.csv   one row per (image, source, layer, subspace position):
                            Psi_top_s, Psi_bottom_s, attenuation, dz_norm, ...
  summary_by_source.csv     medians/means of Psi_top_s, Psi_bottom_s, attenuation,
                            label_preserved, ig_pcc, ig_topk  per (source, layer)
  summary_overall.csv       the same aggregated over layers (one row per source)
  spectral_svals.csv        singular-value spectrum + null/near-null counts per layer
  figures/                  bottom-vs-top alignment bar charts per layer, and
                            Psi_bottom vs eps / per source (publication style, pdf+png)
  config.txt                exact settings of the run

Usage
-----
    uv run python spectralProbeIG.py \
        --data_dir data/imagenet_val_1000 \
        --labels   data/imagenet_val_1000/labels.csv \
        --eps 4 --steps 300 --num_images 50 --subspace_s 64 \
        --sources ig_untar_topk ig_untar_revdomb clf_untargeted random \
        --out_dir spectralProbeResults
  (eps in units of 1/255. Start SMALL: --num_images 20 --steps 200 to check timing;
   the IG attacks run 30 forward passes + double backward per step.)


export CUDA_VISIBLE_DEVICES=2
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python spectralProbeIG.py \
  --data_dir data/imagenet_val_1000 --labels data/imagenet_val_1000/labels.csv \
  --eps 4 --steps 200 --num_images 20 --subspace_s 64 \
  --sources ig_untar_topk ig_untar_revdomb clf_untargeted random


  
uv run python ExplanationMasscenterAttackonIG.py --max_images 100 --eps 2 4 8 --n_qual 10
uv run python ExplanationCosineAttackonIG.py     --max_images 100 --eps 2 4 8 --n_qual 10

"""

import argparse
import contextlib
import csv
import importlib.util
import io
import os
import random
import re
import sys
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


# --------------------------------------------------------------------------- #
# Import the existing attack modules (they live next to this file)
# --------------------------------------------------------------------------- #
def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def quiet(fn, *a, **k):
    """Run fn while swallowing its stdout (the attack files have debug prints)."""
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
    """
    Linear layers whose INPUT change we probe. We take the MLP down-projection of the
    last CNBlock in each ConvNeXt stage (Linear 4C -> C, which has a 75% exact null
    space), plus the classifier head. Returns [(name, module)].
    nn.Linear in ConvNeXt acts on channels-last tokens [B, H, W, C_in], so the layer
    input is a per-token vector in R^{C_in} - exactly the z_m of the paper.
    """
    layers = []
    for si, stage in enumerate(base.features):
        blocks = [m for m in stage.modules() if type(m).__name__ == "CNBlock"]
        if not blocks:
            continue
        blk = blocks[-1]
        lin = [m for m in blk.modules() if isinstance(m, nn.Linear)]
        if len(lin) >= 2:
            layers.append((f"stage{si}_pwconv2", lin[1]))  # the down-projection 4C->C
    layers.append(("classifier_head", base.classifier[2]))
    return layers


class InputRecorder:
    """Forward pre-hook: store the flattened per-token INPUT of a linear layer."""
    def __init__(self):
        self.z = None

    def __call__(self, module, inputs):
        a = inputs[0]
        self.z = a.reshape(-1, a.shape[-1]).detach()  # [tokens, C_in]


# --------------------------------------------------------------------------- #
# Spectral alignment measures
# --------------------------------------------------------------------------- #
@torch.no_grad()
def layer_svd(W, device):
    """Right singular vectors V [n, n] (columns) and singular values s, for W [m,n]."""
    s, Vh = torch.linalg.svd(W.to(torch.float32).to(device), full_matrices=True)[1:]
    return Vh.transpose(0, 1).contiguous(), s  # V, s


@torch.no_grad()
def alignment(dz, V, s_dim):
    """
    dz: [tokens, n] representation change at a layer's input.
    Returns Psi_top_s, Psi_bottom_s (paper Eq.1, averaged over normalised token
    changes) and the mean |dz|. Tokens with ~zero change are dropped.
    """
    n = dz.shape[1]
    norm = dz.norm(dim=1, keepdim=True)
    keep = (norm.squeeze(1) > 1e-12)
    if keep.sum() == 0:
        return None
    u = dz[keep] / norm[keep]                          # normalised token changes
    Vtop = V[:, :s_dim]                                # [n, s]
    Vbot = V[:, n - s_dim:]
    proj_top = (u @ Vtop).pow(2).sum(1)                # energy in top-s subspace
    proj_bot = (u @ Vbot).pow(2).sum(1)
    return {"psi_top": proj_top.mean().item(), "psi_bottom": proj_bot.mean().item(),
            "dz_norm": norm[keep].mean().item(), "n_tokens": int(keep.sum())}


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
    p.add_argument("--out_dir", default="spectralProbeResults")
    p.add_argument("--eps", type=float, default=4.0, help="L_inf budget (units of 1/255)")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--num_images", type=int, default=50)
    p.add_argument("--subspace_s", type=int, default=64, help="s: top-s / bottom-s dim")
    p.add_argument("--num_summands", type=int, default=30, help="IG steps")
    p.add_argument("--topk_attack", type=int, default=1000)
    p.add_argument("--sources", nargs="+",
                   default=["ig_untar_topk", "ig_untar_revdomb", "clf_untargeted", "random"],
                   help="subset of: ig_targeted ig_untar_topk ig_untar_revdomb clf_untargeted random")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    eps = args.eps / 255.0
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    # attack modules
    IGt = load_module("ig_targeted", "ExplanationAttackonIG.py")
    IGk = load_module("ig_notarget", "ExplanationNotargetAttackonIG.py")
    IGr = load_module("ig_revdomb", "ExplanationUntargetedDrombowskiAttackonIG.py")
    CLF = load_module("clf_attack", "adversarialAttack.py")

    model, pixel_tf, categories = build_model(device)
    base = model.model
    probe_layers = select_probe_layers(base)
    s_dim = args.subspace_s

    # SVD + recorders for each probed layer
    recorders, Vs, svals, Wnorms = {}, {}, {}, {}
    handles = []
    for name, mod in probe_layers:
        rec = InputRecorder()
        handles.append(mod.register_forward_pre_hook(rec))
        recorders[name] = rec
        V, s = layer_svd(mod.weight.data, device)
        Vs[name], svals[name], Wnorms[name] = V, s, float(s.max())

    print(f"Model: ConvNeXt-Base | device: {device} | images: {args.num_images} | "
          f"eps: {args.eps}/255 | steps: {args.steps} | s: {s_dim}")
    print("Probed layers (input dim n, exact null space n-m):")
    for name, mod in probe_layers:
        m_, n_ = mod.weight.shape
        nn_small = int((svals[name] < 1e-3).sum())
        print(f"  {name:22s} n={n_:5d}  exact_null={max(n_-m_,0):5d}  sigma<1e-3={nn_small:5d}  "
              f"sigma_min={svals[name].min():.2e}")

    # spectral spectrum file
    with open(os.path.join(args.out_dir, "spectral_svals.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "n", "m", "sigma_max", "sigma_min", "exact_null",
                    "n_below_1e-2", "n_below_1e-3", "n_below_1e-4"])
        for name, mod in probe_layers:
            m_, n_ = mod.weight.shape
            s = svals[name]
            w.writerow([name, n_, m_, f"{s.max():.6e}", f"{s.min():.6e}", max(n_ - m_, 0),
                        int((s < 1e-2).sum()), int((s < 1e-3).sum()), int((s < 1e-4).sum())])

    # images
    labels = load_labels(args.labels)
    files = list_images(args.data_dir, labels, args.num_images, args.seed)
    if not files:
        raise SystemExit("No labelled images found - check --data_dir and --labels.")
    # deterministic target for the two attacks that need one
    target_files = [files[(i + len(files) // 2) % len(files)] for i in range(len(files))]

    prefac = [1e11, 1e6]

    def record_z(xin):
        """Forward pass that fills the recorders; return {name: z [tokens, n]}."""
        with torch.no_grad():
            model(xin)
        return {name: recorders[name].z.clone() for name, _ in probe_layers}

    def run_source(src, x, x_t):
        """Return x_adv [1,3,H,W] for the given perturbation source."""
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
        name_cur = fname
        x = load_img(args.data_dir, fname, pixel_tf, device)
        x_t = load_img(args.data_dir, tname, pixel_tf, device)

        z_clean = record_z(x)
        h_clean, _, pred_clean = quiet(IGt.get_expl_ig, model, x, None, args.num_summands)
        h_clean = h_clean.detach()

        for src in args.sources:
            x_adv = run_source(src, x, x_t).detach()
            z_adv = record_z(x_adv)
            h_adv, _, pred_adv = quiet(IGt.get_expl_ig, model, x_adv, None, args.num_summands)
            h_adv = h_adv.detach()

            linf = float((x_adv - x).abs().max() * 255)
            label_pres = int(pred_adv.item() == pred_clean.item())
            ig_pcc = float(IGt.pcc(h_adv, h_clean).mean())
            ig_topk = float(IGt.topk_intersection(h_adv, h_clean, args.topk_attack).mean())

            for name, mod in probe_layers:
                dz = (z_adv[name] - z_clean[name])            # [tokens, n]
                a = alignment(dz, Vs[name], s_dim)
                if a is None:
                    continue
                # attenuation |W dz| / (|W| |dz|)
                with torch.no_grad():
                    Wdz = F.linear(dz, mod.weight.data.to(dz.dtype))
                    atten = float((Wdz.norm(dim=1) /
                                   (Wnorms[name] * dz.norm(dim=1) + 1e-12)).mean())
                rows.append({
                    "image": fname, "target": tname, "source": src, "layer": name,
                    "eps_x255": args.eps, "linf_x255": round(linf, 4),
                    "label_preserved": label_pres, "ig_pcc": ig_pcc, "ig_topk": ig_topk,
                    "psi_top_s": a["psi_top"], "psi_bottom_s": a["psi_bottom"],
                    "psi_ratio_bottom_over_top": a["psi_bottom"] / (a["psi_top"] + 1e-12),
                    "attenuation": atten, "dz_norm": a["dz_norm"],
                    "n_tokens": a["n_tokens"], "s_dim": s_dim})
        if idx % 5 == 0 or idx == len(files):
            print(f"  [{idx:4d}/{len(files)}]  ({time.time() - t0:.0f}s)")

    for h in handles:
        h.remove()

    # ---- write per-token/per-layer rows ----
    cols = list(rows[0].keys())
    with open(os.path.join(args.out_dir, "per_token_alignment.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)

    # ---- aggregate ----
    def agg(subset):
        out = {}
        for k in ["psi_top_s", "psi_bottom_s", "psi_ratio_bottom_over_top", "attenuation",
                  "dz_norm", "label_preserved", "ig_pcc", "ig_topk"]:
            v = np.array([r[k] for r in subset], dtype=float)
            out["median_" + k] = float(np.median(v))
            out["mean_" + k] = float(np.mean(v))
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

    # ---- console summary ----
    print(f"\n{'=' * 78}\nH1 probe summary  (eps={args.eps}/255, s={s_dim})\n{'=' * 78}")
    print(f"{'source':18s}{'Psi_bottom':>12}{'Psi_top':>10}{'bot/top':>10}"
          f"{'atten':>9}{'label_kept':>12}{'IG PCC':>9}")
    for rec in overall:
        print(f"{rec['source']:18s}{rec['median_psi_bottom_s']:>12.4f}"
              f"{rec['median_psi_top_s']:>10.4f}{rec['median_psi_ratio_bottom_over_top']:>10.2f}"
              f"{rec['median_attenuation']:>9.3f}{rec['median_label_preserved']:>12.2f}"
              f"{rec['median_ig_pcc']:>9.3f}")
    print("\nExpected if H1 holds: the IG explanation attacks (label_kept ~ 1) show higher "
          "Psi_bottom and lower attenuation than clf_untargeted and random.")

    # ---- figures ----
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

    # bottom vs top alignment per layer, grouped by source
    x = np.arange(len(layer_names))
    width = 0.8 / max(len(args.sources), 1)
    fig, ax = plt.subplots(figsize=(7.16, 2.6))
    for i, src in enumerate(args.sources):
        vals = []
        for lname in layer_names:
            sub = [r["psi_bottom_s"] for r in rows if r["source"] == src and r["layer"] == lname]
            vals.append(np.median(sub) if sub else 0)
        ax.bar(x + i * width, vals, width, label=src, color=colors.get(src, None))
    ax.set_xticks(x + width * (len(args.sources) - 1) / 2)
    ax.set_xticklabels(layer_names, rotation=30, ha="right")
    ax.set_ylabel(r"median $\Psi_{\mathrm{bottom}}$")
    ax.legend(ncol=len(args.sources), fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "psi_bottom_by_layer.pdf"))
    fig.savefig(os.path.join(fig_dir, "psi_bottom_by_layer.png"), dpi=300)
    plt.close(fig)

    # bottom/top ratio per source (overall)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    srcs = [r["source"] for r in overall]
    ratio = [r["median_psi_ratio_bottom_over_top"] for r in overall]
    ax.bar(range(len(srcs)), ratio, color=[colors.get(s, None) for s in srcs])
    ax.axhline(1.0, color="k", lw=0.6, ls="--")
    ax.set_xticks(range(len(srcs))); ax.set_xticklabels(srcs, rotation=30, ha="right")
    ax.set_ylabel(r"median $\Psi_{\mathrm{bottom}}/\Psi_{\mathrm{top}}$")
    fig.tight_layout()
    fig.savefig(os.path.join(fig_dir, "bottom_top_ratio.pdf"))
    fig.savefig(os.path.join(fig_dir, "bottom_top_ratio.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
