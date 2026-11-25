"""
Generalized forward spectral probe (--model convnext_base | vit_b_16 | resnet50).

Same experiment as spectralProbeIG.py, but across the three classifiers, so we can test the
cross-model prediction from the spectrum check (spectralSpectrumCheck.py -> Table 1):
    near-null prevalence:  ConvNeXt (0%)  <  ResNet-50 (13%, early)  <  ViT-B/16 (26%, all attn out_proj)
    => IG explanation attacks should route their representation change dz into the BOTTOM-V
       (near-null) subspace of the probed layers on ViT/ResNet (low attenuation), but NOT on
       ConvNeXt (which has no near-null directions).

For a linear layer y = W z (W = U S V^T) we hook the layer INPUT z, and per source measure:
    psi_top_s / psi_bottom_s = energy of dz = z(x+delta) - z(x) in the top-s / bottom-s
                               right singular subspaces of W
    attenuation             = |W dz| / (|W| |dz|)   (-> 0 if dz is in the near-null subspace)
for sources ig_untar_topk, ig_untar_revdomb (explanation attacks, label kept), clf_untargeted
(classification attack), random (control). The attack files are reused unchanged
(get_expl_ig / ig_explanation_attack are architecture-agnostic: they take `model` as an arg).

PROBED LAYERS (the near-null targets from the spectrum check are marked *):
    convnext_base : stage{1,3,5,7} MLP down-projection pwconv2 (Linear 4C->C) + classifier head
                    (all well-conditioned: this model is the NEGATIVE CONTROL)
    vit_b_16      : * each block's attention output projection self_attention.out_proj (768x768,
                      sigma_min ~1e-5, the near-null target) + each block's MLP fc2 (well-
                      conditioned, shown for contrast)
    resnet50      : * layer1 1x1 convs conv1/conv3/downsample + layer1.1.conv1 (near-null,
                      sigma_min ~1e-7) + fc head. (3x3/7x7 convs are skipped: the filter-matrix
                      SVD is not a pointwise operator, so dz-at-input probing is ill-defined.)

ViT attention note. torchvision's nn.MultiheadAttention runs the FUNCTIONAL path and passes
out_proj.weight rather than calling the module, so a forward hook on out_proj never fires, and
out_proj's output cannot recover its near-null input (that is exactly what out_proj destroys).
We therefore replace each block's nn.MultiheadAttention with ProbeMHA, a thin wrapper that
reproduces standard self-attention and applies out_proj AS A MODULE (so the hook captures its
input). ProbeMHA is validated against the original module at build time (--check_mha).

Outputs (folder created at runtime; --out_dir, default spectralProbeGeneralResults/<model>/):
    per_token_alignment.csv, summary_by_source.csv, summary_overall.csv,
    figures/ (psi_bottom per layer, bottom/top ratio per source), config.txt

Usage
-----
    uv run python spectralProbeGeneral.py --model vit_b_16 \
        --data_dir data/imagenet_val_1000 --labels data/imagenet_val_1000/labels.csv \
        --eps 4 --steps 200 --num_images 20 --subspace_s 64

        

export CUDA_VISIBLE_DEVICES=5
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python spectralProbeGeneral.py --model vit_b_16 --eps 4 --steps 200 --num_images 20


# run later
uv run python spectralProbeGeneral.py --model resnet50      --eps 4 --steps 200 --num_images 20
uv run python spectralProbeGeneral.py --model convnext_base --eps 4 --steps 200 --num_images 20
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
# Model wrapper
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


class ProbeMHA(nn.Module):
    """
    Drop-in replacement for the torchvision ViT block's nn.MultiheadAttention (self-attention,
    batch_first, no mask) that applies out_proj as a MODULE so a forward hook on out_proj fires
    with the PRE-projection tensor (the concatenated head output) as its input.
    Reproduces nn.MultiheadAttention's math; validated against the original at build time.
    """
    def __init__(self, mha):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj  # same module object -> hook target

    def forward(self, query, key=None, value=None, need_weights=False, **kw):
        x = query  # self-attention: query == key == value
        B, N, E = x.shape
        h, hd = self.num_heads, E // self.num_heads
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)  # [B,N,3E]
        q, k, v = qkv.split(E, dim=-1)
        q = q.reshape(B, N, h, hd).transpose(1, 2)  # [B,h,N,hd]
        k = k.reshape(B, N, h, hd).transpose(1, 2)
        v = v.reshape(B, N, h, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        attn = attn.softmax(dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, N, E)  # pre-out_proj [B,N,E]
        out = self.out_proj(o)                           # module call -> hook fires (input = o)
        return out, None


def build_model(model_name, device, check_mha=False):
    import torchvision.models as M
    spec = {
        "convnext_base": ("convnext_base", "ConvNeXt_Base_Weights"),
        "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
        "resnet50": ("resnet50", "ResNet50_Weights"),
    }[model_name]
    weights = getattr(M, spec[1]).IMAGENET1K_V1
    tf = weights.transforms()
    base = getattr(M, spec[0])(weights=weights).eval()

    if model_name == "vit_b_16":
        # validate ProbeMHA on one block, then swap all blocks
        if check_mha:
            blk = base.encoder.layers[0]
            x = torch.randn(2, base.encoder.pos_embedding.shape[1], base.hidden_dim)
            with torch.no_grad():
                ln = blk.ln_1(x)
                ref, _ = blk.self_attention(ln, ln, ln, need_weights=False)
                alt, _ = ProbeMHA(blk.self_attention)(ln, ln, ln, need_weights=False)
            err = (ref - alt).abs().max().item()
            print(f"  ProbeMHA vs nn.MultiheadAttention max|diff| = {err:.2e}")
            assert err < 1e-4, "ProbeMHA does not match nn.MultiheadAttention"
        for blk in base.encoder.layers:
            blk.self_attention = ProbeMHA(blk.self_attention)

    model = NormalizedModel(base, tf.mean, tf.std).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    pixel_tf = transforms.Compose([
        transforms.Resize(tf.resize_size, interpolation=tf.interpolation, antialias=True),
        transforms.CenterCrop(tf.crop_size),
        transforms.ToTensor(),
    ])
    return model, pixel_tf, weights.meta["categories"]


def select_probe_layers(model_name, base):
    """Return [(name, module, kind, weight_matrix)]. kind in {linear, convNd->treated pointwise}."""
    layers = []
    if model_name == "convnext_base":
        for si, stage in enumerate(base.features):
            blocks = [m for m in stage.modules() if type(m).__name__ == "CNBlock"]
            if not blocks:
                continue
            lin = [m for m in blocks[-1].modules() if isinstance(m, nn.Linear)]
            if len(lin) >= 2:
                layers.append((f"stage{si}_pwconv2", lin[1], "linear_last", lin[1].weight))
        layers.append(("classifier_head", base.classifier[2], "linear_last",
                       base.classifier[2].weight))

    elif model_name == "vit_b_16":
        for i, blk in enumerate(base.encoder.layers):
            op = blk.self_attention.out_proj            # * near-null target
            layers.append((f"block{i}_attn_out", op, "linear_last", op.weight))
            fc2 = blk.mlp[3]                            # well-conditioned contrast
            layers.append((f"block{i}_mlp_fc2", fc2, "linear_last", fc2.weight))

    elif model_name == "resnet50":
        names = {"layer1.0.conv1": None, "layer1.0.conv3": None, "layer1.0.downsample.0": None,
                 "layer1.1.conv1": None}                # * near-null 1x1 convs
        mods = dict(base.named_modules())
        for nm in names:
            m = mods.get(nm)
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                W = m.weight.reshape(m.out_channels, -1)
                layers.append((nm.replace(".", "_"), m, "conv1x1", W))
        layers.append(("fc_head", base.fc, "linear_last", base.fc.weight))
    else:
        raise ValueError(model_name)
    return layers


class Recorder:
    """Forward hook capturing the layer INPUT as a [tokens, in_dim] matrix, kind-aware."""
    def __init__(self, kind):
        self.kind = kind
        self.z = None

    def __call__(self, module, inputs, output):
        a = inputs[0]
        if self.kind == "conv1x1":              # [B,C,H,W] -> tokens are spatial positions
            a = a.permute(0, 2, 3, 1)
        self.z = a.reshape(-1, a.shape[-1]).detach()


# --------------------------------------------------------------------------- #
# Spectral helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def layer_svd(W, device):
    s, Vh = torch.linalg.svd(W.to(torch.float32).to(device), full_matrices=True)[1:]
    return Vh.transpose(0, 1).contiguous(), s


@torch.no_grad()
def alignment(dz, V, s_dim):
    n = dz.shape[1]
    norm = dz.norm(dim=1, keepdim=True)
    keep = (norm.squeeze(1) > 1e-12)
    if keep.sum() == 0:
        return None
    u = dz[keep] / norm[keep]
    top = (u @ V[:, :s_dim]).pow(2).sum(1)
    bot = (u @ V[:, n - s_dim:]).pow(2).sum(1)
    return {"psi_top": top.mean().item(), "psi_bottom": bot.mean().item(),
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
    p.add_argument("--model", default="vit_b_16",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default="spectralProbeGeneralResults")
    p.add_argument("--eps", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--num_images", type=int, default=20)
    p.add_argument("--subspace_s", type=int, default=64)
    p.add_argument("--num_summands", type=int, default=30)
    p.add_argument("--topk_attack", type=int, default=1000)
    p.add_argument("--sources", nargs="+",
                   default=["ig_untar_topk", "ig_untar_revdomb", "clf_untargeted", "random"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    eps = args.eps / 255.0
    out_dir = os.path.join(args.out_dir, args.model)
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    IGt = load_module("ig_targeted", "ExplanationAttackonIG.py")
    IGk = load_module("ig_notarget", "ExplanationNotargetAttackonIG.py")
    IGr = load_module("ig_revdomb", "ExplanationUntargetedDrombowskiAttackonIG.py")
    CLF = load_module("clf_attack", "adversarialAttack.py")

    print(f"Building {args.model} ...")
    model, pixel_tf, categories = build_model(args.model, device, check_mha=(args.model == "vit_b_16"))
    probe_layers = select_probe_layers(args.model, model.model)
    s_dim = args.subspace_s

    recorders, Vs, svals, Wnorms, handles = {}, {}, {}, {}, []
    for name, mod, kind, W in probe_layers:
        rec = Recorder(kind)
        handles.append(mod.register_forward_hook(rec))
        recorders[name] = rec
        V, s = layer_svd(W, device)
        Vs[name], svals[name], Wnorms[name] = V, s, float(s.max())

    print(f"Model: {args.model} | device: {device} | images: {args.num_images} | "
          f"eps: {args.eps}/255 | steps: {args.steps} | s: {s_dim} | layers: {len(probe_layers)}")
    print("Probed layers (in_dim n, sigma_min = near-null indicator):")
    for name, mod, kind, W in probe_layers:
        print(f"  {name:22s} {kind:12s} {tuple(W.shape)}  sigma_min={svals[name].min():.2e}")

    labels = load_labels(args.labels)
    files = list_images(args.data_dir, labels, args.num_images, args.seed)
    if not files:
        raise SystemExit("No labelled images found - check --data_dir and --labels.")
    target_files = [files[(i + len(files) // 2) % len(files)] for i in range(len(files))]
    prefac = [1e11, 1e6]

    def record_z(xin):
        with torch.no_grad():
            model(xin)
        return {name: recorders[name].z.clone() for name, *_ in probe_layers}

    def run_source(src, x, x_t, name_cur):
        if src == "random":
            return torch.clamp(x + (torch.rand_like(x) * 2 - 1) * eps, 0, 1)
        if src == "ig_targeted":
            xa, _ = quiet(IGt.ig_explanation_attack, model, x, x_t, eps, args.steps,
                          0.1 * eps, prefac, args.num_summands, [], False); return xa
        if src == "ig_untar_topk":
            xa, _ = quiet(IGk.ig_explanation_attack, model, x, x_t, eps, args.steps,
                          0.1 * eps, [100.0, 1e6], args.num_summands, [], False, args.topk_attack); return xa
        if src == "ig_untar_revdomb":
            xa, _ = quiet(IGr.ig_explanation_attack, model, x, x_t, eps, args.steps,
                          0.1 * eps, prefac, args.num_summands, [], False); return xa
        if src == "clf_untargeted":
            y = torch.tensor([labels[name_cur]], device=device)
            xa, _ = quiet(CLF.adam_linf_attack, model, x, y, eps, args.steps,
                          0.25 * eps, "margin", 1); return xa
        raise ValueError(src)

    weight_of = {name: W for name, _, _, W in probe_layers}
    rows = []
    t0 = time.time()
    for idx, (fname, tname) in enumerate(zip(files, target_files), 1):
        x = load_img(args.data_dir, fname, pixel_tf, device)
        x_t = load_img(args.data_dir, tname, pixel_tf, device)
        z_clean = record_z(x)
        h_clean, _, pred_clean = quiet(IGt.get_expl_ig, model, x, None, args.num_summands)
        h_clean = h_clean.detach()

        for src in args.sources:
            x_adv = run_source(src, x, x_t, fname).detach()
            z_adv = record_z(x_adv)
            h_adv, _, pred_adv = quiet(IGt.get_expl_ig, model, x_adv, None, args.num_summands)
            h_adv = h_adv.detach()
            linf = float((x_adv - x).abs().max() * 255)
            label_pres = int(pred_adv.item() == pred_clean.item())
            ig_pcc = float(IGt.pcc(h_adv, h_clean).mean())
            ig_topk = float(IGt.topk_intersection(h_adv, h_clean, args.topk_attack).mean())

            for name, mod, kind, W in probe_layers:
                dz = z_adv[name] - z_clean[name]
                a = alignment(dz, Vs[name], s_dim)
                if a is None:
                    continue
                with torch.no_grad():
                    Wdz = F.linear(dz, W.to(dz.dtype))
                    atten = float((Wdz.norm(dim=1) / (Wnorms[name] * dz.norm(dim=1) + 1e-12)).mean())
                rows.append({"image": fname, "source": src, "layer": name, "kind": kind,
                             "eps_x255": args.eps, "linf_x255": round(linf, 4),
                             "sigma_min": float(svals[name].min()),
                             "label_preserved": label_pres, "ig_pcc": ig_pcc, "ig_topk": ig_topk,
                             "psi_top_s": a["psi_top"], "psi_bottom_s": a["psi_bottom"],
                             "psi_ratio_bottom_over_top": a["psi_bottom"] / (a["psi_top"] + 1e-12),
                             "attenuation": atten, "dz_norm": a["dz_norm"], "n_tokens": a["n_tokens"]})
        if idx % 5 == 0 or idx == len(files):
            print(f"  [{idx:4d}/{len(files)}]  ({time.time() - t0:.0f}s)")

    for h in handles:
        h.remove()

    with open(os.path.join(out_dir, "per_token_alignment.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    def agg(sub):
        out = {}
        for k in ["psi_top_s", "psi_bottom_s", "psi_ratio_bottom_over_top", "attenuation",
                  "dz_norm", "label_preserved", "ig_pcc", "ig_topk"]:
            v = np.array([r[k] for r in sub], float)
            out["median_" + k] = float(np.median(v)); out["mean_" + k] = float(np.mean(v))
        out["n"] = len(sub); return out

    layer_names = [n for n, *_ in probe_layers]
    with open(os.path.join(out_dir, "summary_by_source.csv"), "w", newline="") as fh:
        w = None
        for src in args.sources:
            for lname in layer_names:
                sub = [r for r in rows if r["source"] == src and r["layer"] == lname]
                if not sub:
                    continue
                rec = {"source": src, "layer": lname, "sigma_min": sub[0]["sigma_min"], **agg(sub)}
                if w is None:
                    w = csv.DictWriter(fh, fieldnames=list(rec.keys())); w.writeheader()
                w.writerow(rec)

    overall = []
    with open(os.path.join(out_dir, "summary_overall.csv"), "w", newline="") as fh:
        w = None
        for src in args.sources:
            sub = [r for r in rows if r["source"] == src]
            rec = {"source": src, **agg(sub)}; overall.append(rec)
            if w is None:
                w = csv.DictWriter(fh, fieldnames=list(rec.keys())); w.writeheader()
            w.writerow(rec)

    # group near-null vs well-conditioned layers (sigma_min < 1e-3) for the key comparison
    print(f"\n{'=' * 90}\n{args.model}: forward probe  (eps={args.eps}/255, s={s_dim})\n{'=' * 90}")
    for group, pred in [("NEAR-NULL layers (sigma_min<1e-3)", lambda r: r["sigma_min"] < 1e-3),
                        ("well-conditioned layers", lambda r: r["sigma_min"] >= 1e-3)]:
        grp = [r for r in rows if pred(r)]
        if not grp:
            continue
        print(f"\n  [{group}]  ({len(set(r['layer'] for r in grp))} layers)")
        print(f"  {'source':18s}{'Psi_bottom':>12}{'Psi_top':>10}{'bot/top':>10}{'atten':>9}"
              f"{'label_kept':>12}{'IG PCC':>9}{'IG topk':>9}")
        for src in args.sources:
            sub = [r for r in grp if r["source"] == src]
            a = agg(sub)
            print(f"  {src:18s}{a['median_psi_bottom_s']:>12.4f}{a['median_psi_top_s']:>10.4f}"
                  f"{a['median_psi_ratio_bottom_over_top']:>10.2f}{a['median_attenuation']:>9.3f}"
                  f"{a['median_label_preserved']:>12.2f}{a['median_ig_pcc']:>9.3f}{a['median_ig_topk']:>9.3f}")
    print("\nKey test (near-null layers only): do the IG attacks (label_kept~1) show LOWER "
          "attenuation / HIGHER Psi_bottom than clf_untargeted and random?")

    try:
        make_figures(rows, layer_names, args, fig_dir)
    except Exception as e:
        print("(figure step skipped:", e, ")")
    with open(os.path.join(out_dir, "config.txt"), "w") as fh:
        fh.write(str(vars(args)) + "\nprobe_layers: " + ", ".join(layer_names) + "\n")
    print(f"\nResults -> {out_dir}")


def make_figures(rows, layer_names, args, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 8, "axes.spines.top": False,
                                "axes.spines.right": False, "pdf.fonttype": 42})
    colors = {"ig_untar_topk": "#009E73", "ig_untar_revdomb": "#56B4E9",
              "clf_untargeted": "#D55E00", "random": "#999999", "ig_targeted": "#0072B2"}
    show = layer_names if len(layer_names) <= 14 else layer_names[::max(1, len(layer_names) // 12)]
    x = np.arange(len(show)); width = 0.8 / max(len(args.sources), 1)

    for metric, ylab, fn in [("attenuation", "median attenuation |Wdz|/|W||dz|", "attenuation_by_layer"),
                             ("psi_bottom_s", r"median $\Psi_{\mathrm{bottom}}$", "psi_bottom_by_layer")]:
        fig, ax = plt.subplots(figsize=(7.16, 2.8))
        for i, src in enumerate(args.sources):
            vals = []
            for lname in show:
                sub = [r[metric] for r in rows if r["source"] == src and r["layer"] == lname]
                vals.append(np.median(sub) if sub else 0)
            ax.bar(x + i * width, vals, width, label=src, color=colors.get(src))
        ax.set_xticks(x + width * (len(args.sources) - 1) / 2)
        ax.set_xticklabels(show, rotation=40, ha="right", fontsize=5)
        ax.set_ylabel(ylab)
        ax.legend(ncol=len(args.sources), fontsize=6, loc="upper center", bbox_to_anchor=(0.5, 1.2))
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, fn + ".pdf"))
        fig.savefig(os.path.join(fig_dir, fn + ".png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
