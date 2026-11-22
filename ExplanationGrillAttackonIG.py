"""
GRILL-IG: UNTARGETED L_inf explanation attack on Integrated Gradients that couples
per-layer (Layer-IG) distortions with dense pairwise (GRILL-advanced) multiplicative
coupling, across ConvNeXt-Base / ViT-B/16 / ResNet-50 (ImageNet).

    *** New attack. Same threat model and framework as ExplanationNotargetAttackonIG.py
        (Ghorbani top-k), ExplanationUntargetedDrombowskiAttackonIG.py (reversed MSE),
        ExplanationMasscenterAttackonIG.py, ExplanationCosineAttackonIG.py. ***
    Frozen pretrained classifier, white-box, perturbation to the INPUT only, hard
    L_inf bound, projected Adam, hard decision constraint (best iterate chosen only
    among iterates whose predicted class == the clean prediction). Untargeted: push
    the explanation away from its own clean map; no target image is used.

Why a new attack (the GRILL connection)
---------------------------------------
    The existing IG attacks all optimise a SINGLE terminal explanation-distortion loss
    on the INPUT-level IG map,  L = D( IG(x_adv), . ).  IG is itself a gradient, so the
    attack's optimisation gradient  dL/dx_adv  is a *double backward* (second order) and
    has to traverse every layer back to the input. Near-zero singular values in the
    weight matrices attenuate that backward signal -- exactly the gradient degradation
    the GRILL papers identify for reconstruction attacks:
        - Ramanaik et al., "Revealing Hidden Vulnerabilities in Autoencoders through
          Gradient Signal Restoration" (GRILL, arXiv 2505.03646): near-zero singular
          values make white-box attack gradients vanish -> suboptimal attacks ->
          illusion of robustness. Fix: couple intermediate distortions multiplicatively
          so well-conditioned pathways rescue the degraded ones,  L = d* * sum_k d_k.
        - GRILL-advanced (draft): dense pairwise coupling  L = sum_{i<j} d_i d_j
          = 1/2 [ (sum_i d_i)^2 - sum_i d_i^2 ]  -- every pathway reinforces every other.
    GRILL-IG applies the dense pairwise coupling to EXPLANATION distortions measured at
    several intermediate layers (Layer-Integrated-Gradients), instead of only the
    terminal input map. When the terminal IG distortion's second-order gradient vanishes
    through the ill-conditioned layers, the intermediate Layer-IG distortions -- shorter,
    less-degraded backward paths -- provide non-vanishing ascent directions and mutually
    scale each other, so the attack finds manipulations the terminal-only baselines miss.

Objective
---------
    d_k(x_adv) = || LayerIG_k(x_adv) - LayerIG_k(x) ||_2        (per sample, >= 0, 0 at x)
    L_couple   = sum_{i<j} d_i d_j = 1/2 [ (sum_k d_k)^2 - sum_k d_k^2 ]
    minimise   - prefactor_expl * L_couple  +  prefactor_out * MSE( g(x_adv), g(x) )
    LayerIG_k  : Layer-Integrated-Gradients of the ORIGINAL predicted class attributed to
                 the output of intermediate layer k (Dombrowski baseline 0, right-Riemann
                 sum over num_summands, grad x (a_k(x) - a_k(0)), |.| summed over the
                 channel/feature axis, normalised to sum 1 -- same recipe as the input IG).
    Coupling layers k = the intermediate layers probed by spectralSpectrumCheck.py
      (the near-null targets): ConvNeXt stage pwconv2; ViT attention out_proj + MLP fc2;
      ResNet layer1 1x1 convs. The classifier head is excluded (distorting it changes the
      decision). d_k starts at 0, so (like the reversed-Dombrowski / cosine / mass-centre
      variants) the attack uses a RANDOM start in the eps-ball.

Cross-model prediction (the paper's test; from spectralSpectrumResults/)
    GRILL-IG should beat the terminal-only baselines MORE on the ill-conditioned models
    (ViT-B/16: 26% near-null, all attention out_proj; ResNet-50: 13%, early convs) than
    on the well-conditioned negative control (ConvNeXt-Base: 0% near-null).

Outputs / plots: per_image_*.csv, summary.csv (+ --save_expl .pt), publication-style
    qualitative grid and quantitative distributions, same style as the baseline files.

Usage
-----
    uv run python ExplanationGrillAttackonIG.py --model vit_b_16 \
        --data_dir data/imagenet_val_1000 --labels data/imagenet_val_1000/labels.csv \
        --eps 4 8 --steps 500 --max_images 100 --n_qual 10

    export CUDA_VISIBLE_DEVICES=0
    cd /data1/chethan/robustExAI
    source .venv/bin/activate
    uv run python ExplanationGrillAttackonIG.py --model vit_b_16  --max_images 100 --eps 4 8 --n_qual 10
    uv run python ExplanationGrillAttackonIG.py --model resnet50  --max_images 100 --eps 4 8 --n_qual 10
    uv run python ExplanationGrillAttackonIG.py --model convnext_base --max_images 100 --eps 4 8 --n_qual 10
    (eps in units of 1/255. For ViT lower --num_summands or --layers if memory is tight.)
"""

import argparse
import csv
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMG_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Model: normalisation inside, so the attack works in [0,1] pixel space
# (identical to ExplanationNotargetAttackonIG.py / spectralProbeGeneral.py)
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
    Drop-in replacement for the torchvision ViT block's nn.MultiheadAttention that applies
    out_proj as a MODULE (so a forward hook on out_proj fires). Reproduces self-attention
    math; validated against the original at build time (--check_mha). Same as the one in
    spectralProbeGeneral.py -- needed so Layer-IG can attribute to the attention output
    projection (the near-null layer).
    """
    def __init__(self, mha):
        super().__init__()
        self.embed_dim = mha.embed_dim
        self.num_heads = mha.num_heads
        self.in_proj_weight = mha.in_proj_weight
        self.in_proj_bias = mha.in_proj_bias
        self.out_proj = mha.out_proj

    def forward(self, query, key=None, value=None, need_weights=False, **kw):
        x = query
        B, N, E = x.shape
        h, hd = self.num_heads, E // self.num_heads
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.split(E, dim=-1)
        q = q.reshape(B, N, h, hd).transpose(1, 2)
        k = k.reshape(B, N, h, hd).transpose(1, 2)
        v = v.reshape(B, N, h, hd).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (hd ** 0.5)
        attn = attn.softmax(dim=-1)
        o = (attn @ v).transpose(1, 2).reshape(B, N, E)
        return self.out_proj(o), None


# --------------------------------------------------------------------------- #
# beta growth (ReLU -> softplus_beta) -- Dombrowski. No-op on GELU models
# (ConvNeXt, ViT); active on ResNet (ReLU) where the double backward needs a
# non-zero second derivative.
# --------------------------------------------------------------------------- #
class BetaActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.beta = None

    def forward(self, x):
        return F.relu(x) if self.beta is None else F.softplus(x, beta=self.beta)


def replace_relu_with_beta_activation(module):
    swapped = []
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            new = BetaActivation()
            setattr(module, name, new)
            swapped.append(new)
        else:
            swapped.extend(replace_relu_with_beta_activation(child))
    return swapped


def change_beta(beta_modules, beta):
    for m in beta_modules:
        m.beta = beta


def get_beta(i, num_iter):
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)


# --------------------------------------------------------------------------- #
# Build model (multi-architecture) + coupling-layer selection
# --------------------------------------------------------------------------- #
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
    beta_modules = replace_relu_with_beta_activation(base)  # ResNet only

    if model_name == "vit_b_16":
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
    crop = tf.crop_size[0] if isinstance(tf.crop_size, (list, tuple)) else tf.crop_size
    return model, pixel_tf, weights.meta["categories"], beta_modules, int(crop)


def select_coupling_layers(model_name, base):
    """
    Intermediate layers whose Layer-IG distortions are coupled (the near-null targets from
    the spectrum check; classifier head EXCLUDED). Returns [(name, module, kind)].
      kind: "conv1x1"  -> output [B,C,H,W], reduce channel axis dim=1
            "linear_last" (ConvNeXt pwconv2) -> output [B,H,W,C], reduce dim=-1
            "tokens" (ViT) -> output [B,N,E], reduce dim=-1
    """
    layers = []
    if model_name == "convnext_base":
        for si, stage in enumerate(base.features):
            blocks = [m for m in stage.modules() if type(m).__name__ == "CNBlock"]
            if not blocks:
                continue
            lin = [m for m in blocks[-1].modules() if isinstance(m, nn.Linear)]
            if len(lin) >= 2:
                layers.append((f"stage{si}_pwconv2", lin[1], "linear_last"))
    elif model_name == "vit_b_16":
        for i, blk in enumerate(base.encoder.layers):
            layers.append((f"block{i}_attn_out", blk.self_attention.out_proj, "tokens"))
            layers.append((f"block{i}_mlp_fc2", blk.mlp[3], "tokens"))
    elif model_name == "resnet50":
        names = ["layer1.0.conv1", "layer1.0.conv3", "layer1.0.downsample.0", "layer1.1.conv1"]
        mods = dict(base.named_modules())
        for nm in names:
            m = mods.get(nm)
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                layers.append((nm.replace(".", "_"), m, "conv1x1"))
    else:
        raise ValueError(model_name)
    return layers


class OutRecorder:
    """Forward hook capturing a layer's OUTPUT (kept in the graph; optional retain_grad for
    the gradient-degradation diagnostic)."""
    def __init__(self, retain=False):
        self.out = None
        self.retain = retain

    def reset(self):
        self.out = None

    def __call__(self, module, inputs, output):
        o = output[0] if isinstance(output, tuple) else output
        if self.retain and o.requires_grad:
            o.retain_grad()
        self.out = o


def attach_recorders(coupling_layers, retain=False):
    recorders, handles = {}, []
    for name, mod, kind in coupling_layers:
        rec = OutRecorder(retain=retain)
        handles.append(mod.register_forward_hook(rec))
        recorders[name] = rec
    return recorders, handles


@torch.no_grad()
def precompute_baseline_acts(model, coupling_layers, recorders, device, crop):
    """a_k(0): layer outputs at the IG baseline (0 in normalised space). Independent of the
    input, so computed once and reused every step."""
    for r in recorders.values():
        r.reset()
    model.model(torch.zeros(1, 3, crop, crop, device=device))
    return {name: recorders[name].out.detach().clone() for name, *_ in coupling_layers}


# --------------------------------------------------------------------------- #
# Channel/feature reduction + sum-to-1 normalisation (same recipe as the input IG)
# --------------------------------------------------------------------------- #
def _reduce_normalize(t, kind):
    if kind == "conv1x1":              # [B,C,H,W]
        m = t.abs().sum(1)             # -> [B,H,W]
    elif kind == "linear_last":        # ConvNeXt pwconv2 [B,H,W,C]
        m = t.abs().sum(-1)            # -> [B,H,W]
    elif kind == "tokens":             # ViT [B,N,E]
        m = t.abs().sum(-1)            # -> [B,N]
    else:                              # head / vector [B,D]
        m = t.abs()
    denom = m.flatten(1).sum(1).clamp_min(1e-12)
    return m / denom.view(-1, *([1] * (m.dim() - 1)))


# --------------------------------------------------------------------------- #
# Input-level Integrated Gradients (for metrics / reporting) -- identical recipe to
# ExplanationNotargetAttackonIG.get_expl_ig (Dombrowski "sum then differentiate").
# --------------------------------------------------------------------------- #
def get_expl_ig(model, x, desired_index=None, num_summands=30, create_graph=False):
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        z = model.normalize(x)
        logits = model.model(z)
        acc, class_idx = F.softmax(logits, dim=1), logits.argmax(dim=1)
        if desired_index is None:
            desired_index = class_idx
        B = z.shape[0]
        prefactors = z.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        scaled = (prefactors.view(num_summands, 1, 1, 1, 1) * z.unsqueeze(0)).reshape(
            num_summands * B, *z.shape[1:])
        y = model.model(scaled).view(num_summands, B, -1)
        y = F.softmax(y, dim=2)
        y = y.gather(2, desired_index.view(1, B, 1).expand(num_summands, B, 1)).squeeze(2)
        y = (1 / num_summands) * torch.sum(y / prefactors.view(num_summands, 1), dim=0)
        heatmap = torch.autograd.grad(y.sum(), z, create_graph=create_graph)[0]
        heatmap = heatmap * z
        heatmap = torch.sum(torch.abs(heatmap), dim=1)
        normalized_heatmap = heatmap / heatmap.flatten(1).sum(1).view(B, 1, 1)
    return normalized_heatmap, acc, class_idx


# --------------------------------------------------------------------------- #
# Layer-Integrated-Gradients at several intermediate layers (the GRILL-IG core).
# One batched forward over the num_summands path points; one autograd.grad gives the
# path gradient w.r.t. every coupling layer's output at once. Returns a dict
# name -> normalised attribution map (sum 1), plus softmax output and class of x
# (read off the alpha=1 path block, no extra forward).
# --------------------------------------------------------------------------- #
def get_layer_ig(model, x, desired_index, coupling_layers, recorders, baseline_acts,
                 num_summands=30, create_graph=False):
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        z = model.normalize(x)
        B = z.shape[0]
        alphas = z.new_tensor([s / num_summands for s in range(1, num_summands + 1)])  # {1/S..1}
        scaled = (alphas.view(num_summands, 1, 1, 1, 1) * z.unsqueeze(0)).reshape(
            num_summands * B, *z.shape[1:])
        for r in recorders.values():
            r.reset()
        logits = model.model(scaled)                              # [S*B, C]; hooks capture outputs
        full_logits = logits.view(num_summands, B, -1)[-1]        # alpha=1 block = logits(x)
        acc, class_idx = F.softmax(full_logits, dim=1), full_logits.argmax(dim=1)
        di = class_idx if desired_index is None else desired_index
        sel = logits.gather(1, di.repeat(num_summands).view(-1, 1)).sum()
        names = [n for n, *_ in coupling_layers]
        A = {n: recorders[n].out for n in names}
        grads = torch.autograd.grad(sel, [A[n] for n in names], create_graph=create_graph)
        ligs = {}
        for (name, mod, kind), g in zip(coupling_layers, grads):
            gk = g.view(num_summands, B, *g.shape[1:]).mean(0)           # mean path grad
            ak = A[name].view(num_summands, B, *A[name].shape[1:])[-1]   # a_k(x)  (alpha=1)
            ak0 = baseline_acts[name]                                    # a_k(0)  (broadcast over B)
            ligs[name] = _reduce_normalize((ak - ak0) * gk, kind)        # [B, *spatial] sum-to-1
    return ligs, acc, class_idx


# --------------------------------------------------------------------------- #
# GRILL-IG attack (projected Adam, dense pairwise coupling of Layer-IG distortions)
# --------------------------------------------------------------------------- #
def grill_ig_attack(model, x, eps, steps, lr, prefactors, num_summands,
                    coupling_layers, recorders, baseline_acts, beta_modules, beta_growth):
    """Returns x_adv (best class-preserving iterate per sample), org_idx (clean class)."""
    change_beta(beta_modules, 1000.0 if beta_growth else None)
    names = [n for n, *_ in coupling_layers]

    clean_ligs, org_acc, org_idx = get_layer_ig(
        model, x, None, coupling_layers, recorders, baseline_acts, num_summands, False)
    clean_ligs = {k: v.detach() for k, v in clean_ligs.items()}
    org_acc = org_acc.detach()

    # d_k = 0 at delta=0  ->  random start in the eps-ball (as in the reversed-Dombrowski /
    # mass-centre / cosine untargeted variants)
    delta = (torch.rand_like(x) * 2 - 1) * eps
    delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    opt = torch.optim.Adam([delta], lr=lr)

    best_adv = x.clone()
    best_obj = torch.full((len(x),), float("inf"), device=x.device)  # minimise -L_couple (+out)

    for step in range(steps + 1):
        if beta_growth:
            change_beta(beta_modules, get_beta(step, steps))

        adv_ligs, adv_acc, class_idx = get_layer_ig(
            model, x + delta, org_idx, coupling_layers, recorders, baseline_acts,
            num_summands, create_graph=True)

        # per-layer distortion d_k (>=0), dense pairwise coupling
        D = torch.stack([((adv_ligs[n] - clean_ligs[n]) ** 2).flatten(1).sum(1).clamp_min(1e-12).sqrt()
                         for n in names], dim=0)               # [K, B]
        L_couple = 0.5 * (D.sum(0) ** 2 - (D ** 2).sum(0))     # [B] = sum_{i<j} d_i d_j
        loss_output = ((adv_acc - org_acc) ** 2).mean(1)       # [B]
        obj = -prefactors[0] * L_couple + prefactors[1] * loss_output  # [B], minimised

        with torch.no_grad():
            keep = (class_idx == org_idx) & (obj < best_obj)
            best_adv[keep] = (x + delta)[keep].detach()
            best_obj[keep] = obj[keep]

        if step == steps:
            break

        opt.zero_grad(set_to_none=True)
        obj.sum().backward()
        opt.step()
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x + delta, 0, 1) - x)

    change_beta(beta_modules, None)
    return best_adv, org_idx


# --------------------------------------------------------------------------- #
# Similarity metrics (same as the baseline files)
# --------------------------------------------------------------------------- #
def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(a, b, data_range=1.0):
    C = a.shape[1]
    w = _gauss_window(device=a.device).expand(C, 1, -1, -1)
    mu_a, mu_b = F.conv2d(a, w, groups=C), F.conv2d(b, w, groups=C)
    s_aa = F.conv2d(a * a, w, groups=C) - mu_a ** 2
    s_bb = F.conv2d(b * b, w, groups=C) - mu_b ** 2
    s_ab = F.conv2d(a * b, w, groups=C) - mu_a * mu_b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (s_aa + s_bb + c2))
    return m.flatten(1).mean(1)


def pcc(a, b):
    a, b = a.flatten(1), b.flatten(1)
    a, b = a - a.mean(1, keepdim=True), b - b.mean(1, keepdim=True)
    return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-12)


def spearman(a, b):
    ra = a.flatten(1).argsort(1).argsort(1).float()
    rb = b.flatten(1).argsort(1).argsort(1).float()
    return pcc(ra, rb)


def topk_intersection(a, b, k):
    ia, ib = a.flatten(1).topk(k, 1).indices, b.flatten(1).topk(k, 1).indices
    return torch.tensor([len(set(ia[i].tolist()) & set(ib[i].tolist())) / k
                         for i in range(len(a))], device=a.device)


def expl_metrics(h1, h2, k):
    m1 = (h1 / h1.flatten(1).max(1).values.view(-1, 1, 1)).unsqueeze(1)
    m2 = (h2 / h2.flatten(1).max(1).values.view(-1, 1, 1)).unsqueeze(1)
    return {"mse": ((h1 - h2) ** 2).flatten(1).mean(1), "pcc": pcc(h1, h2), "ssim": ssim(m1, m2),
            "spearman": spearman(h1, h2), "topk": topk_intersection(h1, h2, k)}


# --------------------------------------------------------------------------- #
# Data (no target pairing: GRILL-IG is purely untargeted)
# --------------------------------------------------------------------------- #
def numeric_key(filename):
    m = re.match(r"(\d+)", filename)
    return (int(m.group(1)) if m else float("inf"), filename)


def load_labels(label_path):
    labels = {}
    with open(label_path, newline="") as fh:
        first = fh.readline()
        fh.seek(0)
        if "," in first:
            for row in csv.reader(fh):
                if not row or row[0].strip().lower() in ("filename", "file", "image"):
                    continue
                labels[row[0].strip()] = int(row[1])
        else:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[1])
    return labels


class LabelledImages(Dataset):
    def __init__(self, data_dir, labels, transform, max_images=None):
        files = [f for f in os.listdir(data_dir)
                 if f.lower().endswith(IMG_EXTENSIONS) and f in labels]
        files = sorted(files, key=numeric_key)
        if max_images:
            files = files[:max_images]
        self.data_dir, self.files, self.labels, self.transform = data_dir, files, labels, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        img = self.transform(Image.open(os.path.join(self.data_dir, f)).convert("RGB"))
        return img, self.labels[f], f


# --------------------------------------------------------------------------- #
# Plots (publication style; pure untargeted -> only the adv-vs-clean pair)
# --------------------------------------------------------------------------- #
COL1, COL2 = 3.5, 7.16
BLUE, GREEN = "#0072B2", "#009E73"
METRIC_LABELS = {"mse": "MSE", "pcc": "PCC", "ssim": "SSIM", "spearman": "Spearman",
                 "topk": "Top-k inters."}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.0,
        "lines.markersize": 3.5, "grid.linewidth": 0.4, "grid.alpha": 0.4,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    import logging
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    return plt


def _save(fig, path_no_ext):
    plt = _plt()
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png")
    plt.close(fig)


def _img_np(t):
    return t.permute(1, 2, 0).cpu().float().numpy().clip(0, 1)


def _map_np(h):
    h = h.cpu().float().numpy()
    return h / (h.max() + 1e-12)


def _pert_np(x, x_adv):
    pert = x_adv - x
    return _img_np(pert / (2 * pert.abs().max() + 1e-12) + 0.5)


def save_qualitative(folder, x, x_adv, h_o, h_a, caption=""):
    plt = _plt()
    os.makedirs(folder, exist_ok=True)
    panels = [("Clean", _img_np(x), None), ("Perturbed", _img_np(x_adv), None),
              ("Perturbation (amplified)", _pert_np(x, x_adv), None),
              ("IG clean", _map_np(h_o), "jet"), ("IG perturbed", _map_np(h_a), "jet"),
              (r"$|$IG pert. $-$ IG clean$|$", _map_np((h_a - h_o).abs()), "hot")]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, COL2 / 3 * 2 + 0.25),
                             gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for ax, (cap, im, cmap) in zip(axes.flatten(), panels):
        ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                  interpolation="nearest")
        ax.set_title(cap, pad=2)
        ax.axis("off")
    _save(fig, os.path.join(folder, "overview"))
    for fname, (_, im, cmap) in zip(
            ["clean", "perturbed", "perturbation", "ig_clean", "ig_perturbed", "ig_difference"],
            panels):
        plt.imsave(os.path.join(folder, f"{fname}.png"), im, cmap=cmap,
                   vmin=0 if cmap else None, vmax=1 if cmap else None)
    with open(os.path.join(folder, "caption.txt"), "w") as fh:
        fh.write(caption + "\n")


def save_qualitative_grid(path_no_ext, samples, max_rows=5):
    if not samples:
        return
    plt = _plt()
    samples = samples[:max_rows]
    cols = ["Clean", "Perturbed", "IG clean", "IG perturbed", r"$|$IG diff$|$"]
    n = len(samples)
    cell = COL2 / len(cols)
    fig, axes = plt.subplots(n, len(cols), figsize=(COL2, cell * n + 0.15),
                             gridspec_kw=dict(wspace=0.02, hspace=0.02), squeeze=False)
    for r, s in enumerate(samples):
        ims = [(_img_np(s["x"]), None), (_img_np(s["x_adv"]), None),
               (_map_np(s["h_o"]), "jet"), (_map_np(s["h_a"]), "jet"),
               (_map_np((s["h_a"] - s["h_o"]).abs()), "hot")]
        for c, (im, cmap) in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], pad=2)
        axes[r, 3].text(0.03, 0.04, f"PCC {s['pcc_ao']:.2f}", transform=axes[r, 3].transAxes,
                        fontsize=6.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
    _save(fig, path_no_ext)
    with open(path_no_ext + "_caption.txt", "w") as fh:
        fh.write("Rows: " + "; ".join(
            f"{s['name']} ({s['pred_o']} -> {s['pred_a']}, PCC(clean,pert)={s['pcc_ao']:.3f})"
            for s in samples) + "\n")


def save_quantitative_eps(folder, cols, eps_255, mnames):
    plt = _plt()
    from matplotlib.ticker import MaxNLocator
    os.makedirs(folder, exist_ok=True)

    fig, axes = plt.subplots(1, len(mnames), figsize=(COL2, 1.55))
    for ax, m in zip(axes, mnames):
        bp = ax.boxplot([cols[f"adv_vs_orig_{m}"]], showfliers=False, widths=0.6,
                        patch_artist=True, medianprops=dict(color="black", lw=0.9),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6), boxprops=dict(lw=0.6))
        bp["boxes"][0].set_facecolor(BLUE)
        ax.set_xticks([])
        ax.set_title(METRIC_LABELS[m], pad=2)
        if m == "mse":
            ax.set_yscale("log")
        else:
            ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0)
        ax.grid(axis="y")
    fig.suptitle("perturbed vs. clean IG", y=1.04, fontsize=8)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    _save(fig, os.path.join(folder, "explanation_metrics_boxplot"))

    hist = [("adv_vs_orig_pcc", "PCC(IG clean, IG pert.)", BLUE),
            ("adv_vs_orig_topk", "Top-k inters.(clean, pert.)", BLUE),
            ("adv_vs_orig_spearman", "Spearman(clean, pert.)", BLUE),
            ("output_change_l2", r"$\|g(x_{adv})-g(x)\|_2$", GREEN),
            ("linf_x255", r"$\|\delta\|_\infty \times 255$", GREEN),
            ("img_ssim", "SSIM(clean, pert. image)", GREEN)]
    fig, axes = plt.subplots(2, 3, figsize=(COL2, 2.9))
    for ax, (k, t, c) in zip(axes.flatten(), hist):
        ax.hist(cols[k], bins=25, color=c, alpha=0.85, edgecolor="white", lw=0.3)
        med = float(np.median(cols[k]))
        ax.axvline(med, color="black", ls="--", lw=0.8, label=f"median {med:.3g}")
        ax.set_xlabel(t)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc="best", handlelength=1.2)
        ax.grid(axis="y")
    axes[0, 0].set_ylabel("# images")
    axes[1, 0].set_ylabel("# images")
    fig.tight_layout(pad=0.2, w_pad=0.8, h_pad=0.6)
    _save(fig, os.path.join(folder, "distributions"))

    with open(os.path.join(folder, "caption.txt"), "w") as fh:
        fh.write(f"eps = {eps_255}/255, n = {len(cols['adv_vs_orig_pcc'])} images.\n")


def save_quantitative_summary(folder, summary):
    if len(summary) < 2:
        return
    plt = _plt()
    os.makedirs(folder, exist_ok=True)
    eps = [s["eps_x255"] for s in summary]
    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.6), sharex=True)
    for ax, m in zip(axes, ["pcc", "ssim", "spearman", "topk"]):
        ax.plot(eps, [s[f"median_adv_vs_orig_{m}"] for s in summary], marker="o", color=BLUE)
        ax.set_xlabel(r"$\epsilon$ ($\times$1/255)")
        ax.set_title(f"median {METRIC_LABELS[m]}(pert,clean)", pad=2)
        ax.set_xticks(eps)
        ax.set_ylim(0, 1.0)
        ax.grid()
    fig.tight_layout(pad=0.2, w_pad=0.6)
    _save(fig, os.path.join(folder, "metrics_vs_eps"))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="convnext_base",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--data_dir", type=str, default="data/imagenet_val_1000")
    p.add_argument("--labels", type=str, default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", type=str, default=None,
                   help="default results/ig_grill_attack_<model>")
    p.add_argument("--eps", type=float, nargs="+", default=[8.0],
                   help="L_inf bound(s) in units of 1/255, e.g. --eps 2 4 8")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1, help="Adam lr = lr_factor * eps")
    p.add_argument("--prefactors", nargs=2, type=float, default=[1.0, 1e6],
                   help="(prefactor_expl for -L_couple, prefactor_out for output MSE). "
                        "Re-check on a few images; L_couple is O(1-10), output MSE O(1e-6).")
    p.add_argument("--num_summands", type=int, default=30, help="IG/Layer-IG steps")
    p.add_argument("--layers", nargs="+", default=None,
                   help="subset of coupling-layer names (default: all intermediate layers)")
    p.add_argument("--no_beta_growth", action="store_true",
                   help="disable Dombrowski beta growth (no effect on GELU models)")
    p.add_argument("--topk", type=int, default=1000, help="k for top-k intersection metric")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_expl", action="store_true")
    p.add_argument("--n_qual", type=int, default=10)
    p.add_argument("--grid_rows", type=int, default=5)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = args.out_dir or f"results/ig_grill_attack_{args.model}"
    os.makedirs(out_dir, exist_ok=True)

    model, pixel_tf, categories, beta_modules, crop = build_model(
        args.model, device, check_mha=(args.model == "vit_b_16"))
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0

    coupling_layers = select_coupling_layers(args.model, model.model)
    if args.layers:
        wanted = set(args.layers)
        coupling_layers = [c for c in coupling_layers if c[0] in wanted]
        if not coupling_layers:
            raise SystemExit(f"--layers matched nothing; available: "
                             f"{[c[0] for c in select_coupling_layers(args.model, model.model)]}")
    recorders, handles = attach_recorders(coupling_layers, retain=False)
    baseline_acts = precompute_baseline_acts(model, coupling_layers, recorders, device, crop)

    dataset = LabelledImages(args.data_dir, load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) < 1:
        raise SystemExit("No labelled images - check --data_dir and --labels.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    print(f"Model: {args.model} | device: {device} | images: {len(dataset)} | "
          f"IG summands: {args.num_summands} | steps: {args.steps} | "
          f"prefactors: {args.prefactors} | beta growth: {'on' if beta_growth else 'off'}")
    print(f"Coupling layers ({len(coupling_layers)}): {[c[0] for c in coupling_layers]}")

    mnames = ["mse", "pcc", "ssim", "spearman", "topk"]
    qual_dir = os.path.join(out_dir, "qualitative")
    quant_dir = os.path.join(out_dir, "quantitative")
    n_qual = 0 if args.no_plots else min(args.n_qual, len(dataset))
    qual_names = set(dataset.files[i] for i in random.Random(args.seed).sample(range(len(dataset)), n_qual))

    summary = []
    for eps_255 in args.eps:
        eps = eps_255 / 255.0
        lr = args.lr_factor * eps
        print(f"\n{'=' * 70}\neps = {eps_255}/255 = {eps:.5f}   (Adam lr = {lr:.6f})\n{'=' * 70}")

        rows, store, qual_samples, cols = [], {}, [], {}
        n = n_clean = n_unchanged = 0
        tag = f"eps{eps_255:g}".replace(".", "p")
        t0 = time.time()

        for b, (x, y, names) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            x_adv, org_idx = grill_ig_attack(
                model, x, eps, args.steps, lr, args.prefactors, args.num_summands,
                coupling_layers, recorders, baseline_acts, beta_modules, beta_growth)

            h_o, acc_o, pred_o = get_expl_ig(model, x, num_summands=args.num_summands)
            h_a, acc_a, pred_a = get_expl_ig(model, x_adv, num_summands=args.num_summands)
            h_o, h_a = h_o.detach(), h_a.detach()

            with torch.no_grad():
                unchanged = pred_a == pred_o
                out_change = (acc_a - acc_o).norm(dim=1)
                linf = (x_adv - x).abs().flatten(1).max(1).values * 255
                img_mse = ((x_adv - x) ** 2).flatten(1).mean(1)
                img_ssim = ssim(x_adv, x)
                res = expl_metrics(h_a, h_o, args.topk)  # adv vs orig

            n += len(y)
            n_clean += (pred_o == y).sum().item()
            n_unchanged += unchanged.sum().item()

            for i in range(len(y)):
                row = {"filename": names[i], "true_idx": y[i].item(),
                       "true_name": categories[y[i].item()],
                       "clean_pred_idx": pred_o[i].item(), "clean_pred_name": categories[pred_o[i].item()],
                       "adv_pred_idx": pred_a[i].item(), "adv_pred_name": categories[pred_a[i].item()],
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
                    save_qualitative(
                        os.path.join(qual_dir, tag, stem), x[i], x_adv[i], h_o[i], h_a[i],
                        caption=f"{names[i]} | eps={eps_255}/255 | model {args.model} | "
                                f"pred {categories[pred_o[i].item()]} -> {categories[pred_a[i].item()]} | "
                                f"PCC(clean,pert)={row['adv_vs_orig_pcc']:.3f}  "
                                f"top-k(clean,pert)={row['adv_vs_orig_topk']:.3f}")
                    qual_samples.append({
                        "name": names[i], "x": x[i].cpu(), "x_adv": x_adv[i].cpu(),
                        "h_o": h_o[i].cpu(), "h_a": h_a[i].cpu(),
                        "pred_o": categories[pred_o[i].item()], "pred_a": categories[pred_a[i].item()],
                        "pcc_ao": row["adv_vs_orig_pcc"]})

            print(f"  batch {b + 1:3d}/{len(loader)} | label unchanged {n_unchanged / n:.3f} | "
                  f"median PCC(adv,orig) {np.median(cols['adv_vs_orig_pcc']):.3f} | "
                  f"median top-k(adv,orig) {np.median(cols['adv_vs_orig_topk']):.3f}")

        elapsed = time.time() - t0
        med = {k: float(np.median(v)) for k, v in cols.items()}
        mean = {k: float(np.mean(v)) for k, v in cols.items()}

        print(f"\n  Images                       : {n}   ({elapsed:.1f}s)")
        print(f"  Clean accuracy               : {n_clean / n:.4f}")
        print(f"  Decision unchanged           : {n_unchanged / n:.4f}")
        print(f"  median ||g(x_adv)-g(x)||     : {med['output_change_l2']:.2e}")
        print(f"  L_inf of delta (x255)        : median {med['linf_x255']:.3f} (bound {eps_255})")
        print(f"  Image SSIM(x, x_adv)         : median {med['img_ssim']:.4f}")
        print(f"  adv vs orig IG  : PCC {med['adv_vs_orig_pcc']:.3f}  SSIM {med['adv_vs_orig_ssim']:.3f}  "
              f"Spearman {med['adv_vs_orig_spearman']:.3f}  top-k {med['adv_vs_orig_topk']:.3f}")

        if not args.no_plots:
            save_quantitative_eps(os.path.join(quant_dir, tag), cols, eps_255, mnames)
            save_qualitative_grid(os.path.join(qual_dir, tag, "grid"), qual_samples, args.grid_rows)

        with open(os.path.join(out_dir, f"per_image_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if args.save_expl:
            torch.save(store, os.path.join(out_dir, f"ig_grill_attack_{tag}.pt"))

        summary.append({"eps_x255": eps_255, "n_images": n, "clean_acc": n_clean / n,
                        "decision_unchanged": n_unchanged / n, "seconds": round(elapsed, 1),
                        **{f"median_{k}": v for k, v in med.items()},
                        **{f"mean_{k}": v for k, v in mean.items()}})

    for h in handles:
        h.remove()

    print(f"\n{'=' * 70}\nSummary ({args.model}, GRILL-IG untargeted explanation attack)\n{'=' * 70}")
    print(f"{'eps(/255)':>10}{'unchanged':>11}{'PCC a-o':>9}{'Spear a-o':>11}{'top-k a-o':>11}{'img SSIM':>10}")
    for s in summary:
        print(f"{s['eps_x255']:>10g}{s['decision_unchanged']:>11.3f}"
              f"{s['median_adv_vs_orig_pcc']:>9.3f}{s['median_adv_vs_orig_spearman']:>11.3f}"
              f"{s['median_adv_vs_orig_topk']:>11.3f}{s['median_img_ssim']:>10.4f}")

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    if not args.no_plots:
        save_quantitative_summary(quant_dir, summary)
    print(f"\nResults -> {out_dir}")


if __name__ == "__main__":
    main()
