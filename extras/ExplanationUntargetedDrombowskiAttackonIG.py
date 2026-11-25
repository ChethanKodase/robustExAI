"""
UNTARGETED (reversed-Dombrowski) sample-specific L_inf explanation attack on
Integrated Gradients (IG) for ConvNeXt-Base (ImageNet).

*** Untargeted version of ExplanationAttackonIG.py (minimal changes) ***
    Targeted (ExplanationAttackonIG.py, Dombrowski Eq. 4):
        minimise   1e11 * MSE( h(x_adv), h_target ) + 1e6 * MSE( g(x_adv), g(x) )
    Untargeted (this file): the image's OWN clean map h(x) is the "target" and the
    sign of the explanation term is reversed (the output term keeps its sign):
        minimise  -1e11 * MSE( h(x_adv), h(x) )   + 1e6 * MSE( g(x_adv), g(x) )
    i.e. push the IG map as far as possible from the clean map while the network
    output (and decision) stays the same.
    Random start: at delta = 0 the attacked map equals the clean map, which is the
    minimum of the MSE, so its gradient is exactly zero and Adam would never move.
    delta therefore starts uniformly at random in the eps-ball (as in PGD), instead
    of Dombrowski's x_adv = x.clone().
    Caveat: the maps sum to 1, so the MSE can also grow by making the map more
    peaked without moving it; judge the result with the "adv_vs_orig" PCC /
    Spearman / top-k metrics, not with the MSE alone.
    Everything else (prefactors, eps bound, projected Adam, decision constraint,
    best iterate = largest MSE among class-preserving iterates, metrics, plots) is
    unchanged. The target image x_t is still loaded, but only for the reference
    columns "orig_vs_target" / "adv_vs_target" and the target panels in the plots;
    it is NOT used by the attack.
    Everything below this block describes the original targeted file.

Same set-up as adversarialAttack.py (data, model wrapper, eps sweep, projected Adam,
best-iterate tracking, per-image CSV + summary). Only the attack OBJECTIVE is
different: instead of causing a misclassification, the attack changes the IG
explanation while keeping the model decision unchanged.

Objective (Dombrowski et al., NeurIPS 2019, Eq. 4 / src/run_attack.py)
-------------------------------------------------------------
    minimise   prefactor_expl * MSE( h(x_adv), h_target )
             + prefactor_out  * MSE( g(x_adv), g(x) )        g = softmax output
    h(.)      : IG explanation of the ORIGINAL predicted class (desired_index = org_idx)
    h_target  : IG explanation of a different (target) image
    -> the explanation is pushed towards an arbitrary target map while the network
       output (and hence the decision) is kept approximately constant.

Changes w.r.t. Dombrowski, needed to fit the adversarialAttack.py set-up
-----------------------------------------------------------------------
    1. Hard L_inf bound: Dombrowski has NO bound on ||delta|| (only a clamp to a
       valid image; the size of delta is kept small implicitly by the small lr and
       the output term). Here delta is projected onto ||delta||_inf <= eps after
       every Adam step, exactly as in adversarialAttack.py, plus clamp to [0,1].
    2. Hard decision constraint for the reported result: Dombrowski only keeps the
       output close with the soft term (they report that the class was unchanged
       for all their examples). Here the soft term is kept unchanged, AND the best
       iterate is chosen only among iterates whose predicted class equals the
       original prediction; the final result is re-checked and reported
       ("label_unchanged").
    3. Pixel space: images are in [0,1] and normalisation is inside the model.
       IG is computed on the NORMALISED input exactly as Dombrowski does
       (baseline 0 in normalised space = the ImageNet mean colour), so the
       explanation itself is identical to theirs.
    4. Batching: Dombrowski attacks one image; here a batch of images is attacked
       with independent per-sample losses (default batch_size 1). torch DataParallel
       for the 30 IG summands is dropped (single GPU).
    5. Target pairing: Dombrowski picks two random test images (one gives the
       target map). Here image i gets target image (i + N//2) mod N (deterministic).
    6. beta growth (ReLU -> softplus_beta, beta 10 -> 100) is implemented as in
       Dombrowski, but it only has an effect on ReLU networks. ConvNeXt uses GELU,
       which already has a non-zero second derivative, so for ConvNeXt it is a no-op.
    7. Delta starts at 0 (Dombrowski: x_adv = x.clone()); no random restarts.

Outputs (per eps)
-----------------
    per_image_eps*.csv : classification (clean/adv prediction, label unchanged,
                         ||g(x_adv)-g(x)||), perturbation size, image similarity,
                         and explanation similarity  (MSE, PCC, SSIM, Spearman,
                         top-k intersection) for
                           original  vs target     (before attack)
                           attacked  vs target     (did the manipulation succeed?)
                           attacked  vs original   (how much did the explanation change?)
    summary.csv        : medians / means of the above over all images
    --save_expl        : .pt with x_adv and the three heatmaps per image

Plots (folders are created at runtime inside --out_dir)
-------------------------------------------------------
    All figures are publication-ready: column widths (3.5 in / 7.16 in), 7-8 pt serif
    fonts, colour-blind-safe colours, saved as vector .pdf (embedded fonts) + 300 dpi .png,
    no titles (the caption information is written to *caption.txt).

    qualitative/eps*/grid.pdf   paper figure: --grid_rows random images x
                                [clean | perturbed | IG clean | IG perturbed | IG target]
    qualitative/eps*/<image>/   for --n_qual randomly chosen images (same images for
                                every eps, seeded):
        overview.pdf/.png       2x4: clean / perturbed / amplified perturbation / target,
                                IG clean / IG perturbed / |IG pert - IG clean| / IG target
        clean.png, perturbed.png, perturbation.png, target.png,
        ig_clean.png, ig_perturbed.png, ig_difference.png, ig_target.png, caption.txt
    quantitative/eps*/          box plots of explanation-similarity metrics for the
                                three pairs, histograms of PCC / top-k (clean vs
                                perturbed), output change, L_inf, image SSIM,
                                scatter PCC(adv,orig) vs PCC(adv,target)
    quantitative/               metrics vs eps (when more than one eps is run)

Usage
-----
    uv run python ExplanationAttackonIG.py \
        --data_dir data/imagenet_val_1000 \
        --labels   data/imagenet_val_1000/labels.csv \
        --eps 8 --steps 500 --batch_size 1 --max_images 100 --n_qual 10
    (eps values are given in units of 1/255)


export CUDA_VISIBLE_DEVICES=1
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python ExplanationUntargetedDrombowskiAttackonIG.py --max_images 100 --eps 2 4 8 --n_qual 10 --grid_rows 5

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
from torchvision.models import ConvNeXt_Base_Weights, convnext_base

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
# --------------------------------------------------------------------------- #
class NormalizedModel(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def normalize(self, x):  # [0,1] -> normalised space (Dombrowski's input space)
        return (x - self.mean) / self.std

    def forward(self, x):  # x in [0,1]
        return self.model(self.normalize(x))


class BetaActivation(nn.Module):
    """
    ReLU that can be switched to softplus_beta.
    Taken from Dombrowski: src/nn/networks.py, ExplainableNet.__init__ / change_beta
      (activation_fn = F.relu if beta is None else torch.nn.Softplus(beta=beta)).
    """
    def __init__(self):
        super().__init__()
        self.beta = None

    def forward(self, x):
        return F.relu(x) if self.beta is None else F.softplus(x, beta=self.beta)


def replace_relu_with_beta_activation(module):
    """Swap every nn.ReLU for a BetaActivation. Returns the list of swapped modules."""
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
    """Dombrowski: src/nn/networks.py, ExplainableNet.change_beta (None -> ReLU)."""
    for m in beta_modules:
        m.beta = beta


def get_beta(i, num_iter):
    """
    Helper method for beta growth.
    Taken verbatim from Dombrowski: src/run_attack.py, get_beta.
    """
    start_beta, end_beta = 10.0, 100.0
    return start_beta * (end_beta / start_beta) ** (i / num_iter)


def build_model(device):
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    tf = weights.transforms()  # resize 232 -> crop 224, mean/std
    base = convnext_base(weights=weights).eval()
    beta_modules = replace_relu_with_beta_activation(base)  # empty for ConvNeXt (GELU)
    model = NormalizedModel(base, tf.mean, tf.std).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Same resize/crop as the pretrained weights, but NO normalisation
    pixel_tf = transforms.Compose([
        transforms.Resize(tf.resize_size, interpolation=tf.interpolation, antialias=True),
        transforms.CenterCrop(tf.crop_size),
        transforms.ToTensor(),  # -> [0,1]
    ])
    return model, pixel_tf, weights.meta["categories"], beta_modules


# --------------------------------------------------------------------------- #
# Data
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
    """Returns (image, label, name, target_image, target_name).
    Target image (source of the target explanation) = image (i + N//2) mod N."""
    def __init__(self, data_dir, labels, transform, max_images=None):
        files = [f for f in os.listdir(data_dir)
                 if f.lower().endswith(IMG_EXTENSIONS) and f in labels]
        files = sorted(files, key=numeric_key)
        if max_images:
            files = files[:max_images]
        self.data_dir, self.files, self.labels, self.transform = data_dir, files, labels, transform

    def __len__(self):
        return len(self.files)

    def _load(self, f):
        return self.transform(Image.open(os.path.join(self.data_dir, f)).convert("RGB"))

    def __getitem__(self, idx):
        f = self.files[idx]
        tf = self.files[(idx + len(self.files) // 2) % len(self.files)]
        return self._load(f), self.labels[f], f, self._load(tf), tf


# --------------------------------------------------------------------------- #
# Integrated Gradients explanation
# --------------------------------------------------------------------------- #
def get_expl_ig(model, x, desired_index=None, num_summands=30, create_graph=False):
    """
    Integrated Gradients heatmap, batched.

    Taken from Dombrowski: src/nn/utils.py, get_expl(), branch
    `method == ExplainingMethod.integrated_grad`, unchanged except:
      * batched over B images (each image its own desired_index),
      * the input is our [0,1] image, mapped to the normalised space first so IG
        is computed on exactly the same tensor as in Dombrowski (baseline 0 in
        normalised space),
      * no torch.nn.DataParallel.

    Returns normalized_heatmap [B,H,W] (sums to 1), acc = softmax output [B,1000],
    class_idx [B]   (same return values as Dombrowski's get_expl).
    """
    #print("inside get_expl_ig")
    #print("x.shape", x.shape)
    with torch.enable_grad():
        if not x.requires_grad:
            x = x.clone().requires_grad_(True)
        z = model.normalize(x)  # Dombrowski's (normalised) input
        #print("z.shape", z.shape)

        # Dombrowski: acc, class_idx = model.classify(x)
        logits = model.model(z)
        #print("logits.shape", logits.shape)
        acc, class_idx = F.softmax(logits, dim=1), logits.argmax(dim=1)
        #print("acc.shape", acc.shape)
        #print("class_idx", class_idx)
        #print("desired_index", desired_index)
        if desired_index is None:
            desired_index = class_idx
        #print("later desired_index", desired_index)
        # Dombrowski: prefactors = [k / num_summands for k in 1..num_summands]
        B = z.shape[0]
        #print("B", B)
        prefactors = z.new_tensor([k / num_summands for k in range(1, num_summands + 1)])
        #print("prefactors", prefactors)
        #print("shape of prefactors.view(num_summands, 1, 1, 1, 1)", prefactors.view(num_summands, 1, 1, 1, 1).shape )
        #print("prefactors.view(num_summands, 1, 1, 1, 1)", prefactors.view(num_summands, 1, 1, 1, 1) )
        #print("z.unsqueeze(0).shape", z.unsqueeze(0).shape)
        scaled = (prefactors.view(num_summands, 1, 1, 1, 1) * z.unsqueeze(0)).reshape(
            num_summands * B, *z.shape[1:])
        #print("scaled.shape", scaled.shape)
        y = model.model(scaled).view(num_summands, B, -1)
        #print("y.shape", y.shape)

        # Dombrowski: "we sum the result and then take the derivative (instead of summing
        # derivatives as in most implementations),
        # (d/dx) (n*y_1(1/n*x) + n/2*y_1(2/n*x) .... + y_n(x) ) = y_1'+....y'_n"
        y = F.softmax(y, dim=2)

        desIndVw = desired_index.view(1, B, 1)
        #print("desIndVw", desIndVw)
        #print("desIndVw.shape", desIndVw.shape)
        desIndVwExpnd = desIndVw.expand(num_summands, B, 1)
        #print("desIndVwExpnd.shape", desIndVwExpnd.shape)
        yGatherdesIndVwExpnd = y.gather(2, desIndVwExpnd)
        yGatherdesIndVwExpndSqz2 = yGatherdesIndVwExpnd.squeeze(2)
        #print("yGatherdesIndVwExpndSqz2.shape", yGatherdesIndVwExpndSqz2.shape)
        #print("before y.shape", y.shape)
        #print("max among the 1000 shape", y[0,0,:].shape)
        #print("max among the 1000 desired_index", y[0,0,desired_index])
        #print("max among the 1000 desired_index", y[1,0,desired_index])
        #print("max among the 1000 desired_index", y[2,0,desired_index])
        #print("max among the 1000 desired_index", y[3,0,desired_index])
        y = y.gather(2, desired_index.view(1, B, 1).expand(num_summands, B, 1)).squeeze(2)  # [n,B] give what is the score for my class(desired index) among all 1000 classes
        #print("here y.shape", y.shape)
        #print("y", y)
        prefactorViewSummand = prefactors.view(num_summands, 1)
        #print("prefactorView.shape", prefactorViewSummand.shape)
        #print("prefactorViewSummand", prefactorViewSummand)
        y = (1 / num_summands) * torch.sum(y / prefactors.view(num_summands, 1), dim=0)    # [B]
        heatmap = torch.autograd.grad(y.sum(), z, create_graph=create_graph)[0]

        # Dombrowski: heatmap = heatmap * x   (grad_times_input / integrated_grad)
        heatmap = heatmap * z
        # Dombrowski: heatmap = torch.sum(torch.abs(heatmap), dim=1); normalise to sum 1
        heatmap = torch.sum(torch.abs(heatmap), dim=1)
        normalized_heatmap = heatmap / heatmap.flatten(1).sum(1).view(B, 1, 1)

    return normalized_heatmap, acc, class_idx


# --------------------------------------------------------------------------- #
# Attack
# --------------------------------------------------------------------------- #
def ig_explanation_attack(model, x, x_target, eps, steps, lr, prefactors,
                          num_summands, beta_modules, beta_growth):
    """
    Projected-Adam explanation-manipulation attack on IG with per-sample perturbations.

    Taken from Dombrowski: src/run_attack.py, main() (from "# produce expls" to the
    end of the optimisation loop), with the changes listed in the module docstring:
    hard L_inf projection, batched per-sample losses, best-iterate tracking with
    the decision constraint.

    Returns x_adv (best iterate per sample), org_idx (original predicted class).
    """
    # Dombrowski: model = ExplainableNet(..., beta=1000 if args.beta_growth else None)
    #print("beta_modules", beta_modules)
    #print("beta_growth", beta_growth)
    change_beta(beta_modules, 1000.0 if beta_growth else None)

    # ---- produce expls (Dombrowski: "# produce expls") ----
    #print("num_summands", num_summands)
    #print("x.shape", x.shape)
    org_expl, org_acc, org_idx = get_expl_ig(model, x, num_summands=num_summands)
    org_expl, org_acc = org_expl.detach(), org_acc.detach()
    # UNTARGETED: the clean map h(x) itself is the "target"; x_target is not used
    target_expl = org_expl

    # Dombrowski: x_adv = x.clone().detach().requires_grad_()  -> delta starts at 0
    # UNTARGETED: random start in the eps-ball (the MSE gradient is zero at delta = 0)
    delta = (torch.rand_like(x) * 2 - 1) * eps
    delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
    # Dombrowski: optimizer = torch.optim.Adam([x_adv], lr=args.lr)
    opt = torch.optim.Adam([delta], lr=lr)

    best_adv = x.clone()
    best_loss = torch.full((len(x),), float("inf"), device=x.device)

    for step in range(steps + 1):
        # Dombrowski: if args.beta_growth: model.change_beta(get_beta(i, args.num_iter))
        if beta_growth:
            change_beta(beta_modules, get_beta(step, steps))

        # ---- calculate loss (Dombrowski: "# calculate loss") ----
        adv_expl, adv_acc, class_idx = get_expl_ig(model, x + delta, desired_index=org_idx,
                                                   num_summands=num_summands, create_graph=True)
        # Dombrowski: F.mse_loss(adv_expl, target_expl), F.mse_loss(adv_acc, org_acc)
        # -> here per sample (mean over pixels / classes), so each image is independent
        # UNTARGETED: sign reversed -> maximise the distance to the clean map
        loss_expl = -((adv_expl - target_expl) ** 2).flatten(1).mean(1)
        loss_output = ((adv_acc - org_acc) ** 2).mean(1)
        #print("prefactors", prefactors)
        total_loss = prefactors[0] * loss_expl + prefactors[1] * loss_output

        # ---- best-iterate bookkeeping (NEW: decision must stay the original one) ----
        with torch.no_grad():
            keep = (class_idx == org_idx) & (loss_expl < best_loss)
            best_adv[keep] = (x + delta)[keep].detach()
            best_loss[keep] = loss_expl[keep]

        if step == steps:
            break

        # ---- update adversarial example (Dombrowski: total_loss.backward(); optimizer.step()) ----
        opt.zero_grad(set_to_none=True)
        total_loss.sum().backward()
        opt.step()

        # ---- projection (Dombrowski: clamp(x_adv) to a valid image; NEW: L_inf ball) ----
        with torch.no_grad():
            delta.clamp_(-eps, eps)
            delta.copy_(torch.clamp(x + delta, 0, 1) - x)

    # Dombrowski: "# test with original model (with relu activations)" -> change_beta(None)
    change_beta(beta_modules, None)
    return best_adv, org_idx


# --------------------------------------------------------------------------- #
# Similarity metrics
#   MSE / PCC / SSIM as in Dombrowski (Sec. 2.1, "Similarity metrics");
#   Spearman rank correlation and top-k intersection as in Ghorbani et al.
# --------------------------------------------------------------------------- #
def _gauss_window(size=11, sigma=1.5, device="cpu"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-g ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).view(1, 1, size, size)


def ssim(a, b, data_range=1.0):
    """Per-sample SSIM for [B,C,H,W] tensors (Gaussian window 11, sigma 1.5)."""
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
    """h1, h2: heatmaps [B,H,W] normalised to sum 1 (as in Dombrowski)."""
    m1 = (h1 / h1.flatten(1).max(1).values.view(-1, 1, 1)).unsqueeze(1)  # [0,1] for SSIM
    m2 = (h2 / h2.flatten(1).max(1).values.view(-1, 1, 1)).unsqueeze(1)
    return {"mse": ((h1 - h2) ** 2).flatten(1).mean(1), "pcc": pcc(h1, h2), "ssim": ssim(m1, m2),
            "spearman": spearman(h1, h2), "topk": topk_intersection(h1, h2, k)}


# --------------------------------------------------------------------------- #
# Plots  (publication style)
#   * widths: single column 3.5 in, double column 7.16 in (IEEE/NeurIPS/CVPR-like)
#   * 7-8 pt fonts at 100 % scale, Okabe-Ito colour-blind-safe palette
#   * every figure saved as vector PDF (TrueType fonts embedded) + 300 dpi PNG
#   * no figure titles: the information goes into the caption (caption.txt)
# --------------------------------------------------------------------------- #
COL1, COL2 = 3.5, 7.16  # inches
PAIR_COLORS = {"orig_vs_target": "#999999", "adv_vs_target": "#E69F00", "adv_vs_orig": "#0072B2"}
PAIR_LABELS = {"orig_vs_target": "clean vs. target (before attack)",
               "adv_vs_target": "perturbed vs. target",
               "adv_vs_orig": "perturbed vs. clean"}
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
        "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.2,
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
    return h / (h.max() + 1e-12)  # heatmap scaled to [0,1] (Dombrowski: heatmap_to_image)


def _pert_np(x, x_adv):
    pert = x_adv - x
    return _img_np(pert / (2 * pert.abs().max() + 1e-12) + 0.5)  # amplified, grey = no change


def save_qualitative(folder, x, x_adv, x_t, h_o, h_a, h_t, caption=""):
    """
    Qualitative figures for one image (double-column width, 2 x 4 panels).
    Layout follows Dombrowski: src/nn/utils.py, plot_overview (images in the top row,
    explanations in the bottom row, 'jet' colormap), extended with the perturbation
    and the heatmap difference.
    """
    plt = _plt()
    os.makedirs(folder, exist_ok=True)
    panels = [("Clean", _img_np(x), None), ("Perturbed", _img_np(x_adv), None),
              (r"Perturbation (amplified)", _pert_np(x, x_adv), None), ("Target", _img_np(x_t), None),
              ("IG clean", _map_np(h_o), "jet"), ("IG perturbed", _map_np(h_a), "jet"),
              (r"$|$IG pert. $-$ IG clean$|$", _map_np((h_a - h_o).abs()), "hot"),
              ("IG target", _map_np(h_t), "jet")]
    fig, axes = plt.subplots(2, 4, figsize=(COL2, COL2 / 4 * 2 + 0.25),
                             gridspec_kw=dict(wspace=0.03, hspace=0.12))
    for ax, (cap, im, cmap) in zip(axes.flatten(), panels):
        ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                  interpolation="nearest")
        ax.set_title(cap, pad=2)
        ax.axis("off")
    _save(fig, os.path.join(folder, "overview"))

    singles = ["clean", "perturbed", "perturbation", "target",
               "ig_clean", "ig_perturbed", "ig_difference", "ig_target"]
    for fname, (_, im, cmap) in zip(singles, panels):
        plt.imsave(os.path.join(folder, f"{fname}.png"), im, cmap=cmap,
                   vmin=0 if cmap else None, vmax=1 if cmap else None)
    with open(os.path.join(folder, "caption.txt"), "w") as fh:
        fh.write(caption + "\n")


def save_qualitative_grid(path_no_ext, samples, max_rows=5):
    """
    Paper figure: one row per image, columns
    clean | perturbed | IG clean | IG perturbed | IG target ; double-column width.
    samples: list of dicts with tensors x, x_adv, x_t, h_o, h_a, h_t and metrics.
    """
    if not samples:
        return
    plt = _plt()
    samples = samples[:max_rows]
    cols = ["Clean", "Perturbed", "IG clean", "IG perturbed", "IG target"]
    n = len(samples)
    w = COL2
    cell = w / len(cols)
    fig, axes = plt.subplots(n, len(cols), figsize=(w, cell * n + 0.15),
                             gridspec_kw=dict(wspace=0.02, hspace=0.02), squeeze=False)
    for r, s in enumerate(samples):
        ims = [(_img_np(s["x"]), None), (_img_np(s["x_adv"]), None),
               (_map_np(s["h_o"]), "jet"), (_map_np(s["h_a"]), "jet"), (_map_np(s["h_t"]), "jet")]
        for c, (im, cmap) in enumerate(ims):
            ax = axes[r, c]
            ax.imshow(im, cmap=cmap, vmin=0 if cmap else None, vmax=1 if cmap else None,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if r == 0:
                ax.set_title(cols[c], pad=2)
        # PCC between clean and perturbed IG written on the perturbed IG panel
        axes[r, 3].text(0.03, 0.04, f"PCC {s['pcc_ao']:.2f}", transform=axes[r, 3].transAxes,
                        fontsize=6.5, color="white",
                        bbox=dict(facecolor="black", alpha=0.55, pad=1.2, lw=0))
    _save(fig, path_no_ext)
    with open(path_no_ext + "_caption.txt", "w") as fh:
        fh.write("Rows (top to bottom): " + "; ".join(
            f"{s['name']} ({s['pred_o']} -> {s['pred_a']}, PCC(clean,pert)={s['pcc_ao']:.3f}, "
            f"PCC(pert,target)={s['pcc_at']:.3f}, target {s['target']})" for s in samples) + "\n")


def save_quantitative_eps(folder, cols, eps_255, pairs, mnames):
    """Distribution plots of the per-image metrics for one eps."""
    plt = _plt()
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    os.makedirs(folder, exist_ok=True)

    # 1. box plots: one panel per metric, three pairs (double column, one row)
    fig, axes = plt.subplots(1, len(mnames), figsize=(COL2, 1.55))
    for ax, m in zip(axes, mnames):
        bp = ax.boxplot([cols[f"{pr}_{m}"] for pr in pairs], showfliers=False, widths=0.6,
                        patch_artist=True, medianprops=dict(color="black", lw=0.9),
                        whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                        boxprops=dict(lw=0.6))
        for patch, pr in zip(bp["boxes"], pairs):
            patch.set_facecolor(PAIR_COLORS[pr])
        ax.set_xticks([])
        ax.set_title(METRIC_LABELS[m], pad=2)
        if m == "mse":
            ax.set_yscale("log")
        else:
            ax.set_ylim(min(0, ax.get_ylim()[0]), 1.0)
        ax.grid(axis="y")
    fig.legend(handles=[Patch(facecolor=PAIR_COLORS[pr], edgecolor="black", lw=0.6,
                              label=PAIR_LABELS[pr]) for pr in pairs],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.2, w_pad=0.6)
    _save(fig, os.path.join(folder, "explanation_metrics_boxplot"))

    # 2. histograms (double column, 2 x 3)
    hist = [("adv_vs_orig_pcc", "PCC(IG clean, IG pert.)", PAIR_COLORS["adv_vs_orig"]),
            ("adv_vs_orig_topk", "Top-k inters.(clean, pert.)", PAIR_COLORS["adv_vs_orig"]),
            ("adv_vs_target_pcc", "PCC(IG pert., IG target)", PAIR_COLORS["adv_vs_target"]),
            ("output_change_l2", r"$\|g(x_{adv})-g(x)\|_2$", "#009E73"),
            ("linf_x255", r"$\|\delta\|_\infty \times 255$", "#009E73"),
            ("img_ssim", "SSIM(clean, pert. image)", "#009E73")]
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

    # 3. scatter (single column)
    fig, ax = plt.subplots(figsize=(COL1, 2.5))
    ax.scatter(cols["adv_vs_orig_pcc"], cols["adv_vs_target_pcc"], s=6, alpha=0.6,
               color=PAIR_COLORS["adv_vs_orig"], lw=0)
    ax.set_xlabel("PCC(IG perturbed, IG clean)")
    ax.set_ylabel("PCC(IG perturbed, IG target)")
    ax.grid()
    fig.tight_layout(pad=0.2)
    _save(fig, os.path.join(folder, "scatter_clean_vs_target_pcc"))

    with open(os.path.join(folder, "caption.txt"), "w") as fh:
        fh.write(f"eps = {eps_255}/255, n = {len(cols['adv_vs_orig_pcc'])} images. "
                 "Box plots: median, IQR, whiskers 1.5 IQR, outliers hidden.\n")


def save_quantitative_summary(folder, summary, pairs):
    """Median metrics as a function of eps (only meaningful for >1 eps)."""
    if len(summary) < 2:
        return
    plt = _plt()
    os.makedirs(folder, exist_ok=True)
    eps = [s["eps_x255"] for s in summary]
    markers = {"orig_vs_target": "s", "adv_vs_target": "^", "adv_vs_orig": "o"}

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 1.6), sharex=True)
    for ax, m in zip(axes, ["pcc", "ssim", "spearman", "topk"]):
        for pr in pairs:
            ax.plot(eps, [s[f"median_{pr}_{m}"] for s in summary], marker=markers[pr],
                    color=PAIR_COLORS[pr], label=PAIR_LABELS[pr])
        ax.set_xlabel(r"$\epsilon$ ($\times$1/255)")
        ax.set_title(f"median {METRIC_LABELS[m]}", pad=2)
        ax.set_xticks(eps)
        ax.set_ylim(0, 1.0)
        ax.grid()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(pad=0.2, w_pad=0.6)
    _save(fig, os.path.join(folder, "metrics_vs_eps"))

    fig, ax = plt.subplots(figsize=(COL1, 2.0))
    ax.plot(eps, [s["decision_unchanged"] for s in summary], marker="o", color="#0072B2",
            label="decision unchanged")
    ax.plot(eps, [s["median_img_ssim"] for s in summary], marker="s", color="#009E73",
            label="median image SSIM")
    ax.set_xlabel(r"$\epsilon$ ($\times$1/255)")
    ax.set_xticks(eps)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid()
    fig.tight_layout(pad=0.2)
    _save(fig, os.path.join(folder, "decision_and_image_similarity_vs_eps"))

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, default="data/imagenet_val_1000")
    p.add_argument("--labels", type=str, default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", type=str, default="results/ig_untargeted_dombrowski_convnext_base")
    p.add_argument("--eps", type=float, nargs="+", default=[8.0],
                   help="L_inf bound(s) in units of 1/255, e.g. --eps 2 4 8")
    p.add_argument("--steps", type=int, default=500,
                   help="Dombrowski Table 1: 500 iterations for IntGrad")
    p.add_argument("--lr_factor", type=float, default=0.1,
                   help="Adam lr = lr_factor * eps")
    p.add_argument("--prefactors", nargs=2, type=float, default=[1e11, 1e6],
                   help="Dombrowski run_attack.py: prefactors of losses (diff expls, class loss)")
    p.add_argument("--num_summands", type=int, default=30,
                   help="Dombrowski utils.py: IG steps")
    p.add_argument("--no_beta_growth", action="store_true",
                   help="disable Dombrowski's beta growth (no effect on ConvNeXt anyway)")
    p.add_argument("--topk", type=int, default=1000, help="k for top-k intersection")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_expl", action="store_true",
                   help="save x_adv and original/target/attacked heatmaps per eps as .pt")
    p.add_argument("--n_qual", type=int, default=10,
                   help="number of randomly chosen images (seeded) for qualitative plots")
    p.add_argument("--grid_rows", type=int, default=5,
                   help="rows (images) in the combined qualitative paper figure")
    p.add_argument("--no_plots", action="store_true", help="skip all plots")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    model, pixel_tf, categories, beta_modules = build_model(device)
    beta_growth = (not args.no_beta_growth) and len(beta_modules) > 0
    dataset = LabelledImages(args.data_dir, load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) < 2:
        raise SystemExit("Need at least 2 labelled images - check --data_dir and --labels.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    print(f"Model: ConvNeXt-Base | device: {device} | images: {len(dataset)} | "
          f"IG summands: {args.num_summands} | steps: {args.steps} | "
          f"prefactors: {args.prefactors} | beta growth: "
          f"{'on' if beta_growth else 'off (no ReLU in model)' if not beta_modules else 'off'}")

    pairs = ["orig_vs_target", "adv_vs_target", "adv_vs_orig"]
    mnames = ["mse", "pcc", "ssim", "spearman", "topk"]

    # randomly chosen images for qualitative plots (same images for every eps)
    qual_dir, quant_dir = os.path.join(args.out_dir, "qualitative"), os.path.join(args.out_dir, "quantitative")
    n_qual = 0 if args.no_plots else min(args.n_qual, len(dataset))
    qual_names = set(dataset.files[i] for i in random.Random(args.seed).sample(range(len(dataset)), n_qual))
    if qual_names:
        print(f"Qualitative plots for {len(qual_names)} random images: "
              f"{sorted(qual_names, key=numeric_key)}")

    summary = []
    for eps_255 in args.eps:
        eps = eps_255 / 255.0
        lr = args.lr_factor * eps
        print(f"\n{'=' * 70}\neps = {eps_255}/255 = {eps:.5f}   (Adam lr = {lr:.6f})\n{'=' * 70}")

        rows, store, qual_samples = [], {}, []
        cols = {}  # metric name -> list of per-image values
        n = n_clean = n_unchanged = 0
        tag = f"eps{eps_255:g}".replace(".", "p")
        t0 = time.time()

        for b, (x, y, names, x_t, t_names) in enumerate(loader):
            x, y, x_t = x.to(device), y.to(device), x_t.to(device)
            #print("x.shape", x.shape)
            x_adv, org_idx = ig_explanation_attack(
                model, x, x_t, eps, args.steps, lr, args.prefactors,
                args.num_summands, beta_modules, beta_growth)

            # ---- final evaluation with the original model (Dombrowski: change_beta(None);
            #      adv_expl = get_expl(model, x_adv, method)) ----
            h_o, acc_o, pred_o = get_expl_ig(model, x, num_summands=args.num_summands)
            h_t, _, _ = get_expl_ig(model, x_t, num_summands=args.num_summands)
            h_a, acc_a, pred_a = get_expl_ig(model, x_adv, num_summands=args.num_summands)
            h_o, h_t, h_a = h_o.detach(), h_t.detach(), h_a.detach()

            with torch.no_grad():
                unchanged = pred_a == pred_o
                out_change = (acc_a - acc_o).norm(dim=1)  # ||g(x_adv) - g(x)|| (Dombrowski)
                linf = (x_adv - x).abs().flatten(1).max(1).values * 255
                img_mse = ((x_adv - x) ** 2).flatten(1).mean(1)
                img_ssim = ssim(x_adv, x)
                res = {pr: expl_metrics(a, c, args.topk)
                       for pr, (a, c) in zip(pairs, [(h_o, h_t), (h_a, h_t), (h_a, h_o)])}

            n += len(y)
            n_clean += (pred_o == y).sum().item()
            n_unchanged += unchanged.sum().item()

            for i in range(len(y)):
                row = {"filename": names[i], "target_filename": t_names[i],
                       "true_idx": y[i].item(), "true_name": categories[y[i].item()],
                       "clean_pred_idx": pred_o[i].item(), "clean_pred_name": categories[pred_o[i].item()],
                       "adv_pred_idx": pred_a[i].item(), "adv_pred_name": categories[pred_a[i].item()],
                       "clean_correct": int(pred_o[i].item() == y[i].item()),
                       "label_unchanged": int(unchanged[i].item()),
                       "output_change_l2": out_change[i].item(),
                       "linf_x255": linf[i].item(), "img_mse": img_mse[i].item(),
                       "img_ssim": img_ssim[i].item()}
                for pr in pairs:
                    for m in mnames:
                        row[f"{pr}_{m}"] = res[pr][m][i].item()
                rows.append(row)
                for kname in ["output_change_l2", "linf_x255", "img_mse", "img_ssim"] + \
                             [f"{pr}_{m}" for pr in pairs for m in mnames]:
                    cols.setdefault(kname, []).append(row[kname])

                if args.save_expl:
                    store[names[i]] = {"target": t_names[i], "x_adv": x_adv[i].cpu().half(),
                                       "h_orig": h_o[i].cpu(), "h_target": h_t[i].cpu(),
                                       "h_adv": h_a[i].cpu()}
                if names[i] in qual_names:
                    stem = os.path.splitext(names[i])[0]
                    save_qualitative(
                        os.path.join(qual_dir, tag, stem), x[i], x_adv[i], x_t[i], h_o[i], h_a[i], h_t[i],
                        caption=f"{names[i]} (target {t_names[i]}) | eps={eps_255}/255 | "
                                f"pred {categories[pred_o[i].item()]} -> {categories[pred_a[i].item()]} | "
                                f"PCC(clean,pert)={row['adv_vs_orig_pcc']:.3f}  "
                                f"top-k(clean,pert)={row['adv_vs_orig_topk']:.3f}  "
                                f"PCC(pert,target)={row['adv_vs_target_pcc']:.3f}")
                    qual_samples.append({
                        "name": names[i], "target": t_names[i],
                        "x": x[i].cpu(), "x_adv": x_adv[i].cpu(), "x_t": x_t[i].cpu(),
                        "h_o": h_o[i].cpu(), "h_a": h_a[i].cpu(), "h_t": h_t[i].cpu(),
                        "pred_o": categories[pred_o[i].item()], "pred_a": categories[pred_a[i].item()],
                        "pcc_ao": row["adv_vs_orig_pcc"], "pcc_at": row["adv_vs_target_pcc"]})

            print(f"  batch {b + 1:3d}/{len(loader)} | label unchanged {n_unchanged / n:.3f} | "
                  f"median PCC(adv,target) {np.median(cols['adv_vs_target_pcc']):.3f} | "
                  f"median PCC(adv,orig) {np.median(cols['adv_vs_orig_pcc']):.3f}")

        elapsed = time.time() - t0
        med = {k: float(np.median(v)) for k, v in cols.items()}
        mean = {k: float(np.mean(v)) for k, v in cols.items()}

        print(f"\n  Images                       : {n}   ({elapsed:.1f}s)")
        print(f"  Clean accuracy               : {n_clean / n:.4f}")
        print(f"  Decision unchanged           : {n_unchanged / n:.4f}")
        print(f"  median ||g(x_adv)-g(x)||     : {med['output_change_l2']:.2e}")
        print(f"  L_inf of delta (x255)        : median {med['linf_x255']:.3f} (bound {eps_255})")
        print(f"  Image SSIM(x, x_adv)         : median {med['img_ssim']:.4f}")
        print(f"  {'(medians)':<22}{'MSE':>11}{'PCC':>8}{'SSIM':>8}{'Spearman':>10}{'top-k':>8}")
        for pr in pairs:
            print(f"  {pr:<22}{med[pr + '_mse']:>11.2e}{med[pr + '_pcc']:>8.3f}"
                  f"{med[pr + '_ssim']:>8.3f}{med[pr + '_spearman']:>10.3f}{med[pr + '_topk']:>8.3f}")

        if not args.no_plots:
            save_quantitative_eps(os.path.join(quant_dir, tag), cols, eps_255, pairs, mnames)
            save_qualitative_grid(os.path.join(qual_dir, tag, "grid"), qual_samples, args.grid_rows)

        with open(os.path.join(args.out_dir, f"per_image_{tag}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if args.save_expl:
            torch.save(store, os.path.join(args.out_dir, f"ig_attack_{tag}.pt"))

        summary.append({"eps_x255": eps_255, "n_images": n, "clean_acc": n_clean / n,
                        "decision_unchanged": n_unchanged / n, "seconds": round(elapsed, 1),
                        **{f"median_{k}": v for k, v in med.items()},
                        **{f"mean_{k}": v for k, v in mean.items()}})

    # ---- summary over eps ----
    print(f"\n{'=' * 70}\nSummary (ConvNeXt-Base, UNTARGETED reversed-Dombrowski IG explanation attack, projected Adam)\n{'=' * 70}")
    print(f"{'eps(/255)':>10}{'unchanged':>11}{'PCC o-t':>9}{'PCC a-t':>9}{'PCC a-o':>9}"
          f"{'top-k a-o':>11}{'img SSIM':>10}")
    for s in summary:
        print(f"{s['eps_x255']:>10g}{s['decision_unchanged']:>11.3f}"
              f"{s['median_orig_vs_target_pcc']:>9.3f}{s['median_adv_vs_target_pcc']:>9.3f}"
              f"{s['median_adv_vs_orig_pcc']:>9.3f}{s['median_adv_vs_orig_topk']:>11.3f}"
              f"{s['median_img_ssim']:>10.4f}")

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    if not args.no_plots:
        save_quantitative_summary(quant_dir, summary, pairs)
    print(f"\nResults -> {args.out_dir}")


if __name__ == "__main__":
    main()
