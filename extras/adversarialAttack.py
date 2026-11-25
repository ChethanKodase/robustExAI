"""
Untargeted, sample-specific L_inf adversarial attack on ConvNeXt-Base (ImageNet).

Question answered
-----------------
    Given a maximum L_inf perturbation eps, how many images can be misclassified?
    -> robust accuracy at eps, and attack success rate.

Attack
------
    * Each image gets its OWN perturbation delta (sample-specific, not universal).
      Images are processed in mini-batches only for speed: the loss is a sum of
      per-sample losses and the model is in eval() mode, so every image is
      attacked independently.
    * Perturbation lives in [0,1] pixel space; ImageNet mean/std normalisation is
      part of the model. Constraint:  ||delta||_inf <= eps  and  x+delta in [0,1].
    * Optimiser: Adam on delta, followed by projection onto the eps-ball after
      every step (projected Adam).
    * Loss (maximised):
        margin : max_{j != y} z_j - z_y      (default; > 0  <=>  misclassified)
        ce     : cross-entropy with the true label
    * Best-iterate tracking: an image counts as fooled if ANY iterate within the
      bound misclassifies it; that iterate is kept as its adversarial example.
    * Optional random restarts: an image is robust only if no restart fools it.

Metrics
-------
    clean accuracy          = correct on x            / all images
    robust accuracy         = correct on x_adv        / all images  (clean errors count as failures)
    attack success rate     = fooled / clean-correct images
    + mean / max L_inf of the final perturbations (sanity check: <= eps)

Data
----
    A folder of images plus labels.csv (filename,label[,class_name]),
    label = ImageNet-1k class index 0-999.

Usage
-----
cd /data1/chethan/robustExAI
uv run python adversarialAttack.py \
    --data_dir data/imagenet_val_1000 \
    --labels   data/imagenet_val_1000/labels.csv \
    --eps 0.5 1 2 4 \
    --steps 1000 --batch_size 32
(eps values are given in units of 1/255)



How to activate uv environment 

cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate


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

    def forward(self, x):  # x in [0,1]
        return self.model((x - self.mean) / self.std)


def build_model(device):
    weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
    tf = weights.transforms()  # resize 232 -> crop 224, mean/std
    base = convnext_base(weights=weights).eval()
    model = NormalizedModel(base, tf.mean, tf.std).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Same resize/crop as the pretrained weights, but NO normalisation
    pixel_tf = transforms.Compose([
        transforms.Resize(tf.resize_size, interpolation=tf.interpolation, antialias=True),
        transforms.CenterCrop(tf.crop_size),
        transforms.ToTensor(),  # -> [0,1]
    ])
    return model, pixel_tf, weights.meta["categories"]


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
        img = Image.open(os.path.join(self.data_dir, f)).convert("RGB")
        return self.transform(img), self.labels[f], f


# --------------------------------------------------------------------------- #
# Attack
# --------------------------------------------------------------------------- #
def margin_loss(logits, y):
    """Per-sample  max_{j!=y} z_j - z_y   (positive <=> misclassified)."""
    true_logit = logits.gather(1, y[:, None]).squeeze(1)
    other = logits.clone()
    other.scatter_(1, y[:, None], float("-inf"))
    return other.max(dim=1).values - true_logit


def per_sample_loss(logits, y, loss_type):
    if loss_type == "margin":
        return margin_loss(logits, y)
    return F.cross_entropy(logits, y, reduction="none")


def adam_linf_attack(model, x, y, eps, steps, lr, loss_type, restarts):
    """
    Untargeted projected-Adam attack with per-sample perturbations.

    Returns x_adv (best iterate per sample) and a bool tensor `fooled`.
    """
    best_adv = x.clone()
    fooled = torch.zeros(len(x), dtype=torch.bool, device=x.device)
    best_score = torch.full((len(x),), -float("inf"), device=x.device)

    for r in range(restarts):
        # uniform random start inside the eps-ball (feasible from step 0)
        delta = (torch.rand_like(x) * 2 - 1) * eps
        delta = (torch.clamp(x + delta, 0, 1) - x).detach().requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=lr)

        for step in range(steps + 1):
            logits = model(x + delta)
            losses = per_sample_loss(logits, y, loss_type)

            # ---- best-iterate bookkeeping (before the update) ----
            with torch.no_grad():
                wrong = logits.argmax(1) != y
                score = margin_loss(logits, y)  # used to rank non-fooled iterates
                # prefer: any fooled iterate > highest margin among non-fooled
                improve = (wrong & ~fooled) | (~fooled & ~wrong & (score > best_score))
                best_adv[improve] = (x + delta)[improve].detach()
                best_score[improve] = score[improve]
                fooled |= wrong

            if step == steps:
                break

            # ---- maximise loss: Adam minimises, so negate ----
            opt.zero_grad(set_to_none=True)
            (-losses.sum()).backward()
            opt.step()
            print("losses.sum().item()", losses.sum().item())
            # ---- projection: L_inf ball and valid image range ----
            with torch.no_grad():
                delta.clamp_(-eps, eps)
                delta.copy_(torch.clamp(x + delta, 0, 1) - x)

    return best_adv, fooled


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", type=str, default="data/imagenet_val_1000")
    p.add_argument("--labels", type=str, default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", type=str, default="results/adv_convnext_base")
    p.add_argument("--eps", type=float, nargs="+", default=[4.0],
                   help="L_inf bound(s) in units of 1/255, e.g. --eps 0.5 1 2 4")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr_factor", type=float, default=0.25,
                   help="Adam lr = lr_factor * eps")
    p.add_argument("--loss", choices=["margin", "ce"], default="margin")
    p.add_argument("--restarts", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_images", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_adv", action="store_true",
                   help="save adversarial images (float tensors, exact) per eps as .pt")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    model, pixel_tf, categories = build_model(device)
    dataset = LabelledImages(args.data_dir, load_labels(args.labels), pixel_tf, args.max_images)
    if len(dataset) == 0:
        raise SystemExit("No labelled images found - check --data_dir and --labels.")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")

    print(f"Model: ConvNeXt-Base | device: {device} | images: {len(dataset)} | "
          f"loss: {args.loss} | steps: {args.steps} | restarts: {args.restarts}")

    summary = []
    for eps_255 in args.eps:
        eps = eps_255 / 255.0
        lr = args.lr_factor * eps
        print(f"\n{'=' * 70}\neps = {eps_255}/255 = {eps:.5f}   (Adam lr = {lr:.6f})\n{'=' * 70}")

        rows, adv_store = [], {}
        n = n_clean = n_robust = 0
        linf_all = []
        t0 = time.time()

        for b, (x, y, names) in enumerate(loader):
            x, y = x.to(device), y.to(device)

            with torch.no_grad():
                clean_logits = model(x)
            clean_pred = clean_logits.argmax(1)
            clean_ok = clean_pred == y

            x_adv, _ = adam_linf_attack(model, x, y, eps, args.steps, lr,
                                        args.loss, args.restarts)

            # final, independent re-check of the stored adversarial examples
            with torch.no_grad():
                adv_logits = model(x_adv)
            adv_pred = adv_logits.argmax(1)
            adv_conf = adv_logits.softmax(1).max(1).values
            robust_ok = (adv_pred == y)
            linf = (x_adv - x).abs().flatten(1).max(1).values * 255

            n += len(y)
            n_clean += clean_ok.sum().item()
            n_robust += robust_ok.sum().item()
            linf_all.append(linf.cpu())

            for i in range(len(y)):
                rows.append([names[i], y[i].item(), categories[y[i].item()],
                             clean_pred[i].item(), categories[clean_pred[i].item()],
                             adv_pred[i].item(), categories[adv_pred[i].item()],
                             f"{adv_conf[i].item():.4f}",
                             int(clean_ok[i].item()), int(robust_ok[i].item()),
                             f"{linf[i].item():.4f}"])
                if args.save_adv:
                    adv_store[names[i]] = x_adv[i].detach().cpu().half()

            print(f"  batch {b + 1:3d}/{len(loader)} | "
                  f"clean acc {n_clean / n:.4f} | robust acc {n_robust / n:.4f}")

        linf_all = torch.cat(linf_all)
        clean_acc = n_clean / n
        robust_acc = n_robust / n
        asr = 1 - (n_robust / n_clean) if n_clean else float("nan")
        elapsed = time.time() - t0

        print(f"\n  Images               : {n}   ({elapsed:.1f}s)")
        print(f"  Clean accuracy       : {clean_acc:.4f}")
        print(f"  Robust accuracy      : {robust_acc:.4f}")
        print(f"  Attack success rate  : {asr:.4f}   (on clean-correct images)")
        print(f"  Misclassified images : {n - n_robust} / {n}")
        print(f"  L_inf of delta (x255): mean {linf_all.mean():.3f}  max {linf_all.max():.3f}  (bound {eps_255})")

        tag = f"eps{eps_255:g}".replace(".", "p")
        with open(os.path.join(args.out_dir, f"per_image_{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["filename", "true_idx", "true_name", "clean_pred_idx", "clean_pred_name",
                        "adv_pred_idx", "adv_pred_name", "adv_confidence",
                        "clean_correct", "robust_correct", "linf_x255"])
            w.writerows(rows)
        if args.save_adv:
            torch.save(adv_store, os.path.join(args.out_dir, f"adv_images_{tag}.pt"))

        summary.append([eps_255, n, clean_acc, robust_acc, asr, n - n_robust,
                        linf_all.mean().item(), linf_all.max().item(), elapsed])

    # ---- summary over eps ----
    print(f"\n{'=' * 70}\nSummary (ConvNeXt-Base, untargeted L_inf, projected Adam, {args.loss} loss)\n{'=' * 70}")
    print(f"{'eps(/255)':>10}{'clean acc':>12}{'robust acc':>12}{'ASR':>10}{'#fooled':>10}")
    for s in summary:
        print(f"{s[0]:>10g}{s[2]:>12.4f}{s[3]:>12.4f}{s[4]:>10.4f}{s[5]:>10d}")

    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["eps_x255", "n_images", "clean_acc", "robust_acc", "attack_success_rate",
                    "n_misclassified", "mean_linf_x255", "max_linf_x255", "seconds"])
        for s in summary:
            w.writerow([s[0], s[1], f"{s[2]:.6f}", f"{s[3]:.6f}", f"{s[4]:.6f}", s[5],
                        f"{s[6]:.4f}", f"{s[7]:.4f}", f"{s[8]:.1f}"])
    print(f"\nResults -> {args.out_dir}")


if __name__ == "__main__":
    main()
