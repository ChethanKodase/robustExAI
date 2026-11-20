"""
Clean-accuracy baseline for the three ImageNet classifiers used in the
explanation-robustness experiments:

    1. ConvNeXt-Base  (torchvision, IMAGENET1K_V1)
    2. ViT-B/16       (torchvision, IMAGENET1K_V1)
    3. ResNet-50      (torchvision, IMAGENET1K_V2)

Each model is loaded in turn, every image in --data_dir is classified one by
one in a for loop, and accuracy / precision / recall / F1 are reported.

Ground-truth labels
-------------------
The images (e.g. 100.JPEG, 101.JPEG, ...) carry no label in their file name,
so a label file is required (--labels). Two formats are accepted:

    CSV with header      filename,label
                         100.JPEG,65
    whitespace-separated 100.JPEG 65

`label` is the ImageNet-1k class index (0-999, standard torchvision order).


uv run python modelsInference.py \
  --data_dir data/imagenet_val_1000 \
  --labels   data/imagenet_val_1000/labels.csv \
  --out_dir  results/clean_inferences



How to activate uv environment 

cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate


"""

import argparse
import csv
import os
import re
import time

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.models import (
    ConvNeXt_Base_Weights,
    ResNet50_Weights,
    ViT_B_16_Weights,
    convnext_base,
    resnet50,
    vit_b_16,
)

IMG_EXTENSIONS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")

# Priority order: ConvNeXt-Base -> ViT-B/16 -> ResNet-50
MODELS = [
    ("ConvNeXt-Base", convnext_base, ConvNeXt_Base_Weights.IMAGENET1K_V1),
    ("ViT-B/16", vit_b_16, ViT_B_16_Weights.IMAGENET1K_V1),
    ("ResNet-50", resnet50, ResNet50_Weights.IMAGENET1K_V2),
]


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #
def numeric_key(filename):
    """Sort 2.JPEG before 10.JPEG."""
    m = re.match(r"(\d+)", filename)
    return (int(m.group(1)) if m else float("inf"), filename)


def list_images(data_dir):
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(IMG_EXTENSIONS)]
    return sorted(files, key=numeric_key)


def load_labels(label_path):
    """Return {filename: class_index}."""
    labels = {}
    with open(label_path, newline="") as fh:
        first = fh.readline()
        fh.seek(0)
        if "," in first:
            reader = csv.reader(fh)
            for row in reader:
                if not row or row[0].strip().lower() in ("filename", "file", "image"):
                    continue
                labels[row[0].strip()] = int(row[1])
        else:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    labels[parts[0]] = int(parts[1])
    return labels


def load_image(path, preprocess, device):
    img = Image.open(path).convert("RGB")  # handles grayscale / CMYK JPEGs
    return preprocess(img).unsqueeze(0).to(device)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def compute_metrics(y_true, y_pred):
    """
    Accuracy plus macro- and weighted-averaged precision / recall / F1.

    Averaging is done over the classes that appear in the ground truth, so a
    small sample is not penalised for the hundreds of ImageNet classes it
    does not contain.
    """
    present = sorted(set(y_true))
    metrics = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ("macro", "weighted"):
        kw = dict(labels=present, average=avg, zero_division=0)
        metrics[f"precision_{avg}"] = precision_score(y_true, y_pred, **kw)
        metrics[f"recall_{avg}"] = recall_score(y_true, y_pred, **kw)
        metrics[f"f1_{avg}"] = f1_score(y_true, y_pred, **kw)
    return metrics


@torch.no_grad()
def run_model(name, builder, weights, image_files, labels, data_dir, device, out_dir):
    print(f"\n{'=' * 70}\nModel: {name}  ({weights})\n{'=' * 70}")

    model = builder(weights=weights).eval().to(device)
    preprocess = weights.transforms()          # model-specific resize/crop/normalise
    categories = weights.meta["categories"]    # ImageNet class names

    y_true, y_pred, rows = [], [], []
    start = time.time()

    for i, fname in enumerate(image_files, 1):
        x = load_image(os.path.join(data_dir, fname), preprocess, device)
        print("x.shape", x.shape)
        probs = model(x).softmax(dim=1)[0]
        conf, pred = probs.max(dim=0)
        pred, conf = pred.item(), conf.item()
        true = labels[fname]

        y_true.append(true)
        y_pred.append(pred)
        rows.append([fname, true, categories[true], pred, categories[pred],
                     f"{conf:.4f}", int(pred == true)])

        if i % 50 == 0 or i == len(image_files):
            running_acc = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)
            print(f"  [{i:4d}/{len(image_files)}] running acc = {running_acc:.4f}")

    elapsed = time.time() - start
    metrics = compute_metrics(y_true, y_pred)
    metrics["n_images"] = len(y_true)
    metrics["seconds"] = elapsed

    print(f"\n  Images            : {len(y_true)}  ({elapsed:.1f}s)")
    print(f"  Accuracy (top-1)  : {metrics['accuracy']:.4f}")
    print(f"  Precision  macro  : {metrics['precision_macro']:.4f}   weighted: {metrics['precision_weighted']:.4f}")
    print(f"  Recall     macro  : {metrics['recall_macro']:.4f}   weighted: {metrics['recall_weighted']:.4f}")
    print(f"  F1         macro  : {metrics['f1_macro']:.4f}   weighted: {metrics['f1_weighted']:.4f}")

    # Per-image predictions
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    pred_path = os.path.join(out_dir, f"predictions_{safe}.csv")
    with open(pred_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["filename", "true_idx", "true_name", "pred_idx", "pred_name",
                    "confidence", "correct"])
        w.writerows(rows)
    print(f"  Per-image predictions -> {pred_path}")

    # Free GPU memory before the next model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str,
                        default="/data1/chethan/interpretAttacks/gemma_attack/dataSamplesForQuant")
    parser.add_argument("--labels", type=str, required=True,
                        help="CSV or whitespace file mapping filename -> ImageNet class index")
    parser.add_argument("--out_dir", type=str, default="results/clean_inference")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    image_files = list_images(args.data_dir)
    labels = load_labels(args.labels)

    missing = [f for f in image_files if f not in labels]
    if missing:
        print(f"WARNING: {len(missing)} images have no label and are skipped "
              f"(e.g. {missing[:5]})")
        image_files = [f for f in image_files if f in labels]
    if not image_files:
        raise SystemExit("No labelled images found - check --data_dir and --labels.")

    print(f"Device: {device} | images: {len(image_files)} | "
          f"distinct classes: {len(set(labels[f] for f in image_files))}")

    summary = {}
    for name, builder, weights in MODELS:
        summary[name] = run_model(name, builder, weights, image_files, labels,
                                  args.data_dir, device, args.out_dir)

    # Comparison table
    cols = ["accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted"]
    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    print(f"{'Model':<15}" + "".join(f"{c.replace('_', ' '):>14}" for c in cols[:4]))
    for name, m in summary.items():
        print(f"{name:<15}" + "".join(f"{m[c]:>14.4f}" for c in cols[:4]))

    summary_path = os.path.join(args.out_dir, "summary_metrics.csv")
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model"] + cols + ["n_images", "seconds"])
        for name, m in summary.items():
            w.writerow([name] + [f"{m[c]:.6f}" for c in cols]
                       + [m["n_images"], f"{m['seconds']:.1f}"])
    print(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()
