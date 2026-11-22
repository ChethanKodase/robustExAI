"""
Spectrum check (diagnostic A): does ConvNeXt-Base / ViT-B/16 / ResNet-50 have the
near-null singular structure that the spectral-subspace attack idea needs?

Motivation: the H1 probe (spectralProbeIG.py) found NO spectral-subspace signal on
ConvNeXt-Base, and spectral_svals.csv showed ConvNeXt's probed layers have NO singular
values below 1e-2 (sigma_min ~ 0.22 on the row space). The SSGRA paper's VLMs, by
contrast, had 25-27% of operators with sigma_min < 1e-3. This script measures, for the
three torchvision classifiers we use, the distribution of extreme singular values across
all intermediate linear operators, mirroring Table 1 of the SSGRA paper, to decide
whether the spectral angle is dead for these models or just worth trying on ViT/ResNet.

No data, no attacks: it only loads the pretrained weights and does SVDs. Needs network
access the first time to download the weights (torchvision hub).

What is an "operator" here:
    * nn.Linear                       -> W = weight                [out, in]
    * nn.Conv2d                       -> W = weight.reshape(out,-1) [out, in*kh*kw]
                                         (the standard "filter matrix" SVD used in
                                          spectral-norm / Lipschitz work; this is an
                                          approximation of the true conv operator)
    * nn.MultiheadAttention           -> in_proj_weight [3*dim, dim]  (qkv packed)
                                         (out_proj is a Linear, caught separately)
sigma_min is the SMALLEST of the min(m,n) singular values (the row-space minimum; the
geometric null space from a rectangular shape is excluded, it is counted separately as
exact_null = |n - m|). A near-null direction = a small but non-zero sigma.

Outputs (folder created at runtime; --out_dir, default spectralSpectrumResults/):
    per_operator_spectrum.csv   one row per operator: model, name, kind, m, n, n_sv,
                                exact_null, sigma_max, sigma_min, sigma_mean,
                                n_below_{1e-2,1e-3,1e-4,1e-5,1e-6}, n_above_{10,100}
    summary_table.csv           SSGRA-style Table 1: per model x (all ops / linear-only),
                                counts and % of operators past each threshold
    figures/sigma_min_hist.{pdf,png}, sigma_max_hist.{pdf,png}
    config.txt

Usage
-----
    uv run python spectralSpectrumCheck.py --models convnext_base vit_b_16 resnet50
    (CPU is fine; SVDs of a few thousand small matrices take under a minute per model.)


export CUDA_VISIBLE_DEVICES=2
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate
uv run python spectralSpectrumCheck.py --models convnext_base vit_b_16 resnet50

"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn

MODEL_WEIGHTS = {
    "convnext_base": ("convnext_base", "ConvNeXt_Base_Weights"),
    "vit_b_16": ("vit_b_16", "ViT_B_16_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "vit_l_16": ("vit_l_16", "ViT_L_16_Weights"),
    "convnext_large": ("convnext_large", "ConvNeXt_Large_Weights"),
}
SIGMA_MIN_THRESH = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
SIGMA_MAX_THRESH = [10.0, 100.0]


def load_pretrained(model_name, weights=True):
    import torchvision.models as M
    ctor_name, weights_name = MODEL_WEIGHTS[model_name]
    ctor = getattr(M, ctor_name)
    if weights:
        w = getattr(M, weights_name).IMAGENET1K_V1
        return ctor(weights=w).eval()
    return ctor(weights=None).eval()


def iter_operators(model):
    """
    Yield (name, kind, W) for every intermediate linear operator.
    kind in {"linear", "conv", "attn_in_proj"}.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            yield name, "linear", module.weight.detach()
        elif isinstance(module, nn.Conv2d):
            w = module.weight.detach()
            yield name, "conv", w.reshape(w.shape[0], -1)
        elif isinstance(module, nn.MultiheadAttention):
            if getattr(module, "in_proj_weight", None) is not None:
                yield name + ".in_proj", "attn_in_proj", module.in_proj_weight.detach()


@torch.no_grad()
def operator_spectrum(W, device):
    W = W.to(torch.float32).to(device)
    m, n = W.shape
    s = torch.linalg.svdvals(W)  # min(m,n) values, descending
    s = s.cpu().numpy()
    row = {
        "m": m, "n": n, "n_sv": len(s), "exact_null": abs(n - m),
        "sigma_max": float(s[0]), "sigma_min": float(s[-1]), "sigma_mean": float(s.mean()),
    }
    for t in SIGMA_MIN_THRESH:
        row[f"n_below_{t:.0e}"] = int((s < t).sum())
    for t in SIGMA_MAX_THRESH:
        row[f"n_above_{t:g}"] = int((s > t).sum())
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["convnext_base", "vit_b_16", "resnet50"],
                   choices=list(MODEL_WEIGHTS))
    p.add_argument("--out_dir", default="spectralSpectrumResults")
    p.add_argument("--kinds", nargs="+", default=["linear", "conv", "attn_in_proj"],
                   help="operator kinds to include")
    p.add_argument("--random_weights", action="store_true",
                   help="use random (untrained) weights - for a code check only")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    fig_dir = os.path.join(args.out_dir, "figures"); os.makedirs(fig_dir, exist_ok=True)

    all_rows, per_model_sigma = [], {}
    for model_name in args.models:
        print(f"\n=== {model_name} ===")
        model = load_pretrained(model_name, weights=not args.random_weights)
        rows = []
        for name, kind, W in iter_operators(model):
            if kind not in args.kinds:
                continue
            rec = {"model": model_name, "name": name, "kind": kind}
            rec.update(operator_spectrum(W, device))
            rows.append(rec)
        all_rows.extend(rows)
        per_model_sigma[model_name] = {
            "sigma_min": np.array([r["sigma_min"] for r in rows]),
            "sigma_max": np.array([r["sigma_max"] for r in rows]),
            "kinds": [r["kind"] for r in rows],
        }
        n_ops = len(rows)
        n_lin = sum(r["kind"] != "conv" for r in rows)
        print(f"  operators: {n_ops} (linear/attn: {n_lin}, conv: {n_ops - n_lin})")
        for t in [1e-2, 1e-3, 1e-4]:
            c = sum(r["sigma_min"] < t for r in rows)
            print(f"  sigma_min < {t:.0e}: {c}/{n_ops} = {c / max(n_ops,1):5.1%}")
        print(f"  sigma_min overall min={min(r['sigma_min'] for r in rows):.2e}  "
              f"sigma_max overall max={max(r['sigma_max'] for r in rows):.2e}")
        del model

    # ---- per-operator CSV ----
    cols = (["model", "name", "kind", "m", "n", "n_sv", "exact_null",
             "sigma_max", "sigma_min", "sigma_mean"]
            + [f"n_below_{t:.0e}" for t in SIGMA_MIN_THRESH]
            + [f"n_above_{t:g}" for t in SIGMA_MAX_THRESH])
    with open(os.path.join(args.out_dir, "per_operator_spectrum.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(all_rows)

    # ---- SSGRA-style summary table (per model, all ops and linear-only) ----
    summ = []
    for model_name in args.models:
        for scope, pred in [("all", lambda r: True),
                            ("linear_attn_only", lambda r: r["kind"] != "conv")]:
            sub = [r for r in all_rows if r["model"] == model_name and pred(r)]
            if not sub:
                continue
            N = len(sub)
            rec = {"model": model_name, "scope": scope, "n_operators": N}
            for t in SIGMA_MIN_THRESH:
                c = sum(r["sigma_min"] < t for r in sub)
                rec[f"sigma_min<{t:.0e}"] = f"{c} ({c / N:.1%})"
            for t in SIGMA_MAX_THRESH:
                c = sum(r["sigma_max"] > t for r in sub)
                rec[f"sigma_max>{t:g}"] = f"{c} ({c / N:.1%})"
            summ.append(rec)
    with open(os.path.join(args.out_dir, "summary_table.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    # ---- console table ----
    print(f"\n{'=' * 78}\nSSGRA-style summary (near-null = small sigma_min on the row space)\n{'=' * 78}")
    print(f"{'model':16s}{'scope':18s}{'#ops':>6}{'<1e-2':>10}{'<1e-3':>10}{'<1e-4':>10}")
    for rec in summ:
        print(f"{rec['model']:16s}{rec['scope']:18s}{rec['n_operators']:>6}"
              f"{rec['sigma_min<1e-02']:>10}{rec['sigma_min<1e-03']:>10}{rec['sigma_min<1e-04']:>10}")

    try:
        make_figures(per_model_sigma, args.models, fig_dir)
    except Exception as e:
        print("(figure step skipped:", e, ")")

    with open(os.path.join(args.out_dir, "config.txt"), "w") as fh:
        fh.write(str(vars(args)) + "\n")
    print(f"\nResults -> {args.out_dir}")
    print("Read: if ViT/ResNet show many sigma_min < 1e-3 (like the VLMs), the spectral "
          "attack is worth trying there; if all are like ConvNeXt (none), the subspace "
          "angle is dead for these models and the backward probe / depth-profile is the way.")


def make_figures(per_model_sigma, models, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.size": 8, "axes.spines.top": False,
                                "axes.spines.right": False, "pdf.fonttype": 42})
    colors = {"convnext_base": "#0072B2", "vit_b_16": "#D55E00", "resnet50": "#009E73",
              "vit_l_16": "#CC79A7", "convnext_large": "#56B4E9"}

    for which, xlabel in [("sigma_min", r"$\sigma_{\min}$ (row space)"),
                          ("sigma_max", r"$\sigma_{\max}$")]:
        fig, ax = plt.subplots(figsize=(3.5, 2.6))
        for mname in models:
            v = per_model_sigma[mname][which]
            v = v[v > 0]
            ax.hist(np.log10(v), bins=30, histtype="step", lw=1.2,
                    color=colors.get(mname), label=mname)
        ax.set_xlabel(r"$\log_{10}$ " + xlabel)
        ax.set_ylabel("# operators")
        if which == "sigma_min":
            for t in [-2, -3]:
                ax.axvline(t, color="k", lw=0.5, ls=":")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{which}_hist.pdf"))
        fig.savefig(os.path.join(fig_dir, f"{which}_hist.png"), dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
