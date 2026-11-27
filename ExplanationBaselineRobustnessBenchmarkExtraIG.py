"""

---------------------------------------------------------------------------------------------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES=0
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate    
uv run python ExplanationBaselineRobustnessBenchmarkExtraIG.py --model resnet50 --baseline max_distance --max_images 100 --eps 0.5 1 2 0.25
uv run python ExplanationBaselineRobustnessBenchmarkExtraIG.py --model resnet50 --baseline max_entropy  --max_images 100 --eps 0.5 1 2 0.25
uv run python ExplanationBaselineRobustnessBenchmarkExtraIG.py --model resnet50 --baseline big          --max_images 100 --eps 0.5 1 2 0.25

export CUDA_VISIBLE_DEVICES=3
cd /data1/chethan/robustExAI
conda deactivate              # if (base) is showing
source .venv/bin/activate    
uv run python ExplanationBaselineRobustnessBenchmarkExtraIG.py --model vit_b_16 --baseline max_distance --max_images 100 --eps 0.5 1 2 0.25
uv run python ExplanationBaselineRobustnessBenchmarkExtraIG.py --model vit_b_16 --baseline max_entropy  --max_images 100 --eps 0.5 1 2 0.25
uv run python ExplanationBaselineRobustnessBenchmarkExtraIG.py --model vit_b_16 --baseline big          --max_images 100 --eps 0.5 1 2 0.25




"""

import argparse
import importlib.util
import os

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "basebench", os.path.join(HERE, "ExplanationBaselineRobustnessBenchmarkIG.py"))
BB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BB)
G = BB.G

NEW_BASELINES = ["max_distance", "max_entropy", "big"]

# globals filled from argv in main(), read by the patched resolve_baselines
_MAXENT = {"img": None}                                  # cached max-entropy baseline (once per process)
_MAXENT_CFG = {"steps": 300, "lr": 0.05}
_BIG_CFG = {"steps": 200, "step": 0.005, "max_eps": 1.0}


# --------------------------------------------------------------------------- #
# Max-Entropy baseline: optimise ONE image for maximum output entropy (most
# "uninformative" input), reused as a constant baseline for every input.
# --------------------------------------------------------------------------- #
def optimize_max_entropy(model, H, W, device, steps, lr):
    b = torch.full((1, 3, H, W), 0.5, device=device, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        p = F.softmax(model.model(model.normalize(b.clamp(0, 1))), dim=1)
        entropy = -(p * (p + 1e-12).log()).sum(1).mean()
        (-entropy).backward()                            # maximise entropy
        opt.step()
        with torch.no_grad():
            b.clamp_(0, 1)
    return b.detach().clamp(0, 1)


def _ensure_maxent(model, x):
    if _MAXENT["img"] is None or _MAXENT["img"].shape[-2:] != x.shape[-2:] \
            or _MAXENT["img"].device != x.device:
        _MAXENT["img"] = optimize_max_entropy(
            model, x.shape[-2], x.shape[-1], x.device, _MAXENT_CFG["steps"], _MAXENT_CFG["lr"])
    return _MAXENT["img"]


# --------------------------------------------------------------------------- #
# BIG: nearest decision-boundary point via untargeted PGD-to-boundary (sign steps,
# stop at the first class flip). Returns the boundary image (detached). This is an
# approximation of the minimal-norm boundary point (DeepFool/CW would be tighter).
# --------------------------------------------------------------------------- #
def find_boundary_baseline(model, x):
    steps, step, max_eps = _BIG_CFG["steps"], _BIG_CFG["step"], _BIG_CFG["max_eps"]
    x = x.detach()
    with torch.no_grad():
        cls = model.model(model.normalize(x)).argmax(1)
    delta = torch.zeros_like(x, requires_grad=True)
    for _ in range(steps):
        with torch.enable_grad():
            logits = model.model(model.normalize((x + delta).clamp(0, 1)))
            loss = F.cross_entropy(logits, cls)
            g = torch.autograd.grad(loss, delta)[0]
        with torch.no_grad():
            delta += step * g.sign()                     # increase the true-class loss -> toward boundary
            delta.clamp_(-max_eps, max_eps)
            delta.copy_((x + delta).clamp(0, 1) - x)
            if (model.model(model.normalize((x + delta).clamp(0, 1))).argmax(1) != cls).all():
                break                                    # crossed the boundary
    return (x + delta).clamp(0, 1).detach()


# --------------------------------------------------------------------------- #
# Patched baseline resolver: handle the three new methods, delegate the rest.
# ig_core / topk_attack / run in BB look up `resolve_baselines` and `build_cache`
# in BB's module globals at call time, so reassigning them below is enough.
# --------------------------------------------------------------------------- #
_orig_resolve = BB.resolve_baselines
_orig_build_cache = BB.build_cache


def resolve_baselines_ext(model, method, x, cache):
    if method == "max_distance":
        return (x < 0.5).to(x.dtype)                     # per-pixel farther of {0,1}
    if method == "max_entropy":
        return _ensure_maxent(model, x)                  # cached constant image [1,3,H,W]
    if method == "big":
        if "baselines" not in cache:                     # compute once (clean x), then fixed
            cache["baselines"] = find_boundary_baseline(model, x)
        return cache["baselines"]
    return _orig_resolve(model, method, x, cache)


def build_cache_ext(method, x, pool, args, gen):
    if method in NEW_BASELINES:
        return {}                                        # big fills its cache lazily in resolve
    return _orig_build_cache(method, x, pool, args, gen)


BB.resolve_baselines = resolve_baselines_ext
BB.build_cache = build_cache_ext


# --------------------------------------------------------------------------- #
# Argparse: mirror the base file (so every setting BB.run reads is present and
# identical) and add the three-method knobs. Then hand off to BB.run / BB.collate.
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collate", action="store_true",
                   help="gather results/igbase_*/summary.csv into the shared ranking (all baselines)")
    p.add_argument("--results_root", default="results")
    p.add_argument("--model", default="resnet50",
                   choices=["convnext_base", "vit_b_16", "resnet50"])
    p.add_argument("--baseline", default="max_distance", choices=NEW_BASELINES)
    p.add_argument("--data_dir", default="data/imagenet_val_1000")
    p.add_argument("--labels", default="data/imagenet_val_1000/labels.csv")
    p.add_argument("--out_dir", default=None,
                   help="default results/igbase_<model>_<baseline>_img<N>")
    p.add_argument("--eps", type=float, nargs="+", default=[0.25, 0.5], help="L_inf in 1/255")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--lr_factor", type=float, default=0.1)
    p.add_argument("--num_summands", type=int, default=30)
    p.add_argument("--pref_expl", type=float, default=100.0)
    p.add_argument("--pref_out", type=float, default=1e6)
    p.add_argument("--topk", type=int, default=1000)
    # unused by the new methods but read by BB.run -> must exist and match defaults
    p.add_argument("--blur_sigma", type=float, default=5.0)
    p.add_argument("--gauss_std", type=float, default=0.3)
    p.add_argument("--gauss_k", type=int, default=1)
    p.add_argument("--eg_k", type=int, default=8)
    p.add_argument("--eg_summands", type=int, default=8)
    p.add_argument("--eg_pool", type=int, default=64)
    # new-method knobs
    p.add_argument("--maxent_steps", type=int, default=300, help="max_entropy optimisation steps")
    p.add_argument("--maxent_lr", type=float, default=0.05, help="max_entropy optimisation lr")
    p.add_argument("--big_steps", type=int, default=200, help="BIG boundary-search max steps")
    p.add_argument("--big_step", type=float, default=0.005, help="BIG boundary-search step (pixel)")
    p.add_argument("--big_max_eps", type=float, default=1.0, help="BIG boundary-search cap in [0,1]")
    # bookkeeping / plotting (mirror base file)
    p.add_argument("--max_images", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_qual", type=int, default=10)
    p.add_argument("--grid_rows", type=int, default=5)
    p.add_argument("--no_plots", action="store_true")
    p.add_argument("--no_beta_growth", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.collate:
        BB.collate(args.results_root)
        return

    _MAXENT_CFG.update(steps=args.maxent_steps, lr=args.maxent_lr)
    _BIG_CFG.update(steps=args.big_steps, step=args.big_step, max_eps=args.big_max_eps)
    _MAXENT["img"] = None                                # fresh per process

    BB.run(args)                                         # identical loop/metrics/plots/folders


if __name__ == "__main__":
    main()
