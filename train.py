#!/usr/bin/env python3
"""One experiment: train under a fixed wall-clock budget, predict val, score
with the frozen ruler, log to MLflow, write runs/<id>/result.json, and
(unless --no-append) append one row to results.tsv.

Fixed by design. The agent edits recipe.py, not this file.

    python train.py --seed 1337 --budget 300 --desc "plain baseline" --verdict baseline
    python train.py --seed 1337 --budget 300 --desc "..." --no-append   # the harness records

Refusals (exit 3, logged as verdict=refused by the caller): a RECIPE value
outside KNOBS, inconsistent channels/strides, a network above PARAM_CAP,
or CUDA out of memory. The fence (scripts/fence.py) runs first; if it
fails nothing is trained.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from hippo.ledger import append_row, next_exp  # noqa: E402
from hippo.partition import load_partition  # noqa: E402

DATA = ROOT / "data" / "Task04_Hippocampus"
RESULTS = ROOT / "results.tsv"
EXPERIMENT = "hippocampus-autoresearch"
REFUSED = 3


# ----------------------------------------------------------------------------
# recipe validation
# ----------------------------------------------------------------------------
def validate_recipe(recipe: dict, knobs: dict) -> None:
    for k, v in recipe.items():
        if k not in knobs:
            raise ValueError(f"unknown knob {k!r}")
        spec = knobs[k]
        if "choices" in spec:
            if v not in spec["choices"] and (not isinstance(v, (tuple, list)) or list(v) not in [list(c) if isinstance(c, (tuple, list)) else c for c in spec["choices"]]):
                raise ValueError(f"{k}={v!r} not in {spec['choices']}")
        elif "range" in spec:
            lo, hi = spec["range"]
            if not (lo <= float(v) <= hi):
                raise ValueError(f"{k}={v!r} outside [{lo}, {hi}]")
    if len(recipe["strides"]) != len(recipe["channels"]) - 1:
        raise ValueError("len(strides) must equal len(channels) - 1")


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_case(cid: str):
    img = nib.load(str(DATA / "imagesTr" / f"{cid}.nii.gz"))
    lab = nib.load(str(DATA / "labelsTr" / f"{cid}.nii.gz"))
    x = np.asanyarray(img.dataobj).astype(np.float32)
    y = np.asanyarray(lab.dataobj)
    if x.ndim == 4:
        x = x[..., 0]
    if y.ndim == 4:
        y = y[..., 0]
    return x, np.rint(y).astype(np.int64), img.affine


def normalize(x: np.ndarray, how: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if how == "none":
        out = x
    elif how == "minmax":
        lo, hi = float(x.min()), float(x.max())
        out = (x - lo) / (hi - lo + 1e-8)
    elif how == "zscore":
        out = (x - x.mean()) / (x.std() + 1e-8)
    elif how == "percentile_clip":
        lo, hi = np.percentile(x, [0.5, 99.5])
        x = np.clip(x, lo, hi)
        out = (x - x.mean()) / (x.std() + 1e-8)
    else:
        raise ValueError(how)
    return np.asarray(out, dtype=np.float32)  # percentile scalars would otherwise promote to float64


def pad_to(x: np.ndarray, shape, value=0.0) -> np.ndarray:
    pads = [(0, max(0, s - d)) for d, s in zip(x.shape, shape)]
    return np.pad(x, pads, constant_values=value) if any(p[1] for p in pads) else x


class PatchSampler:
    def __init__(self, cases, patch, fg_oversample, rng):
        self.cases = cases  # list of (x, y) padded to >= patch
        self.patch = tuple(patch)
        self.fg = float(fg_oversample)
        self.rng = rng
        self.fg_idx = [np.argwhere(y > 0) for _, y in cases]

    def sample(self):
        i = self.rng.integers(len(self.cases))
        x, y = self.cases[i]
        p = self.patch
        if self.rng.random() < self.fg and len(self.fg_idx[i]):
            c = self.fg_idx[i][self.rng.integers(len(self.fg_idx[i]))]
            starts = [int(np.clip(ci - pi // 2, 0, si - pi)) for ci, pi, si in zip(c, p, x.shape)]
        else:
            starts = [int(self.rng.integers(0, si - pi + 1)) for pi, si in zip(p, x.shape)]
        sl = tuple(slice(s, s + pi) for s, pi in zip(starts, p))
        return x[sl], y[sl]


def build_aug(recipe: dict, seed: int):
    from monai.transforms import (Compose, RandAffined, RandFlipd, RandGaussianNoised,
                                  RandRotate90d, RandScaleIntensityd)
    t = []
    if recipe["aug_flip"]:
        for ax in (0, 1, 2):
            t.append(RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=ax))
    if recipe["aug_rot90"]:
        t.append(RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3))
    if recipe["aug_affine"]:
        t.append(RandAffined(keys=["image", "label"], prob=0.3, rotate_range=(0.26, 0.26, 0.26),
                             scale_range=(0.15, 0.15, 0.15), mode=("bilinear", "nearest"), padding_mode="border"))
    if recipe["aug_scale_intensity"] > 0:
        t.append(RandScaleIntensityd(keys=["image"], factors=recipe["aug_scale_intensity"], prob=0.5))
    if recipe["aug_gaussian_noise"] > 0:
        t.append(RandGaussianNoised(keys=["image"], prob=0.3, std=recipe["aug_gaussian_noise"]))
    if not t:
        return None
    c = Compose(t)
    c.set_random_state(seed=seed)
    return c


# ----------------------------------------------------------------------------
# model, loss, optimizer
# ----------------------------------------------------------------------------
def build_model(recipe: dict):
    import torch
    from monai.networks.nets import DynUNet
    ch = list(recipe["channels"])
    st = [1] + list(recipe["strides"])
    n = len(ch)
    ds = bool(recipe["deep_supervision"])
    model = DynUNet(
        spatial_dims=3, in_channels=1, out_channels=3,
        kernel_size=[3] * n, strides=st, upsample_kernel_size=st[1:], filters=ch,
        norm_name=("instance" if recipe["norm"] == "instance" else "batch"),
        deep_supervision=ds, deep_supr_num=max(1, min(2, n - 2)) if ds else 1,
        res_block=recipe["num_res_units"] > 0, dropout=recipe["dropout"] or None,
    )
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params


def build_loss(recipe: dict, device):
    import torch
    from monai.losses import DiceCELoss, DiceLoss
    w = recipe["class_weights"]
    wt = torch.tensor(w, dtype=torch.float32, device=device) if w else None
    if recipe["loss"] == "ce":
        return torch.nn.CrossEntropyLoss(weight=wt)          # target (B, H, W, D)
    if recipe["loss"] == "dice":
        inner = DiceLoss(to_onehot_y=True, softmax=True)      # target (B, 1, H, W, D)
    elif recipe["loss"] == "dice_ce":
        inner = DiceCELoss(to_onehot_y=True, softmax=True, weight=wt)
    else:
        raise ValueError(recipe["loss"])
    return lambda logits, y: inner(logits, y[:, None])


def build_optimizer(recipe: dict, params):
    import torch
    if recipe["optimizer"] == "adam":
        return torch.optim.Adam(params, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    if recipe["optimizer"] == "adamw":
        return torch.optim.AdamW(params, lr=recipe["lr"], weight_decay=recipe["weight_decay"])
    if recipe["optimizer"] == "sgd_nesterov":
        return torch.optim.SGD(params, lr=recipe["lr"], momentum=recipe["momentum"], nesterov=True,
                               weight_decay=recipe["weight_decay"])
    raise ValueError(recipe["optimizer"])


def lr_at(recipe: dict, frac: float) -> float:
    base = recipe["lr"]
    if recipe["schedule"] == "flat":
        return base
    if recipe["schedule"] == "poly":
        return base * (1 - frac) ** 0.9
    if recipe["schedule"] == "cosine":
        return base * 0.5 * (1 + math.cos(math.pi * frac))
    raise ValueError(recipe["schedule"])


# ----------------------------------------------------------------------------
# the experiment
# ----------------------------------------------------------------------------
def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "nogit"


def run_fence() -> None:
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fence.py")], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:], file=sys.stderr)
        raise SystemExit("fence failed; nothing trained")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--budget", type=float, default=300.0, help="training seconds after warm-up")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--desc", default="", help="one-line description of the change")
    ap.add_argument("--verdict", default="scored", help="baseline | smoke | scored (harness overrides)")
    ap.add_argument("--no-append", action="store_true", help="write result.json only; the harness records")
    ap.add_argument("--run-dir", type=Path, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--arm", default="agent", help="baseline | agent | random | human")
    ap.add_argument("--recipe-json", type=Path, default=None,
                    help="use this recipe (JSON of RECIPE) instead of recipe.py; the random climber's path")
    args = ap.parse_args()

    run_fence()
    recipe_mod = importlib.import_module("recipe")
    recipe, knobs, cap = dict(recipe_mod.RECIPE), recipe_mod.KNOBS, recipe_mod.PARAM_CAP
    if args.recipe_json:
        loaded = json.loads(args.recipe_json.read_text())
        recipe = {k: (tuple(v) if isinstance(v, list) and k in ("patch", "channels", "strides") else v)
                  for k, v in loaded.items()}
    try:
        validate_recipe(recipe, knobs)
    except ValueError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return REFUSED

    import torch
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model, n_params = build_model(recipe)
    if n_params > cap:
        print(f"REFUSED: {n_params} params > PARAM_CAP {cap}", file=sys.stderr)
        return REFUSED
    model = model.to(device)

    part = load_partition(ROOT / "manifests" / "partition.json")  # raises on a leak
    patch = tuple(recipe["patch"])
    train_cases = []
    for cid in part.train:
        x, y, _ = load_case(cid)
        x = normalize(x, recipe["normalization"])
        train_cases.append((pad_to(x, patch), pad_to(y, patch, 0)))
    sampler = PatchSampler(train_cases, patch, recipe["fg_oversample"], rng)
    aug = build_aug(recipe, args.seed)
    loss_fn = build_loss(recipe, device)
    opt = build_optimizer(recipe, model.parameters())
    bs = int(recipe["batch_size"])
    commit = git_head()
    scorer_sha = json.loads((ROOT / "manifests" / "frozen.json").read_text())["sha256"]

    # MLflow: repo-local file store unless MLFLOW_TRACKING_URI says otherwise
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}"))
    mlflow.set_experiment(EXPERIMENT)
    run_name = f"{(args.desc or 'exp')[:40]}-s{args.seed}"
    with mlflow.start_run(run_name=run_name) as mlrun:
        run_id = mlrun.info.run_id
        run_dir = args.run_dir or (ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-s{args.seed}-{run_id[:8]}")
        run_dir = run_dir if run_dir.is_absolute() else (Path.cwd() / run_dir)
        run_dir = run_dir.resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        mlflow.log_params({k: (json.dumps(v) if isinstance(v, (list, tuple, dict)) or v is None else v)
                           for k, v in recipe.items()})
        mlflow.log_params({"seed": args.seed, "budget_s": args.budget, "device": device.type,
                           "n_params": n_params, "n_train": len(part.train), "n_val": len(part.val)})
        mlflow.set_tags({"commit": commit, "scorer_sha256": scorer_sha, "desc": args.desc,
                         "verdict": args.verdict, "arm": args.arm})
        mlflow.log_artifact(str(args.recipe_json if args.recipe_json else ROOT / "recipe.py"))

        def batch():
            xs, ys = [], []
            for _ in range(bs):
                xp, yp = sampler.sample()
                if aug is not None:
                    d = aug({"image": torch.from_numpy(xp)[None], "label": torch.from_numpy(yp)[None]})
                    xs.append(d["image"].float()); ys.append(d["label"].long()[0])
                else:
                    xs.append(torch.from_numpy(xp)[None]); ys.append(torch.from_numpy(yp))
            return torch.stack(xs).float().to(device), torch.stack(ys).long().to(device)

        def step(frac: float) -> float:
            for g in opt.param_groups:
                g["lr"] = lr_at(recipe, frac)
            x, y = batch()
            opt.zero_grad(set_to_none=True)
            out = model(x)
            if out.ndim == 6:  # deep supervision: (B, heads, C, ...)
                weights = [1.0] + [0.5 ** i for i in range(1, out.shape[1])]
                loss = sum(w * loss_fn(out[:, i], y) for i, w in enumerate(weights)) / sum(weights)
            else:
                loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            return float(loss.detach())

        model.train()
        try:
            step(0.0)  # warm-up step (kernels, allocator); excluded from the budget
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            steps, last_loss, losses = 0, float("nan"), []
            while True:
                elapsed = time.time() - t0
                if elapsed >= args.budget:
                    break
                last_loss = step(min(elapsed / args.budget, 1.0))
                steps += 1
                losses.append(last_loss)
                if steps % args.log_every == 0:
                    mlflow.log_metric("train_loss", float(np.mean(losses[-args.log_every:])), step=steps)
                    mlflow.log_metric("lr", lr_at(recipe, elapsed / args.budget), step=steps)
            train_seconds = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            print("REFUSED: CUDA out of memory", file=sys.stderr)
            mlflow.set_tag("verdict", "refused")
            return REFUSED

        # inference on val, then the frozen ruler
        from monai.inferers import sliding_window_inference
        model.eval()
        pred_dir = run_dir / "pred"
        pred_dir.mkdir(exist_ok=True)
        with torch.no_grad():
            for cid in part.val:
                x, _, affine = load_case(cid)
                shape = x.shape
                xn = pad_to(normalize(x, recipe["normalization"]), patch)
                xt = torch.from_numpy(xn)[None, None].to(device)
                logits = sliding_window_inference(xt, roi_size=patch, sw_batch_size=4, predictor=model,
                                                  overlap=0.5, mode="gaussian")
                pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
                pred = pred[: shape[0], : shape[1], : shape[2]]
                nib.save(nib.Nifti1Image(pred, affine), str(pred_dir / f"{cid}.nii.gz"))
        from eval.score import score_split  # the frozen ruler
        rows, summary = score_split(pred_dir, "val", ROOT / "manifests" / "partition.json", DATA)
        with open(run_dir / "val_scores.tsv", "w") as f:
            f.write("case\tdice_ant\tdice_post\tdice_mean\n")
            for r in rows:
                f.write(f"{r['case']}\t{r['dice_ant']:.6f}\t{r['dice_post']:.6f}\t{r['dice_mean']:.6f}\n")
        mlflow.log_metrics({"val_dice_mean": summary["dice_mean"], "val_dice_ant": summary["dice_ant"],
                            "val_dice_post": summary["dice_post"], "steps": steps,
                            "train_seconds": train_seconds, "final_train_loss": last_loss})
        result = {
            "exp": None, "arm": args.arm, "commit": commit, "seed": args.seed, "val_dice_mean": summary["dice_mean"],
            "val_dice_ant": summary["dice_ant"], "val_dice_post": summary["dice_post"],
            "train_seconds": round(train_seconds, 1), "steps": steps, "description": args.desc,
            "verdict": args.verdict, "mlflow_run_id": run_id, "n_params": n_params,
            "scorer_sha256": scorer_sha, "recipe": {k: (list(v) if isinstance(v, tuple) else v) for k, v in recipe.items()},
            "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
        }
        (run_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        mlflow.log_artifact(str(run_dir / "result.json"))
        mlflow.log_artifact(str(run_dir / "val_scores.tsv"))
        for f in pred_dir.glob("*.nii.gz"):  # predictions are not kept; the scores are
            f.unlink()
        pred_dir.rmdir()
        if not args.no_append:
            result["exp"] = next_exp(RESULTS)
            append_row(RESULTS, result)
            (run_dir / "result.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({k: result[k] for k in ("exp", "arm", "commit", "seed", "val_dice_mean", "val_dice_ant",
                                                   "val_dice_post", "train_seconds", "steps", "verdict",
                                                   "mlflow_run_id", "n_params")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
