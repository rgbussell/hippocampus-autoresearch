#!/usr/bin/env python3
"""Frozen ruler: per-case Dice for anterior (1) and posterior (2) hippocampus.

Library:
    dice_per_case(pred, ref) -> {"dice_ant", "dice_post", "dice_mean"}
    score_split(pred_dir, split, partition_path, data_root) -> (rows, summary)

CLI:
    python eval/score.py --pred-dir runs/<id>/pred --split val
    python eval/score.py --pred-dir ... --split holdout   # refused unless
        HIPPO_HOLDOUT_OK=1 is set by the person scoring it, once, at the close.

Dice is computed on the full volume at native resolution. An empty
prediction against a non-empty reference scores 0.0 (never NaN). If BOTH are
empty for a label the label's Dice is 1.0 (nothing to find, nothing found).
No cropping, no post-processing, no label remapping here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.partition import load_partition  # noqa: E402

LABELS = {"ant": 1, "post": 2}


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.count_nonzero(a & b)
    denom = np.count_nonzero(a) + np.count_nonzero(b)
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def dice_per_case(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch pred {pred.shape} vs ref {ref.shape}")
    pred = np.asarray(pred).astype(np.int16)
    ref = np.asarray(ref).astype(np.int16)
    out = {}
    for name, lab in LABELS.items():
        out[f"dice_{name}"] = _dice(pred == lab, ref == lab)
    out["dice_mean"] = float((out["dice_ant"] + out["dice_post"]) / 2.0)
    return out


def load_label(path: Path) -> np.ndarray:
    arr = np.asanyarray(nib.load(str(path)).dataobj)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    return np.rint(arr).astype(np.int16)


def score_split(pred_dir: Path, split: str, partition_path: Path, data_root: Path):
    if split == "holdout" and os.environ.get("HIPPO_HOLDOUT_OK") != "1":
        raise PermissionError(
            "hold-out scoring is refused inside the loop; a person sets HIPPO_HOLDOUT_OK=1 once at the close")
    part = load_partition(partition_path)
    ids = part.split(split)
    rows = []
    for cid in ids:
        ref = load_label(data_root / "labelsTr" / f"{cid}.nii.gz")
        pp = pred_dir / f"{cid}.nii.gz"
        if not pp.exists():
            raise FileNotFoundError(f"missing prediction {pp}")
        pred = load_label(pp)
        d = dice_per_case(pred, ref)
        d["case"] = cid
        rows.append(d)
    summary = {
        "split": split,
        "n": len(rows),
        "dice_mean": float(np.mean([r["dice_mean"] for r in rows])),
        "dice_ant": float(np.mean([r["dice_ant"] for r in rows])),
        "dice_post": float(np.mean([r["dice_post"] for r in rows])),
        "dice_mean_median": float(np.median([r["dice_mean"] for r in rows])),
    }
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["val", "holdout", "train"], default="val")
    ap.add_argument("--partition", type=Path, default=ROOT / "manifests" / "partition.json")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data" / "Task04_Hippocampus")
    ap.add_argument("--out", type=Path, help="write per-case TSV + summary JSON stem")
    args = ap.parse_args()
    rows, summary = score_split(args.pred_dir, args.split, args.partition, args.data_root)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out.with_suffix(".tsv"), "w") as f:
            f.write("case\tdice_ant\tdice_post\tdice_mean\n")
            for r in rows:
                f.write(f"{r['case']}\t{r['dice_ant']:.6f}\t{r['dice_post']:.6f}\t{r['dice_mean']:.6f}\n")
        args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
