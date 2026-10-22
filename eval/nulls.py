#!/usr/bin/env python3
"""INS-1 null suite for the ruler, run on the REAL validation references.

identity  -> mean Dice exactly 1.0
empty     -> mean Dice exactly 0.0 (never NaN)
swapped   -> per-class Dice 0.0 for both labels (the ruler is not label-blind)
shifted   -> a 2-voxel shift scores strictly below identity on every case

Writes manifests/w1_ruler_nulls.json. Run once before the freeze; re-run
after any (logged) change to the ruler.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from eval.score import dice_per_case, load_label  # noqa: E402
from hippo.partition import load_partition  # noqa: E402


def shifted(ref: np.ndarray, k: int = 2) -> np.ndarray:
    out = np.zeros_like(ref)
    out[k:, :, :] = ref[:-k, :, :]
    return out


def main() -> int:
    part = load_partition(ROOT / "manifests" / "partition.json")
    data = ROOT / "data" / "Task04_Hippocampus" / "labelsTr"
    ident, empty, swap_max, shift_below, n = [], [], 0.0, True, 0
    for cid in part.val:
        ref = load_label(data / f"{cid}.nii.gz")
        if not ((ref == 1).any() and (ref == 2).any()):
            continue  # a reference missing a label cannot test swap/empty
        n += 1
        ident.append(dice_per_case(ref, ref)["dice_mean"])
        empty.append(dice_per_case(np.zeros_like(ref), ref)["dice_mean"])
        sw = np.where(ref == 1, 2, np.where(ref == 2, 1, 0)).astype(ref.dtype)
        d = dice_per_case(sw, ref)
        swap_max = max(swap_max, d["dice_ant"], d["dice_post"])
        if not dice_per_case(shifted(ref), ref)["dice_mean"] < 1.0:
            shift_below = False
    out = {
        "n_cases": n,
        "identity_mean_dice": float(min(ident)) if min(ident) == max(ident) else float(np.mean(ident)),
        "empty_mean_dice": float(max(empty)),
        "swapped_labels_per_class_max": float(swap_max),
        "shifted_below_identity": bool(shift_below),
        "shift_voxels": 2,
        "note": "identity/empty report the worst case over val; all four must hold on every case",
    }
    (ROOT / "manifests" / "w1_ruler_nulls.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    ok = (out["identity_mean_dice"] == 1.0 and out["empty_mean_dice"] == 0.0
          and out["swapped_labels_per_class_max"] == 0.0 and out["shifted_below_identity"])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
