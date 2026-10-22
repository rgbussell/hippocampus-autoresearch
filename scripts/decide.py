#!/usr/bin/env python3
"""The keep rule (INS-2), applied to one result against the arm's incumbent.

    decide.py --result R.json --band BAND.json --arm agent           -> kept|confirm|reverted
    decide.py --result R.json --confirm R2.json --band ... --arm ... -> kept|reverted (two-seed mean)
    decide.py --promote R.json --arm agent                           -> writes manifests/incumbent_<arm>.json

The incumbent file holds the best KEPT score for the arm. If it does not
exist, the incumbent is the baseline seed-band mean. Threshold =
band["keep_threshold_dice"].
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import keep_decision  # noqa: E402


def incumbent_path(arm: str) -> Path:
    return ROOT / "manifests" / f"incumbent_{arm}.json"


def incumbent_score(arm: str, band: dict) -> float:
    p = incumbent_path(arm)
    if p.exists():
        return float(json.loads(p.read_text())["val_dice_mean"])
    return float(band["mean"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path)
    ap.add_argument("--confirm", type=Path)
    ap.add_argument("--band", type=Path, default=ROOT / "manifests" / "w2_baseline_seed_band.json")
    ap.add_argument("--arm", default="agent")
    ap.add_argument("--promote", type=Path)
    args = ap.parse_args()
    if args.promote:
        r = json.loads(args.promote.read_text())
        incumbent_path(args.arm).write_text(json.dumps({
            "arm": args.arm, "val_dice_mean": r["val_dice_mean"], "commit": r["commit"],
            "mlflow_run_id": r["mlflow_run_id"], "description": r["description"],
            "recipe": r["recipe"]}, indent=1) + "\n")
        return 0
    band = json.loads(args.band.read_text())
    thr = float(band["keep_threshold_dice"])
    inc = incumbent_score(args.arm, band)
    r = json.loads(args.result.read_text())
    score = float(r["val_dice_mean"])
    if args.confirm:
        r2 = json.loads(args.confirm.read_text())
        score = (score + float(r2["val_dice_mean"])) / 2.0
        d = "kept" if score - inc > thr else "reverted"
    else:
        d = keep_decision(score, inc, thr)
    print(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
