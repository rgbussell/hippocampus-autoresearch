#!/usr/bin/env python3
"""The keep rule (INS-2), applied to one experiment against the arm's incumbent.

    decide.py --result R1.json R2.json --band BAND.json --arm agent   -> kept|reverted (two-seed mean, v3)
    decide.py --result R.json --band BAND.json --arm agent            -> kept|confirm|reverted (single seed, v2)
    decide.py --result R.json --confirm R2.json --band ... --arm ...  -> kept|reverted (v2 confirm)
    decide.py --promote R1.json [R2.json] --arm agent                 -> writes manifests/incumbent_<arm>.json

The incumbent file holds the best KEPT score for the arm (a two-seed mean in
v3). If it does not exist, the incumbent is the baseline band mean.
Threshold = band["keep_threshold_dice"].
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import keep_decision, two_seed_decision  # noqa: E402


def incumbent_path(arm: str) -> Path:
    return ROOT / "manifests" / f"incumbent_{arm}.json"


def incumbent_score(arm: str, band: dict) -> float:
    p = incumbent_path(arm)
    if p.exists():
        return float(json.loads(p.read_text())["val_dice_mean"])
    return float(band["mean"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path, nargs="+")
    ap.add_argument("--confirm", type=Path)
    ap.add_argument("--band", type=Path, default=ROOT / "manifests" / "w2_baseline_seed_band.json")
    ap.add_argument("--arm", default="agent")
    ap.add_argument("--promote", type=Path, nargs="+")
    args = ap.parse_args()
    if args.promote:
        rs = [json.loads(p.read_text()) for p in args.promote]
        incumbent_path(args.arm).write_text(
            json.dumps(
                {
                    "arm": args.arm,
                    "val_dice_mean": float(statistics.mean(r["val_dice_mean"] for r in rs)),
                    "seeds": [r["seed"] for r in rs],
                    "runs": [
                        {k: r[k] for k in ("seed", "val_dice_mean", "commit", "mlflow_run_id", "steps")} for r in rs
                    ],
                    "commit": rs[0]["commit"],
                    "mlflow_run_id": rs[0]["mlflow_run_id"],
                    "description": rs[0]["description"],
                    "recipe": rs[0]["recipe"],
                },
                indent=1,
            )
            + "\n"
        )
        return 0
    band = json.loads(args.band.read_text())
    thr = float(band["keep_threshold_dice"])
    inc = incumbent_score(args.arm, band)
    rs = [json.loads(p.read_text()) for p in args.result]
    if len(rs) >= 2:
        d = two_seed_decision(float(statistics.mean(r["val_dice_mean"] for r in rs)), inc, thr)
    else:
        score = float(rs[0]["val_dice_mean"])
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
