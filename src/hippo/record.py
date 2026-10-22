"""Append one results.tsv row from a result.json (the single writer's CLI).

python -m hippo.record --result runs/x/result.json --verdict kept --arm agent
python -m hippo.record --refused --arm agent --desc "..." --seed 1337
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from hippo.ledger import append_row, next_exp

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results.tsv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", type=Path)
    ap.add_argument("--verdict", default="scored")
    ap.add_argument("--arm", default="agent")
    ap.add_argument("--refused", action="store_true")
    ap.add_argument("--desc", default="")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.refused:
        commit = (
            subprocess.run(
                ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
            ).stdout.strip()
            or "nogit"
        )
        row = {
            "exp": next_exp(RESULTS),
            "arm": args.arm,
            "commit": commit,
            "seed": args.seed,
            "val_dice_mean": "",
            "val_dice_ant": "",
            "val_dice_post": "",
            "train_seconds": "",
            "steps": "",
            "description": args.desc,
            "verdict": "refused",
            "mlflow_run_id": "",
        }
    else:
        r = json.loads(args.result.read_text())
        row = dict(r, exp=next_exp(RESULTS), arm=args.arm, verdict=args.verdict)
        r.update(exp=row["exp"], arm=args.arm, verdict=args.verdict)
        args.result.write_text(json.dumps(r, indent=1) + "\n")
    n = append_row(RESULTS, row)
    print(f"results.tsv row {n}: {row['verdict']} {row['description'][:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
