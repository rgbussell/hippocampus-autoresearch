#!/usr/bin/env python3
"""Summarise the W3 dry run from results.tsv into manifests/w3_dryrun.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import read_rows  # noqa: E402

PRIMARY = {"kept", "reverted", "refused"}  # a confirm row is the second seed of one experiment


def main() -> int:
    rows = [r for r in read_rows(ROOT / "results.tsv") if r["arm"] in ("agent", "random")]
    out = {
        "agent_experiments": sum(1 for r in rows if r["arm"] == "agent" and r["verdict"] in PRIMARY),
        "random_experiments": sum(1 for r in rows if r["arm"] == "random" and r["verdict"] in PRIMARY),
        "verdicts": {
            arm: {
                v: sum(1 for r in rows if r["arm"] == arm and r["verdict"] == v)
                for v in sorted({r["verdict"] for r in rows})
            }
            for arm in ("agent", "random")
        },
        "rows": [
            {k: r[k] for k in ("exp", "arm", "commit", "seed", "val_dice_mean", "steps", "description", "verdict")}
            for r in rows
        ],
    }
    (ROOT / "manifests" / "w3_dryrun.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("agent_experiments", "random_experiments", "verdicts")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
