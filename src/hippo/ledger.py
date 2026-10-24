"""results.tsv: one row per experiment, append-only, single writer.

Columns (see README): exp, arm, commit, seed, val_dice_mean, val_dice_ant,
val_dice_post, train_seconds, steps, description, verdict, mlflow_run_id.
Arms: baseline | agent | random | human.

Verdicts: baseline | kept | reverted | confirm (second seed of a borderline
gain) | refused (outside KNOBS, over PARAM_CAP, or out of memory) | smoke.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

COLUMNS = [
    "exp",
    "arm",
    "commit",
    "seed",
    "val_dice_mean",
    "val_dice_ant",
    "val_dice_post",
    "train_seconds",
    "steps",
    "description",
    "verdict",
    "mlflow_run_id",
]


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def next_exp(path: Path) -> int:
    return len(read_rows(path)) + 1


def append_row(path: Path, row: dict) -> int:
    """Append one row; refuse any tab or newline in the description."""
    desc = str(row.get("description", "")).replace("\t", " ").replace("\n", " ")
    row = dict(row, description=desc)
    missing = [c for c in COLUMNS if c not in row]
    if missing:
        raise ValueError(f"row missing columns {missing}")
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(row)
    return int(row["exp"])


def noise_floor(band_path: Path) -> float:
    """Half-width of the baseline seed band, measured in W2 (INS-2)."""
    d = json.loads(Path(band_path).read_text())
    return float(d["half_width_dice"])


def seed_band(scores: list[float]) -> dict:
    return {
        "n_seeds": len(scores),
        "scores": scores,
        "mean": float(statistics.mean(scores)),
        "min": float(min(scores)),
        "max": float(max(scores)),
        "half_width_dice": float((max(scores) - min(scores)) / 2.0),
        "stdev": float(statistics.stdev(scores)) if len(scores) > 1 else 0.0,
    }


def keep_threshold(half_width: float, same_seed_deltas: list[float], margin: float = 2.0) -> float:
    """The keep threshold of INS-2 (protocol v2, EXP-005/006).

    max(cross-seed half-width, margin x the largest same-seed |delta|). A
    wall-clock budget on a GPU is not deterministic, so the same recipe and
    seed scatter; the margin keeps the gate above that scatter instead of
    exactly at the one value it was measured from. The threshold is then
    checked OUT OF SAMPLE with a fresh same-seed rerun that took no part in
    setting it (seed_band.py).
    """
    worst = max([abs(d) for d in same_seed_deltas], default=0.0)
    return float(max(half_width, margin * worst))


def pairwise_max_delta(scores: list[float]) -> float:
    return float(max((abs(a - b) for i, a in enumerate(scores) for b in scores[i + 1 :]), default=0.0))


def keep_decision(candidate: float, incumbent: float, half_width: float) -> str:
    """kept | confirm | reverted. The gate of INS-2.

    kept     candidate - incumbent >  half_width
    confirm  0 < gain <= 2*half_width and not already kept: run a second seed
    reverted otherwise
    """
    gain = candidate - incumbent
    if gain > half_width:
        return "kept"
    if gain > 0 and gain <= 2 * half_width:
        return "confirm"
    return "reverted"
