#!/usr/bin/env python3
"""INS-2 (protocol v3): the noise floor of TWO-SEED MEANS, the keep threshold,
and an out-of-sample degeneracy check.

    python scripts/seed_band.py --budget 600 --seeds 1337 2027 --calibration-pairs 3 \
        --out manifests/w4_baseline_seed_band.json [--reuse]

Every campaign experiment is scored as the mean over the same two seeds, so
the noise floor is measured on exactly that statistic:

  1. --calibration-pairs runs of the baseline pair (both seeds)  -> pair means
  2. keep_threshold_dice = max(half-width of the pair means,
                               2 x largest pairwise |delta| of the pair means)
  3. one FRESH pair that took no part in step 2: its mean must NOT be kept
     against the calibration mean under the threshold (degeneracy check)

Exit 1 if the fresh pair would have been kept. The budget rule of the W4
pre-registration is reported as gate_ok (threshold <= gate_target).
History: v1 (half-width of three single seeds) failed the check, EXP-005; v2
(single-seed scatter with margin, out-of-sample rerun) passed at 300 s with a
0.011 gate, EXP-006; v3 is the two-seed campaign protocol.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import keep_threshold, pairwise_max_delta, two_seed_decision  # noqa: E402

MARGIN = 2.0
GATE_TARGET = 0.005  # pre-registered: if the gate is wider, re-measure once at the next budget


def run(seed: int, budget: float, device: str, desc: str, tag: str) -> dict:
    run_dir = ROOT / "runs" / f"{tag}-{desc.replace(' ', '-')}-s{seed}"
    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--seed",
        str(seed),
        "--budget",
        str(budget),
        "--device",
        device,
        "--desc",
        desc,
        "--verdict",
        "baseline",
        "--arm",
        "baseline",
        "--run-dir",
        str(run_dir),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"seed {seed} failed with code {r.returncode}")
    return json.loads((run_dir / "result.json").read_text())


def slim(r: dict) -> dict:
    return {k: r[k] for k in ("seed", "val_dice_mean", "steps", "mlflow_run_id", "exp")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 2027])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--calibration-pairs", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "manifests" / "w4_baseline_seed_band.json")
    ap.add_argument("--tag", default="w4")
    ap.add_argument("--reuse", action="store_true", help="reuse finished pairs recorded in the manifest")
    args = ap.parse_args()
    if len(args.seeds) < 2:
        raise SystemExit("protocol v3 needs at least two seeds")

    prev: dict = {}
    if args.reuse and args.out.exists():
        prev = json.loads(args.out.read_text())
        if prev.get("budget_s") != args.budget or prev.get("seeds") != args.seeds:
            raise SystemExit("--reuse refused: budget or seeds differ from the manifest")
    label = f"{args.budget:g}s"

    def pair(k: int, kind: str) -> dict:
        runs = [
            slim(run(s, args.budget, args.device, f"plain baseline {label} {kind} pair {k}", args.tag))
            for s in args.seeds
        ]
        return {"pair": k, "kind": kind, "runs": runs, "mean": float(statistics.mean(r["val_dice_mean"] for r in runs))}

    cal = list(prev.get("calibration_pairs", []))
    for k in range(len(cal) + 1, args.calibration_pairs + 1):
        cal.append(pair(k, "calibration"))
    pair_means = [c["mean"] for c in cal]
    half_width = float((max(pair_means) - min(pair_means)) / 2.0)
    worst = pairwise_max_delta(pair_means)
    threshold = keep_threshold(half_width, [worst], MARGIN)
    mean = float(statistics.mean(pair_means))

    fresh = prev.get("fresh_pair") or pair(args.calibration_pairs + 1, "fresh")
    fresh_delta = fresh["mean"] - mean
    kept = two_seed_decision(fresh["mean"], mean, threshold) == "kept"

    first = ROOT / "runs" / f"{args.tag}-plain-baseline-{label}-calibration-pair-1-s{args.seeds[0]}" / "result.json"
    recipe = prev.get("recipe") or json.loads(first.read_text())["recipe"]
    out = {
        "protocol": "v3: two-seed means; threshold = max(half-width of calibration pair means, "
        "margin x largest pairwise delta); degeneracy checked out of sample with a fresh pair",
        "recipe": recipe,
        "budget_s": args.budget,
        "device": args.device,
        "seeds": args.seeds,
        "seeds_per_experiment": len(args.seeds),
        "n_calibration_pairs": len(cal),
        "calibration_pairs": cal,
        "pair_means": pair_means,
        "mean": mean,
        "min": float(min(pair_means)),
        "max": float(max(pair_means)),
        "half_width_dice": half_width,
        "stdev": float(statistics.stdev(pair_means)) if len(pair_means) > 1 else 0.0,
        "single_run_scores": [r["val_dice_mean"] for c in cal for r in c["runs"]],
        "max_pairwise_pair_delta": worst,
        "margin": MARGIN,
        "keep_threshold_dice": threshold,
        "gate_target": GATE_TARGET,
        "gate_ok": threshold <= GATE_TARGET,
        "fresh_pair": fresh,
        "fresh_pair_delta": fresh_delta,
        "fresh_pair_kept": kept,
        "rerun_same_seed_kept": kept,  # the INS-2 panel key, kept for continuity
    }
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "budget_s",
                    "pair_means",
                    "mean",
                    "half_width_dice",
                    "max_pairwise_pair_delta",
                    "keep_threshold_dice",
                    "gate_ok",
                    "fresh_pair_delta",
                    "fresh_pair_kept",
                )
            }
        )
    )
    return 0 if not kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
