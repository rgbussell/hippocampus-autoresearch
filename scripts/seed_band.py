#!/usr/bin/env python3
"""INS-2: measure the baseline noise floor, set the keep threshold, and check
it out of sample (protocol v2, after EXP-005).

    python scripts/seed_band.py --budget 300 --seeds 1337 2027 3117 --device cuda
    python scripts/seed_band.py --reuse            # keep finished runs from the manifest

Runs the CURRENT recipe.py (the plain baseline at W2):

  1. once per seed                      -> cross-seed band, half_width_dice
  2. --calibration-reruns more times with the first seed
                                        -> same-seed scatter (wall-clock budget + cuDNN)
  3. keep_threshold_dice = max(half_width, 2 x largest pairwise same-seed |delta|)
  4. one FRESH same-seed rerun that took no part in step 3
                                        -> degeneracy check: it must NOT be kept

Writes manifests/w2_baseline_seed_band.json. Exit 1 if the fresh rerun would
have been kept: the gate is then measuring noise and the campaign must not
start. EXP-005 showed why the fresh run matters: a threshold set equal to the
one delta it is checked against passes by construction.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import keep_decision, keep_threshold, pairwise_max_delta, seed_band  # noqa: E402

OUT = ROOT / "manifests" / "w2_baseline_seed_band.json"
MARGIN = 2.0


def run(seed: int, budget: float, device: str, desc: str, verdict: str) -> dict:
    run_dir = ROOT / "runs" / f"w2-{desc.replace(' ', '-')}-s{seed}"
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
        verdict,
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
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 2027, 3117])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--calibration-reruns", type=int, default=3)
    ap.add_argument("--reuse", action="store_true", help="reuse finished runs recorded in the manifest")
    args = ap.parse_args()

    prev: dict = {}
    if args.reuse and OUT.exists():
        prev = json.loads(OUT.read_text())
        if prev.get("budget_s") != args.budget or prev.get("device") != args.device:
            raise SystemExit("--reuse refused: budget or device differ from the manifest")
    prev_runs = {r["seed"]: r for r in prev.get("runs", [])}
    prev_cal = list(prev.get("calibration_reruns", []))
    if "rerun_same_seed" in prev and not prev_cal:  # first-pass manifest (EXP-005)
        prev_cal = [prev["rerun_same_seed"]]

    results = [
        prev_runs.get(s) or slim(run(s, args.budget, args.device, "plain baseline", "baseline")) for s in args.seeds
    ]
    recipe = prev.get("recipe")
    if recipe is None:
        first = ROOT / "runs" / f"w2-plain-baseline-s{args.seeds[0]}" / "result.json"
        recipe = json.loads(first.read_text())["recipe"]
    scores = [r["val_dice_mean"] for r in results]
    band = seed_band(scores)

    s0 = args.seeds[0]
    cal = list(prev_cal)
    for i in range(len(cal), args.calibration_reruns):
        cal.append(slim(run(s0, args.budget, args.device, f"plain baseline same-seed rerun {i + 1}", "baseline")))
    same_seed_scores = [scores[0]] + [c["val_dice_mean"] for c in cal]
    worst = pairwise_max_delta(same_seed_scores)
    threshold = keep_threshold(band["half_width_dice"], [worst], MARGIN)

    fresh = prev.get("fresh_rerun")
    if fresh is None:
        fresh = slim(run(s0, args.budget, args.device, "plain baseline fresh rerun (out-of-sample check)", "baseline"))
    fresh_delta = fresh["val_dice_mean"] - scores[0]
    kept = keep_decision(fresh["val_dice_mean"], scores[0], threshold) == "kept"
    half_width_kept = keep_decision(fresh["val_dice_mean"], scores[0], band["half_width_dice"]) == "kept"

    out = dict(band)
    out.update(
        {
            "protocol": "v2: threshold = max(half_width, margin x largest pairwise same-seed delta); "
            "degeneracy checked out of sample with a fresh same-seed rerun",
            "recipe": recipe,
            "budget_s": args.budget,
            "device": args.device,
            "seeds": args.seeds,
            "runs": results,
            "calibration_reruns": cal,
            "n_calibration_same_seed_reruns": len(cal),
            "same_seed_scores": same_seed_scores,
            "max_pairwise_same_seed_delta": worst,
            "margin": MARGIN,
            "keep_threshold_dice": threshold,
            "keep_threshold_note": (
                "half-width of the seed band"
                if threshold == band["half_width_dice"]
                else f"{MARGIN:g} x the largest pairwise same-seed delta (exceeds the half-width)"
            ),
            "fresh_rerun": fresh,
            "fresh_rerun_delta": fresh_delta,
            "fresh_rerun_kept": kept,
            "rerun_same_seed_kept": kept,  # the INS-2 panel key: out-of-sample, under keep_threshold_dice
            "fresh_rerun_kept_under_half_width": half_width_kept,
        }
    )
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(
        json.dumps(
            {
                k: out[k]
                for k in (
                    "n_seeds",
                    "scores",
                    "mean",
                    "half_width_dice",
                    "same_seed_scores",
                    "max_pairwise_same_seed_delta",
                    "keep_threshold_dice",
                    "fresh_rerun_delta",
                    "fresh_rerun_kept",
                )
            }
        )
    )
    return 0 if not kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
