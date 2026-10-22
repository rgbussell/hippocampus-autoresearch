#!/usr/bin/env python3
"""INS-2: measure the baseline noise floor and run the degeneracy check.

    python scripts/seed_band.py --budget 300 --seeds 1337 2027 3117 --device cuda

Runs the CURRENT recipe.py (the plain baseline at W2) once per seed with
--verdict baseline, then re-runs the first seed once more. Writes
manifests/w2_baseline_seed_band.json with:

  n_seeds, scores, mean, min, max, half_width_dice, stdev
  rerun_same_seed_delta     |score(rerun) - score(first)|   the nondeterminism band
  rerun_same_seed_kept      whether the keep rule would have KEPT the unchanged
                            recipe (it must not; if it would, the gate is noise)

The keep threshold used from then on is half_width_dice. If the same-seed
rerun delta exceeds it, the threshold is raised to that delta and the
manifest says so.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import keep_decision, seed_band  # noqa: E402

OUT = ROOT / "manifests" / "w2_baseline_seed_band.json"


def run(seed: int, budget: float, device: str, desc: str, verdict: str) -> dict:
    run_dir = ROOT / "runs" / f"w2-{desc.replace(' ', '-')}-s{seed}"
    cmd = [sys.executable, str(ROOT / "train.py"), "--seed", str(seed), "--budget", str(budget),
           "--device", device, "--desc", desc, "--verdict", verdict, "--run-dir", str(run_dir)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"seed {seed} failed with code {r.returncode}")
    return json.loads((run_dir / "result.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 2027, 3117])
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    results = [run(s, args.budget, args.device, "plain baseline", "baseline") for s in args.seeds]
    scores = [r["val_dice_mean"] for r in results]
    band = seed_band(scores)
    rerun = run(args.seeds[0], args.budget, args.device, "plain baseline same-seed rerun", "baseline")
    delta = abs(rerun["val_dice_mean"] - scores[0])
    threshold = max(band["half_width_dice"], delta)
    kept = keep_decision(rerun["val_dice_mean"], scores[0], band["half_width_dice"]) == "kept"
    out = dict(band)
    out.update({
        "budget_s": args.budget, "device": args.device, "seeds": args.seeds,
        "runs": [{"seed": r["seed"], "val_dice_mean": r["val_dice_mean"], "steps": r["steps"],
                  "mlflow_run_id": r["mlflow_run_id"], "exp": r["exp"]} for r in results],
        "rerun_same_seed": {"val_dice_mean": rerun["val_dice_mean"], "steps": rerun["steps"],
                            "mlflow_run_id": rerun["mlflow_run_id"], "exp": rerun["exp"]},
        "rerun_same_seed_delta": delta,
        "rerun_same_seed_kept": kept,
        "keep_threshold_dice": threshold,
        "keep_threshold_note": ("half-width of the seed band" if threshold == band["half_width_dice"]
                                else "raised to the same-seed rerun delta, which exceeded the half-width"),
    })
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_seeds", "scores", "mean", "half_width_dice",
                                           "rerun_same_seed_delta", "rerun_same_seed_kept", "keep_threshold_dice")}))
    return 0 if not kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
