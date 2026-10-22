#!/usr/bin/env python3
"""Render the training curve of one or more MLflow runs from the repo store.

    python scripts/show_curves.py <run_id> [<run_id> ...] [--out docs/curves.png]
    python scripts/show_curves.py --latest 3

Reads train_loss and lr histories plus the final val Dice from the MLflow
file store (mlruns/ or MLFLOW_TRACKING_URI). A kept change should be
inspected, not just trusted: a loss curve that never moved is a budget
problem, not a recipe result.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_ids", nargs="*")
    ap.add_argument("--latest", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "curves.png")
    args = ap.parse_args()
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", f"file:{ROOT / 'mlruns'}"))
    client = MlflowClient()
    exp = client.get_experiment_by_name("hippocampus-autoresearch")
    if exp is None:
        raise SystemExit("no experiment yet")
    ids = list(args.run_ids)
    if args.latest:
        runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"], max_results=args.latest)
        ids += [r.info.run_id for r in runs]
    if not ids:
        raise SystemExit("give run ids or --latest N")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for rid in ids:
        r = client.get_run(rid)
        hist = client.get_metric_history(rid, "train_loss")
        xs = [h.step for h in hist]; ys = [h.value for h in hist]
        label = f"{r.data.tags.get('desc', '')[:24]} s{r.data.params.get('seed')} val={r.data.metrics.get('val_dice_mean', float('nan')):.3f}"
        a1.plot(xs, ys, label=label)
        lr = client.get_metric_history(rid, "lr")
        a2.plot([h.step for h in lr], [h.value for h in lr], label=label)
        print(f"{rid[:8]}  {label}  steps={r.data.metrics.get('steps')}  s={r.data.metrics.get('train_seconds')}")
    a1.set_xlabel("optimizer step"); a1.set_ylabel("train loss"); a1.set_title("training loss")
    a2.set_xlabel("optimizer step"); a2.set_ylabel("lr"); a2.set_title("learning rate")
    a1.legend(fontsize=7)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
