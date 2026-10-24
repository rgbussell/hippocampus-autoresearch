#!/usr/bin/env python3
"""The random-edit null climber. Same knob space, same budget, same seeds, same keep rule.

    python scripts/random_climber.py --n 1 --band manifests/w4_baseline_seed_band.json

Each iteration mutates ONE knob of the arm's incumbent recipe, drawn
uniformly from KNOBS (choices, or a range, log-uniform where declared),
writes it to runs/random/recipe_<i>.json, trains it with train.py
--recipe-json once per seed (protocol v3: two seeds, scored on the mean),
applies scripts/decide.py against manifests/incumbent_random.json, and
records one row per seed with arm=random. It never touches recipe.py and
never commits; the caller commits results.tsv. The mutation RNG seed defaults
to 7 + the number of random experiments already in results.tsv, so repeated
--n 1 calls walk one reproducible sequence.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import read_rows  # noqa: E402

ARM = "random"
INCUMBENT = ROOT / "manifests" / f"incumbent_{ARM}.json"


def mutate(recipe: dict, knobs: dict, rng: np.random.Generator) -> tuple[dict, str]:
    for _ in range(100):
        k = rng.choice(sorted(knobs))
        spec = knobs[k]
        if "choices" in spec:
            v = spec["choices"][rng.integers(len(spec["choices"]))]
        else:
            lo, hi = spec["range"]
            v = (
                float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
                if spec.get("log")
                else float(rng.uniform(lo, hi))
            )
            v = round(v, 6)
        same = (
            (list(v) == list(recipe[k]))
            if isinstance(v, (list, tuple)) and isinstance(recipe[k], (list, tuple))
            else (v == recipe[k])
        )
        if not same:
            v = list(v) if isinstance(v, tuple) else v
            new = dict(recipe, **{k: v})
            if len(new["strides"]) != len(new["channels"]) - 1:
                continue  # inconsistent; draw again
            return new, f"random: {k} {recipe[k]!r} -> {v!r}"
    raise RuntimeError("could not find a mutation")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--budget", type=float, default=None, help="default: the band's budget_s")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--seeds", type=int, nargs="+", default=[1337, 2027], help="training seeds; the score is their mean"
    )
    ap.add_argument("--seed", type=int, default=None, help="mutation RNG seed (default 7 + random experiments so far)")
    ap.add_argument("--band", type=Path, default=ROOT / "manifests" / "w4_baseline_seed_band.json")
    args = ap.parse_args()
    band_doc = json.loads(args.band.read_text())
    budget = args.budget if args.budget is not None else float(band_doc["budget_s"])
    done = sum(
        1
        for r in read_rows(ROOT / "results.tsv")
        if r["arm"] == ARM and r["verdict"] in ("kept", "reverted", "refused")
    )
    rng = np.random.default_rng(args.seed if args.seed is not None else 7 + done)
    recipe_mod = importlib.import_module("recipe")
    knobs = recipe_mod.KNOBS
    # Start from the baseline recipe recorded by the seed band, never from the
    # working-tree recipe.py, which the agent arm mutates.
    base = band_doc["recipe"]
    out_dir = ROOT / "runs" / "random"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    env = {**dict(os.environ), "PYTHONPATH": str(ROOT / "src")}
    for i in range(args.n):
        incumbent = json.loads(INCUMBENT.read_text())["recipe"] if INCUMBENT.exists() else base
        new, desc = mutate(incumbent, knobs, rng)
        rj = out_dir / f"recipe_{int(time.time())}_{i}.json"
        rj.write_text(json.dumps(new, indent=1) + "\n")
        results: list[Path] = []
        refused = False
        for seed in args.seeds:
            run = ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{ARM}-s{seed}"
            r = subprocess.run(
                [
                    py,
                    str(ROOT / "train.py"),
                    "--seed",
                    str(seed),
                    "--budget",
                    str(budget),
                    "--device",
                    args.device,
                    "--desc",
                    desc,
                    "--arm",
                    ARM,
                    "--no-append",
                    "--recipe-json",
                    str(rj),
                    "--run-dir",
                    str(run),
                ],
                capture_output=True,
                text=True,
            )
            if r.returncode == 3:
                subprocess.run(
                    [py, "-m", "hippo.record", "--refused", "--arm", ARM, "--desc", desc, "--seed", str(seed)],
                    check=True,
                    cwd=ROOT,
                    env=env,
                )
                print(f"[{i + 1}/{args.n}] refused {desc}")
                refused = True
                break
            if r.returncode != 0:
                print(r.stderr[-2000:], file=sys.stderr)
                raise SystemExit(f"train.py failed at iteration {i}")
            results.append(run / "result.json")
        if refused:
            continue
        d = subprocess.run(
            [
                py,
                str(ROOT / "scripts" / "decide.py"),
                "--result",
                *map(str, results),
                "--band",
                str(args.band),
                "--arm",
                ARM,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        verdict = d
        for res in results:
            subprocess.run(
                [py, "-m", "hippo.record", "--result", str(res), "--verdict", verdict, "--arm", ARM],
                check=True,
                cwd=ROOT,
                env=env,
            )
            verdict = "pair"
        if d == "kept":
            subprocess.run(
                [py, str(ROOT / "scripts" / "decide.py"), "--promote", *map(str, results), "--arm", ARM], check=True
            )
        mean = sum(json.loads(p.read_text())["val_dice_mean"] for p in results) / len(results)
        print(f"[{i + 1}/{args.n}] {d} {desc} mean={mean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
