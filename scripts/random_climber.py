#!/usr/bin/env python3
"""The random-edit null climber. Same knob space, same budget, same keep rule.

    python scripts/random_climber.py --n 50 --budget 300 --seed 7

Each iteration mutates ONE knob of the arm's incumbent recipe, drawn
uniformly from KNOBS (choices, or a range, log-uniform where declared),
writes it to runs/random/recipe_<i>.json, trains it with train.py
--recipe-json, applies scripts/decide.py against manifests/incumbent_random.json,
and records the row with arm=random. It never touches recipe.py, so it can
run beside the agent arm (one at a time on the GPU of record).
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

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
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--budget", type=float, default=300.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=7, help="the mutation RNG seed; training seed is fixed at 1337")
    ap.add_argument("--band", type=Path, default=ROOT / "manifests" / "w2_baseline_seed_band.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    recipe_mod = importlib.import_module("recipe")
    knobs = recipe_mod.KNOBS
    # Start from the baseline recipe recorded by the seed band, never from the
    # working-tree recipe.py, which the agent arm mutates.
    base = json.loads(args.band.read_text())["recipe"]
    out_dir = ROOT / "runs" / "random"
    out_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    for i in range(args.n):
        incumbent = json.loads(INCUMBENT.read_text())["recipe"] if INCUMBENT.exists() else base
        new, desc = mutate(incumbent, knobs, rng)
        rj = out_dir / f"recipe_{int(time.time())}_{i}.json"
        rj.write_text(json.dumps(new, indent=1) + "\n")
        run = ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{ARM}-s1337"
        r = subprocess.run(
            [
                py,
                str(ROOT / "train.py"),
                "--seed",
                "1337",
                "--budget",
                str(args.budget),
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
                [py, "-m", "hippo.record", "--refused", "--arm", ARM, "--desc", desc, "--seed", "1337"],
                check=True,
                cwd=ROOT,
                env={**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")},
            )
            print(f"[{i + 1}/{args.n}] refused {desc}")
            continue
        if r.returncode != 0:
            print(r.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"train.py failed at iteration {i}")
        env = {**dict(__import__("os").environ), "PYTHONPATH": str(ROOT / "src")}
        d = subprocess.run(
            [
                py,
                str(ROOT / "scripts" / "decide.py"),
                "--result",
                str(run / "result.json"),
                "--band",
                str(args.band),
                "--arm",
                ARM,
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if d == "confirm":
            run2 = ROOT / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{ARM}-s2027"
            subprocess.run(
                [
                    py,
                    str(ROOT / "train.py"),
                    "--seed",
                    "2027",
                    "--budget",
                    str(args.budget),
                    "--device",
                    args.device,
                    "--desc",
                    desc + " (confirm seed)",
                    "--arm",
                    ARM,
                    "--no-append",
                    "--recipe-json",
                    str(rj),
                    "--run-dir",
                    str(run2),
                ],
                check=True,
            )
            d = subprocess.run(
                [
                    py,
                    str(ROOT / "scripts" / "decide.py"),
                    "--result",
                    str(run / "result.json"),
                    "--confirm",
                    str(run2 / "result.json"),
                    "--band",
                    str(args.band),
                    "--arm",
                    ARM,
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            subprocess.run(
                [py, "-m", "hippo.record", "--result", str(run2 / "result.json"), "--verdict", "confirm", "--arm", ARM],
                check=True,
                cwd=ROOT,
                env=env,
            )
        subprocess.run(
            [py, "-m", "hippo.record", "--result", str(run / "result.json"), "--verdict", d, "--arm", ARM],
            check=True,
            cwd=ROOT,
            env=env,
        )
        if d == "kept":
            subprocess.run(
                [py, str(ROOT / "scripts" / "decide.py"), "--promote", str(run / "result.json"), "--arm", ARM],
                check=True,
            )
        print(f"[{i + 1}/{args.n}] {d} {desc} val={json.loads((run / 'result.json').read_text())['val_dice_mean']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
