#!/usr/bin/env python3
"""Write docs/report.html: a self-contained report of the climb so far.

    python scripts/report.py [--out docs/report.html]

Reads results.tsv, the manifests, runs/*/result.json and the MLflow file
store, and writes one HTML page with inline SVG charts, a metrics strip,
a narrative generated from the ledger, and the full experiment table. No
external assets, no network, nothing private: only what the repository
already publishes.
"""

from __future__ import annotations

import argparse
import html
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hippo.ledger import read_rows  # noqa: E402

# Validated palette (dataviz reference instance; first three slots pass all-pairs).
SERIES = {"baseline": "#2a78d6", "agent": "#eb6834", "random": "#1baf7a"}
NNUNET = {
    "aug_flip": "mirroring augmentation",
    "aug_rot90": "rotation augmentation",
    "aug_affine": "spatial (affine) augmentation",
    "aug_scale_intensity": "intensity augmentation",
    "aug_gaussian_noise": "Gaussian-noise augmentation",
    "normalization": "modality-specific intensity normalization",
    "fg_oversample": "foreground oversampling",
    "loss": "Dice + cross-entropy compound loss",
    "schedule": "poly learning-rate decay",
    "optimizer": "SGD with Nesterov momentum",
    "deep_supervision": "deep supervision",
    "patch": "patch-size heuristic",
    "channels": "network width/depth heuristic",
    "strides": "network depth heuristic",
    "norm": "instance normalization",
    "batch_size": "batch-size heuristic",
}


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def f4(x) -> str:
    return f"{float(x):.4f}"


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def mlflow_history(run_id: str, metric: str) -> list[tuple[int, float]]:
    """Read a metric history straight from the MLflow file store (no import)."""
    for p in (ROOT / "mlruns").glob(f"*/{run_id}/metrics/{metric}"):
        out = []
        for line in p.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                out.append((int(parts[2]), float(parts[1])))
        return sorted(out)
    return []


# ---------------------------------------------------------------- SVG helpers


class Plot:
    def __init__(self, w=760, h=320, ml=72, mr=16, mt=16, mb=40):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, ml, mr, mt, mb
        self.parts: list[str] = []
        self.xlim = (0.0, 1.0)
        self.ylim = (0.0, 1.0)

    def set_lims(self, xs, ys, ypad=0.0):
        xs, ys = list(xs), list(ys)
        x0, x1 = (min(xs), max(xs)) if xs else (0, 1)
        y0, y1 = (min(ys), max(ys)) if ys else (0, 1)
        if x1 == x0:
            x1 = x0 + 1
        if y1 == y0:
            y1 = y0 + 1e-3
        pad = (y1 - y0) * ypad
        self.xlim, self.ylim = (x0, x1), (y0 - pad, y1 + pad)

    def px(self, x):
        x0, x1 = self.xlim
        return self.ml + (x - x0) / (x1 - x0) * (self.w - self.ml - self.mr)

    def py(self, y):
        y0, y1 = self.ylim
        return self.h - self.mb - (y - y0) / (y1 - y0) * (self.h - self.mt - self.mb)

    @staticmethod
    def ticks(lo, hi, n=5):
        import math

        span = hi - lo
        raw = span / max(n, 1)
        mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
        step = min((s for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10) * mag
        t = math.ceil(lo / step) * step
        out = []
        while t <= hi + 1e-12:
            out.append(round(t, 10))
            t += step
        return out

    def axes(self, xlabel="", ylabel="", xfmt=str, yfmt=lambda v: f"{v:.3f}", xticks=None):
        g = []
        for t in self.ticks(*self.ylim):
            y = self.py(t)
            g.append(f'<line class="grid" x1="{self.ml}" x2="{self.w - self.mr}" y1="{y:.1f}" y2="{y:.1f}"/>')
            g.append(f'<text class="tick" x="{self.ml - 6}" y="{y + 4:.1f}" text-anchor="end">{yfmt(t)}</text>')
        for t in xticks if xticks is not None else self.ticks(*self.xlim, n=8):
            x = self.px(t)
            g.append(
                f'<text class="tick" x="{x:.1f}" y="{self.h - self.mb + 16}" text-anchor="middle">{xfmt(t)}</text>'
            )
        g.append(
            f'<line class="axis" x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.h - self.mb}" y2="{self.h - self.mb}"/>'
        )
        if xlabel:
            g.append(
                f'<text class="label" x="{(self.ml + self.w - self.mr) / 2:.0f}" y="{self.h - 6}" text-anchor="middle">{esc(xlabel)}</text>'
            )
        if ylabel:
            g.append(
                f'<text class="label" transform="translate(14,{(self.mt + self.h - self.mb) / 2:.0f}) rotate(-90)" text-anchor="middle">{esc(ylabel)}</text>'
            )
        self.parts = g + self.parts

    def line(self, pts, color, width=2, dash="", cls=""):
        if len(pts) < 2:
            return
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.px(x):.1f},{self.py(y):.1f}" for i, (x, y) in enumerate(pts))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path class="{cls}" d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>')

    def step(self, pts, color, width=2, dash=""):
        if not pts:
            return
        d = f"M{self.px(pts[0][0]):.1f},{self.py(pts[0][1]):.1f}"
        for (_x0, _y0), (x1, y1) in zip(pts, pts[1:], strict=False):
            d += f" H{self.px(x1):.1f} V{self.py(y1):.1f}"
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{da}/>')

    def hline(self, y, color, dash="4 4", label=""):
        yy = self.py(y)
        self.parts.append(
            f'<line x1="{self.ml}" x2="{self.w - self.mr}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>'
        )
        if label:
            self.parts.append(
                f'<text class="note" x="{self.w - self.mr - 4}" y="{yy - 5:.1f}" text-anchor="end">{esc(label)}</text>'
            )

    def band(self, y0, y1, color):
        self.parts.insert(
            0,
            f'<rect x="{self.ml}" y="{self.py(y1):.1f}" width="{self.w - self.ml - self.mr}" height="{abs(self.py(y0) - self.py(y1)):.1f}" fill="{color}" opacity="0.10"/>',
        )

    def dot(self, x, y, color, tip, r=5, hollow=False):
        fill = "var(--surface)" if hollow else color
        self.parts.append(
            f'<circle class="pt" cx="{self.px(x):.1f}" cy="{self.py(y):.1f}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="2" data-tip="{esc(tip)}"/>'
        )

    def text(self, x, y, s, color="var(--text-2)", anchor="start", dy=0):
        self.parts.append(
            f'<text class="note" x="{self.px(x):.1f}" y="{self.py(y) + dy:.1f}" fill="{color}" text-anchor="{anchor}">{esc(s)}</text>'
        )

    def render(self, title: str) -> str:
        return (
            f"<figure><figcaption>{esc(title)}</figcaption>"
            f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{esc(title)}">'
            + "".join(self.parts)
            + "</svg></figure>"
        )


def legend(items) -> str:
    return (
        '<div class="legend">'
        + "".join(f'<span><i style="background:{c}"></i>{esc(n)}</span>' for n, c in items)
        + "</div>"
    )


# ---------------------------------------------------------------- data


def campaign_experiments(rows, band):
    """Group campaign rows (after the band's rows) into experiments: a verdict
    row plus its 'pair' rows; return per-arm lists with the two-seed mean."""
    band_exps = {r["exp"] for p in band.get("calibration_pairs", []) for r in p["runs"]}
    band_exps |= {r["exp"] for r in band.get("fresh_pair", {}).get("runs", [])}
    first_band = min(band_exps) if band_exps else 10**9
    out = {"agent": [], "random": []}
    for r in rows:
        if int(r["exp"]) <= max(band_exps, default=first_band) or r["arm"] not in out:
            continue
        if r["verdict"] == "pair" and out[r["arm"]]:
            out[r["arm"]][-1]["scores"].append(float(r["val_dice_mean"]))
            out[r["arm"]][-1]["steps"].append(int(r["steps"]))
            continue
        out[r["arm"]].append(
            {
                "exp": int(r["exp"]),
                "arm": r["arm"],
                "description": r["description"],
                "verdict": r["verdict"],
                "scores": [float(r["val_dice_mean"])],
                "steps": [int(r["steps"])],
                "ant": float(r["val_dice_ant"] or 0),
                "post": float(r["val_dice_post"] or 0),
                "commit": r["commit"],
            }
        )
    for arm in out:
        for e in out[arm]:
            e["mean"] = statistics.mean(e["scores"])
    return out


def knob_of(desc: str) -> str | None:
    d = desc.split(":", 1)[-1] if desc.startswith("random:") else desc
    first = d.strip().split(" ")[0]
    return first if first in NNUNET else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "report.html")
    args = ap.parse_args()

    rows = read_rows(ROOT / "results.tsv")
    band = load_json(ROOT / "manifests" / "w4_baseline_seed_band.json") or {}
    band300 = load_json(ROOT / "manifests" / "w2_baseline_seed_band.json") or {}
    frozen = load_json(ROOT / "manifests" / "frozen.json") or {}
    inc = {arm: load_json(ROOT / "manifests" / f"incumbent_{arm}.json") for arm in ("agent", "random")}
    base_mean = float(band.get("mean", 0))
    thr = float(band.get("keep_threshold_dice", 0))
    budget = int(float(band.get("budget_s", 0)))
    exps = campaign_experiments(rows, band) if band else {"agent": [], "random": []}
    n_target = 25

    # ---- metrics strip
    def arm_stats(arm):
        es = exps[arm]
        kept = [e for e in es if e["verdict"] == "kept"]
        cur = float(inc[arm]["val_dice_mean"]) if inc[arm] else base_mean
        best = max([e["mean"] for e in es], default=base_mean)
        return {"n": len(es), "kept": len(kept), "incumbent": cur, "best": best}

    st = {arm: arm_stats(arm) for arm in ("agent", "random")}
    tiles = [
        (
            "Baseline, two-seed mean",
            f4(base_mean),
            f"{budget} s per seed, {band.get('n_calibration_pairs', 0)} calibration pairs",
        ),
        ("Keep threshold", f4(thr), "2x largest pair-to-pair scatter, checked out of sample"),
        (
            "Agent incumbent",
            f4(st["agent"]["incumbent"]),
            f"{st['agent']['kept']} kept of {st['agent']['n']} experiments",
        ),
        (
            "Random incumbent",
            f4(st["random"]["incumbent"]),
            f"{st['random']['kept']} kept of {st['random']['n']} experiments",
        ),
        ("Campaign progress", f"{st['agent']['n']} / {n_target}", "agent experiments run of the pre-registered 25"),
        ("Claim", "0.890", "two-seed val Dice within 25 experiments per arm"),
    ]
    strip = (
        '<div class="tiles">'
        + "".join(
            f'<div class="tile"><div class="k">{esc(k)}</div><div class="v">{esc(v)}</div><div class="s">{esc(s)}</div></div>'
            for k, v, s in tiles
        )
        + "</div>"
    )

    # ---- chart 1: the climb
    p1 = Plot()
    allm = [e["mean"] for a in exps.values() for e in a] + [base_mean, base_mean + thr, 0.89]
    nmax = max(len(exps["agent"]), len(exps["random"]), 1)
    p1.set_lims([0, max(nmax, 5)], allm, ypad=0.12)
    p1.axes(
        "campaign experiment (per arm)",
        "validation Dice, mean of two seeds",
        xfmt=lambda v: f"{int(v)}",
        xticks=list(range(0, max(nmax, 5) + 1)),
    )
    p1.band(base_mean, base_mean + thr, SERIES["baseline"])
    p1.hline(base_mean, SERIES["baseline"], label=f"baseline {base_mean:.4f}")
    p1.hline(0.89, "#52514e", dash="2 4", label="claim 0.890")
    for arm in ("random", "agent"):
        off = 0.08 if arm == "random" else -0.08
        stair = [(0, base_mean)]
        cur = base_mean
        for i, e in enumerate(exps[arm], 1):
            if e["verdict"] == "kept":
                cur = e["mean"]
            stair.append((i + off, cur))
        p1.step(stair, SERIES[arm], width=2)
        for i, e in enumerate(exps[arm], 1):
            tip = f"#{e['exp']} {arm} {e['verdict']}: {e['description']} | seeds {', '.join(f4(s) for s in e['scores'])} | mean {f4(e['mean'])}"
            p1.dot(i + off, e["mean"], SERIES[arm], tip, hollow=(e["verdict"] != "kept"))
    chart1 = p1.render(
        "The climb: every experiment's two-seed mean, with each arm's incumbent as a staircase"
    ) + legend(
        [
            ("agent (filled = kept)", SERIES["agent"]),
            ("random-edit control", SERIES["random"]),
            ("baseline + keep band", SERIES["baseline"]),
        ]
    )

    # ---- chart 2: the noise floor
    p2 = Plot(h=300)
    b300 = [r for r in rows if r["arm"] == "baseline" and float(r["train_seconds"]) < budget]
    b600 = [r for r in rows if r["arm"] == "baseline" and float(r["train_seconds"]) >= budget and budget]
    ys = [float(r["val_dice_mean"]) for r in b300 + b600]
    p2.set_lims([0, 3], ys + [base_mean + thr], ypad=0.15)
    p2.axes("", "validation Dice", xfmt=lambda v: "", xticks=[])
    p2.text(
        1, p2.ylim[1], f"{int(float(b300[0]['train_seconds'])) if b300 else 300} s, single runs", anchor="middle", dy=12
    )
    p2.text(2, p2.ylim[1], f"{budget} s, pairs and pair means", anchor="middle", dy=12)
    p2.band(base_mean, base_mean + thr, SERIES["baseline"])
    p2.hline(base_mean + thr, SERIES["baseline"], label=f"keep line {base_mean + thr:.4f}")
    import random as _r

    jit = _r.Random(1)  # noqa: S311 (jitter for plotting only)
    for r in b300:
        p2.dot(
            1 + jit.uniform(-0.18, 0.18),
            float(r["val_dice_mean"]),
            SERIES["baseline"],
            f"#{r['exp']} seed {r['seed']}: {f4(r['val_dice_mean'])} ({r['description']})",
            r=4,
            hollow=True,
        )
    for pair in band.get("calibration_pairs", []) + ([band["fresh_pair"]] if band.get("fresh_pair") else []):
        for run in pair["runs"]:
            p2.dot(
                2 + (0.14 if run["seed"] == 2027 else -0.14),
                run["val_dice_mean"],
                SERIES["baseline"],
                f"#{run['exp']} seed {run['seed']}: {f4(run['val_dice_mean'])} (pair {pair['pair']}, {pair.get('kind', 'calibration')})",
                r=4,
                hollow=True,
            )
        p2.dot(
            2,
            pair["mean"],
            SERIES["agent"] if pair.get("kind") == "fresh" else SERIES["baseline"],
            f"pair {pair['pair']} mean {f4(pair['mean'])}"
            + (" (fresh, out-of-sample: not kept)" if pair.get("kind") == "fresh" else ""),
            r=6,
        )
    chart2 = p2.render(
        "The noise floor: the unchanged recipe, re-run. Filled = two-seed pair means; the orange mean is the out-of-sample check"
    ) + legend([("single-seed runs", SERIES["baseline"]), ("fresh pair (must not be kept)", SERIES["agent"])])

    # ---- chart 3: per class
    p3 = Plot(w=420, h=260)
    agent_inc = inc["agent"]
    base_runs = [r for p in band.get("calibration_pairs", []) for r in p["runs"]]
    base_rows = {int(r["exp"]): r for r in rows}
    bant = statistics.mean(float(base_rows[r["exp"]]["val_dice_ant"]) for r in base_runs) if base_runs else 0
    bpost = statistics.mean(float(base_rows[r["exp"]]["val_dice_post"]) for r in base_runs) if base_runs else 0
    if agent_inc:
        iant = (
            statistics.mean(
                float(base_rows[r.get("exp", 0)]["val_dice_ant"])
                for r in agent_inc["runs"]
                if r.get("exp", 0) in base_rows
            )
            if any(r.get("exp") in base_rows for r in agent_inc["runs"])
            else None
        )
        if iant is None:
            # incumbent json carries no exp ids: look the rows up by mlflow run id
            byrun = {r["mlflow_run_id"]: r for r in rows}
            sel = [byrun[r["mlflow_run_id"]] for r in agent_inc["runs"] if r["mlflow_run_id"] in byrun]
            iant = statistics.mean(float(r["val_dice_ant"]) for r in sel) if sel else bant
            ipost = statistics.mean(float(r["val_dice_post"]) for r in sel) if sel else bpost
        else:
            ipost = statistics.mean(float(base_rows[r["exp"]]["val_dice_post"]) for r in agent_inc["runs"])
    else:
        iant, ipost = bant, bpost
    vals = [bant, iant, bpost, ipost]
    p3.set_lims([0, 4], vals + [0.84], ypad=0.3)
    p3.ylim = (min(vals + [0.85]) - 0.01, max(vals) + 0.01)
    p3.axes("", "validation Dice", xfmt=lambda v: "", xticks=[])
    for k, (lab, v, col) in enumerate(
        [
            ("anterior / baseline", bant, SERIES["baseline"]),
            ("anterior / agent", iant, SERIES["agent"]),
            ("posterior / baseline", bpost, SERIES["baseline"]),
            ("posterior / agent", ipost, SERIES["agent"]),
        ]
    ):
        x0, x1 = p3.px(k + 0.25), p3.px(k + 0.75)
        y, yb = p3.py(v), p3.py(p3.ylim[0])
        p3.parts.append(
            f'<rect class="pt" x="{x0:.1f}" y="{y:.1f}" width="{x1 - x0:.1f}" height="{yb - y:.1f}" rx="4" fill="{col}" data-tip="{esc(lab)}: {f4(v)}"/>'
        )
        p3.parts.append(
            f'<text class="tick" x="{(x0 + x1) / 2:.1f}" y="{yb + 14:.1f}" text-anchor="middle">{"ant" if k < 2 else "post"}</text>'
        )
        p3.parts.append(
            f'<text class="note" x="{(x0 + x1) / 2:.1f}" y="{y - 5:.1f}" text-anchor="middle">{v:.3f}</text>'
        )
    chart3 = p3.render("Per class: anterior and posterior hippocampus, baseline vs agent incumbent") + legend(
        [("baseline (calibration mean)", SERIES["baseline"]), ("agent incumbent", SERIES["agent"])]
    )

    # ---- chart 4: loss curves (baseline pair 1 seed 1337 vs latest agent incumbent seed 1337)
    p4 = Plot(w=420, h=260)
    curves = []
    if base_runs:
        curves.append(("baseline", base_runs[0]["mlflow_run_id"], SERIES["baseline"]))
    if agent_inc:
        curves.append(("agent incumbent", agent_inc["runs"][0]["mlflow_run_id"], SERIES["agent"]))
    series = []
    for name, rid, col in curves:
        h = mlflow_history(rid, "train_loss")
        if h:
            # smooth: median of a window, thinned to <= 300 points
            n = max(1, len(h) // 300)
            sm = [(h[i][0], statistics.median(v for _, v in h[i : i + n])) for i in range(0, len(h), n)]
            series.append((name, sm, col))
    if series:
        allx = [x for _, s, _ in series for x, _ in s]
        ally = [y for _, s, _ in series for _, y in s]
        p4.set_lims(allx, ally, ypad=0.05)
        p4.axes(
            "optimizer step",
            "training loss (windowed median)",
            xfmt=lambda v: f"{int(v / 1000)}k",
            yfmt=lambda v: f"{v:.3f}",
        )
        for _name, sm, col in series:
            p4.line(sm, col, width=2)
        chart4 = p4.render("Training loss, seed 1337: baseline vs agent incumbent (from the MLflow store)") + legend(
            [(n, c) for n, _, c in series]
        )
    else:
        chart4 = '<figure><figcaption>Training loss</figcaption><p class="muted">MLflow store not present in this checkout.</p></figure>'

    # ---- narrative
    kept_agent = [e for e in exps["agent"] if e["verdict"] == "kept"]
    rev_agent = [e for e in exps["agent"] if e["verdict"] != "kept"]
    kept_random = [e for e in exps["random"] if e["verdict"] == "kept"]

    def comp(e):
        k = knob_of(e["description"])
        return f" (nnU-Net: {NNUNET[k]})" if k else ""

    gain = st["agent"]["incumbent"] - base_mean
    narrative = f"""
<p>A deliberately plain 3-D U-Net is the starting point: min-max normalization, uniform 32&#179; patches,
plain cross-entropy, Adam at a flat 1e-3, no augmentation, three levels of 16/32/64 channels. Trained for
{budget} s per seed on one GPU it scores <b>{base_mean:.4f}</b> mean Dice on the frozen validation split
(mean of seeds 1337 and 2027, averaged over {band.get("n_calibration_pairs", 0)} calibration pairs).
Re-running that unchanged recipe moves the two-seed mean by up to {float(band.get("max_pairwise_pair_delta", 0)):.4f},
so the keep rule demands a gain above <b>{thr:.4f}</b>, twice that scatter. A fresh pair that took no part in
setting the threshold scored {float(band.get("fresh_pair_delta", 0)):+.4f} and was correctly not kept.</p>
<p>Two arms climb from that point under the same budget and the same rule, alternating one experiment each.
The <b>agent</b> reads the baseline and its own rows, edits one knob in <code>recipe.py</code>, and says why.
The <b>random-edit control</b> draws one mutation from the same knob space. Neither sees the hold-out split.
The pre-registered claim is 0.890 within 25 experiments per arm, with at least three kept edits that map to
named nnU-Net components; the kill condition is the control matching the agent within the noise floor.</p>
<p><b>So far:</b> the agent has run {st["agent"]["n"]} experiment{"s" if st["agent"]["n"] != 1 else ""} and kept
{st["agent"]["kept"]}, moving its incumbent to <b>{st["agent"]["incumbent"]:.4f}</b> ({gain:+.4f} over the baseline).
The control has run {st["random"]["n"]} and kept {st["random"]["kept"]}, incumbent {st["random"]["incumbent"]:.4f}.</p>
"""
    if kept_agent:
        narrative += (
            "<p><b>Kept by the agent:</b></p><ul>"
            + "".join(
                f"<li>#{e['exp']} {esc(e['description'])}{esc(comp(e))}: seeds {', '.join(f4(s) for s in e['scores'])}, mean {f4(e['mean'])}.</li>"
                for e in kept_agent
            )
            + "</ul>"
        )
    if rev_agent:
        narrative += (
            "<p><b>Tried and reverted by the agent:</b></p><ul>"
            + "".join(f"<li>#{e['exp']} {esc(e['description'])}: mean {f4(e['mean'])}.</li>" for e in rev_agent)
            + "</ul>"
        )
    if kept_random:
        narrative += (
            "<p><b>Kept by the random control:</b> "
            + "; ".join(f"#{e['exp']} {esc(e['description'])} ({f4(e['mean'])})" for e in kept_random)
            + ".</p>"
        )
    if band300:
        narrative += f"""<p class="muted">History: at a 300 s single-seed budget (rows 1 to 21) the same baseline scored
{float(band300.get("mean", 0)):.4f} and the same-seed scatter forced a 0.011 gate under which all ten dry-run edits
reverted. The campaign bought resolution with two seeds at {budget} s. The {budget} s baseline is lower than the 300 s one:
the flat schedule does not turn extra steps into Dice, which is itself a recipe fact.</p>"""

    # ---- table
    def trow(r):
        return (
            f"<tr class='{esc(r['arm'])}'><td>{esc(r['exp'])}</td><td>{esc(r['arm'])}</td><td>{esc(r['seed'])}</td>"
            f"<td>{f4(r['val_dice_mean'])}</td><td>{f4(r['val_dice_ant'])}</td><td>{f4(r['val_dice_post'])}</td>"
            f"<td>{int(float(r['train_seconds']))}</td><td>{esc(r['steps'])}</td><td>{esc(r['description'])}</td><td>{esc(r['verdict'])}</td></tr>"
        )

    table = (
        "<table><thead><tr><th>exp</th><th>arm</th><th>seed</th><th>Dice</th><th>ant</th><th>post</th><th>s</th><th>steps</th><th>description</th><th>verdict</th></tr></thead><tbody>"
        + "".join(trow(r) for r in rows)
        + "</tbody></table>"
    )

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scorer = (frozen.get("files", {}) or {}).get("eval/score.py", frozen.get("sha256", ""))
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hippocampus-autoresearch: campaign report</title>
<style>
:root{{color-scheme:light;--surface:#fcfcfb;--text-1:#0b0b0b;--text-2:#52514e;--text-3:#8a8984;--line:#e3e2dd;--card:#ffffff}}
body{{margin:0;background:var(--surface);color:var(--text-1);font:15px/1.5 -apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:32px 20px 64px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:18px;margin:36px 0 10px}} .sub{{color:var(--text-2);margin:0 0 20px}}
.tiles{{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:18px 0}} @media(max-width:900px){{.tiles{{grid-template-columns:repeat(3,1fr)}}}} @media(max-width:520px){{.tiles{{grid-template-columns:1fr 1fr}}}}
.tile{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}}
.tile .k{{font-size:12px;color:var(--text-2)}} .tile .v{{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}} .tile .s{{font-size:11px;color:var(--text-3)}}
figure{{margin:16px 0 4px;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}}
figcaption{{font-size:13px;color:var(--text-2);margin-bottom:6px}}
svg{{width:100%;height:auto;display:block}} .grid{{stroke:var(--line);stroke-width:1}} .axis{{stroke:#bdbcb6;stroke-width:1}}
.tick{{font-size:11px;fill:var(--text-3)}} .label{{font-size:12px;fill:var(--text-2)}} .note{{font-size:11px;fill:var(--text-2)}}
.pt{{cursor:pointer}} .pt:hover{{stroke-width:3}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:var(--text-2);margin:4px 0 12px 4px}}
.legend i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} @media(max-width:800px){{.row{{grid-template-columns:1fr}}}}
table{{border-collapse:collapse;width:100%;font-size:12px;font-variant-numeric:tabular-nums}} th,td{{text-align:left;padding:4px 6px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--text-2);font-weight:600;position:sticky;top:0;background:var(--surface)}} td:nth-child(9){{max-width:420px}}
tr.agent td:nth-child(2){{color:{SERIES["agent"]}}} tr.random td:nth-child(2){{color:{SERIES["random"]}}} tr.baseline td:nth-child(2){{color:{SERIES["baseline"]}}}
.muted{{color:var(--text-2)}} code{{font-size:13px;background:#f1f0ec;padding:1px 4px;border-radius:3px}}
#tip{{position:fixed;pointer-events:none;background:#0b0b0b;color:#fcfcfb;font-size:12px;padding:6px 8px;border-radius:4px;max-width:360px;display:none;z-index:9}}
details summary{{cursor:pointer;color:var(--text-2)}}
</style></head><body><main>
<h1>Can a coding agent rediscover nnU-Net?</h1>
<p class="sub">MSD Task04 Hippocampus. Campaign 1 report, generated {now}. Frozen scorer sha256 <code>{esc(scorer)[:12]}&#8230;</code>.
Method: <a href="METHOD.md">METHOD.md</a>. Background: <a href="AUTORESEARCH.md">AUTORESEARCH.md</a>.</p>
{strip}
<h2>What is happening</h2>
{narrative}
<h2>The climb</h2>
{chart1}
<h2>The ruler's resolution</h2>
{chart2}
<div class="row"><div>{chart3}</div><div>{chart4}</div></div>
<h2>Every experiment</h2>
<details open><summary>{len(rows)} rows of results.tsv (one row per seed; the 'pair' verdict marks the second seed of a two-seed experiment)</summary>{table}</details>
<p class="muted">The hold-out split is never scored by this loop; it is scored once, by a person, at the close.
The random arm's rows are the control and are not read by the agent when it chooses its next edit.</p>
</main><div id="tip"></div>
<script>
const tip=document.getElementById('tip');
document.querySelectorAll('.pt').forEach(el=>{{
  el.addEventListener('mousemove',e=>{{tip.textContent=el.dataset.tip;tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+12)+'px';}});
  el.addEventListener('mouseleave',()=>tip.style.display='none');
}});
</script></body></html>"""
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page.rstrip("\n") + "\n")
    print(
        f"wrote {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out} ({len(page) // 1024} kB, {len(rows)} rows)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
