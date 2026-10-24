"""INS-2 keep rule and the results ledger writer."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import (  # noqa: E402
    COLUMNS,
    append_row,
    keep_decision,
    keep_threshold,
    next_exp,
    pairwise_max_delta,
    read_rows,
    seed_band,
)


def test_seed_band_half_width():
    b = seed_band([0.80, 0.82, 0.81])
    assert b["n_seeds"] == 3 and abs(b["half_width_dice"] - 0.01) < 1e-12


def test_keep_rule_thresholds():
    hw = 0.01
    assert keep_decision(0.83, 0.81, hw) == "kept"  # gain 0.02 > hw
    assert keep_decision(0.815, 0.81, hw) == "confirm"  # 0 < gain <= 2hw
    assert keep_decision(0.81, 0.81, hw) == "reverted"  # tie
    assert keep_decision(0.80, 0.81, hw) == "reverted"  # worse
    assert keep_decision(0.8101, 0.81, hw) == "confirm"  # tiny positive gain asks for a second seed


def test_keep_threshold_sits_above_same_seed_scatter():
    # EXP-005: half-width 0.0001 but same-seed scatter 0.0006 -> the gate must clear the scatter with margin
    thr = keep_threshold(0.0001, [0.0006, -0.0004])
    assert abs(thr - 0.0012) < 1e-12
    assert keep_decision(0.8783, 0.8777, thr) != "kept"  # the observed rerun is no longer a keep
    assert keep_threshold(0.01, [0.001]) == 0.01  # wide seed band dominates
    assert keep_threshold(0.0, []) == 0.0
    assert abs(pairwise_max_delta([0.8777, 0.8784, 0.8775]) - 0.0009) < 1e-12


def test_unchanged_recipe_is_not_kept():
    # degeneracy: same score as incumbent must never be kept
    assert keep_decision(0.8123, 0.8123, 0.004) != "kept"


def test_ledger_append_and_numbering(tmp_path):
    p = tmp_path / "results.tsv"
    row = {c: "" for c in COLUMNS}
    row.update(
        exp=next_exp(p),
        arm="agent",
        commit="abc",
        seed=1,
        val_dice_mean=0.8,
        description="first\tline\nbreak",
        verdict="baseline",
        mlflow_run_id="x",
    )
    assert append_row(p, row) == 1
    row2 = dict(row, exp=next_exp(p), description="second")
    assert append_row(p, row2) == 2
    rows = read_rows(p)
    assert [r["exp"] for r in rows] == ["1", "2"]
    assert "\t" not in rows[0]["description"] and "\n" not in rows[0]["description"]


def test_missing_column_is_refused(tmp_path):
    with pytest.raises(ValueError):
        append_row(tmp_path / "r.tsv", {"exp": 1})
