"""INS-2 keep rule and the results ledger writer."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.ledger import COLUMNS, append_row, keep_decision, next_exp, read_rows, seed_band  # noqa: E402


def test_seed_band_half_width():
    b = seed_band([0.80, 0.82, 0.81])
    assert b["n_seeds"] == 3 and abs(b["half_width_dice"] - 0.01) < 1e-12


def test_keep_rule_thresholds():
    hw = 0.01
    assert keep_decision(0.83, 0.81, hw) == "kept"          # gain 0.02 > hw
    assert keep_decision(0.815, 0.81, hw) == "confirm"       # 0 < gain <= 2hw
    assert keep_decision(0.81, 0.81, hw) == "reverted"       # tie
    assert keep_decision(0.80, 0.81, hw) == "reverted"       # worse
    assert keep_decision(0.8101, 0.81, hw) == "confirm"      # tiny positive gain asks for a second seed


def test_unchanged_recipe_is_not_kept():
    # degeneracy: same score as incumbent must never be kept
    assert keep_decision(0.8123, 0.8123, 0.004) != "kept"


def test_ledger_append_and_numbering(tmp_path):
    p = tmp_path / "results.tsv"
    row = {c: "" for c in COLUMNS}
    row.update(exp=next_exp(p), arm="agent", commit="abc", seed=1, val_dice_mean=0.8,
               description="first\tline\nbreak", verdict="baseline", mlflow_run_id="x")
    assert append_row(p, row) == 1
    row2 = dict(row, exp=next_exp(p), description="second")
    assert append_row(p, row2) == 2
    rows = read_rows(p)
    assert [r["exp"] for r in rows] == ["1", "2"]
    assert "\t" not in rows[0]["description"] and "\n" not in rows[0]["description"]


def test_missing_column_is_refused(tmp_path):
    with pytest.raises(ValueError):
        append_row(tmp_path / "r.tsv", {"exp": 1})
