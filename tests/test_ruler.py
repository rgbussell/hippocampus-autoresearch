"""INS-1 nulls on synthetic volumes (fast, no data) + the scorer's refusals."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eval.score import dice_per_case, score_split  # noqa: E402


def _ref():
    r = np.zeros((20, 24, 20), dtype=np.int16)
    r[4:10, 5:12, 6:12] = 1
    r[10:16, 5:12, 6:12] = 2
    return r


def test_identity_is_exactly_one():
    r = _ref()
    assert dice_per_case(r, r) == {"dice_ant": 1.0, "dice_post": 1.0, "dice_mean": 1.0}


def test_empty_is_exactly_zero_not_nan():
    r = _ref()
    d = dice_per_case(np.zeros_like(r), r)
    assert d["dice_mean"] == 0.0 and not np.isnan(d["dice_mean"])


def test_swapped_labels_score_zero_per_class():
    r = _ref()
    sw = np.where(r == 1, 2, np.where(r == 2, 1, 0)).astype(r.dtype)
    d = dice_per_case(sw, r)
    assert d["dice_ant"] == 0.0 and d["dice_post"] == 0.0


def test_shift_is_strictly_below_identity_and_monotone():
    r = _ref()
    prev = 1.0
    for k in (1, 2, 3):
        s = np.zeros_like(r); s[k:] = r[:-k]
        d = dice_per_case(s, r)["dice_mean"]
        assert d < prev
        prev = d


def test_both_empty_label_scores_one():
    r = np.zeros((4, 4, 4), dtype=np.int16); r[0, 0, 0] = 1
    d = dice_per_case(r, r)
    assert d["dice_post"] == 1.0 and d["dice_ant"] == 1.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice_per_case(np.zeros((2, 2, 2)), np.zeros((2, 2, 3)))


def test_holdout_is_refused_inside_the_loop(tmp_path, monkeypatch):
    monkeypatch.delenv("HIPPO_HOLDOUT_OK", raising=False)
    with pytest.raises(PermissionError):
        score_split(tmp_path, "holdout", ROOT / "manifests" / "partition.json", tmp_path)
