"""INS-3: the scanner catches a planted hold-out id; the scorer hash is pinned."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from fence import scan_for_ids  # noqa: E402
from hippo.partition import sha256_file  # noqa: E402


def test_planted_id_is_caught(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("case hippocampus_123 dice 0.91\n")
    assert scan_for_ids([log], {"hippocampus_123"})


def test_clean_log_is_clean(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("case hippocampus_001 dice 0.91\n")
    assert not scan_for_ids([log], {"hippocampus_123"})


def test_scorer_hash_matches_frozen_manifest():
    frozen = ROOT / "manifests" / "frozen.json"
    if not frozen.exists():
        import pytest
        pytest.skip("not frozen yet")
    d = json.loads(frozen.read_text())
    assert sha256_file(ROOT / d["path"]) == d["sha256"]


def test_results_tsv_has_no_holdout_ids():
    part = json.loads((ROOT / "manifests" / "partition.json").read_text())
    hits = scan_for_ids([ROOT / "results.tsv", ROOT / "runs"], set(part["holdout"]))
    assert not hits, hits
