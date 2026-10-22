"""The fence must fire on a planted leak and stay quiet on the real split."""

import json
from pathlib import Path

import pytest

from hippo.partition import SIZES, Partition, PartitionLeak, assert_disjoint, build_partition, load_partition

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "partition.json"


def _fake_cases(n=260):
    sizes = {f"hippocampus_{i:03d}": 40000 + 37 * i for i in range(n)}
    hashes = {k: "0" * 64 for k in sizes}
    return sizes, hashes


def test_build_is_deterministic_and_sized():
    sizes, hashes = _fake_cases()
    a = build_partition(sizes, hashes)
    b = build_partition(sizes, hashes)
    assert a.to_json() == b.to_json()
    assert {k: len(getattr(a, k)) for k in SIZES} == SIZES


def test_build_is_stratified():
    sizes, hashes = _fake_cases()
    p = build_partition(sizes, hashes)
    for name in ("val", "holdout"):
        terciles = [p.strata[c] for c in getattr(p, name)]
        assert {0, 1, 2} <= set(terciles), f"{name} missing a tercile"


def test_planted_leak_raises():
    sizes, hashes = _fake_cases()
    p = build_partition(sizes, hashes)
    leaked = Partition.from_json(p.to_json())
    leaked.train.append(leaked.holdout[0])  # plant the violation
    with pytest.raises(PartitionLeak):
        assert_disjoint(leaked)


def test_duplicate_raises():
    sizes, hashes = _fake_cases()
    p = build_partition(sizes, hashes)
    p.val.append(p.val[0])
    with pytest.raises(PartitionLeak):
        assert_disjoint(p)


def test_empty_split_raises():
    with pytest.raises(PartitionLeak):
        assert_disjoint(Partition(seed=1, train=["a"], val=["b"], holdout=[]))


@pytest.mark.skipif(not MANIFEST.exists(), reason="run prepare.py first")
def test_committed_manifest_is_clean():
    p = load_partition(MANIFEST)
    d = json.loads(MANIFEST.read_text())
    assert d["n"] == SIZES
    assert len(p.sha256) == sum(SIZES.values())
    assert d["planted_leak_test_passes"] is True
