"""The frozen partition: train / val / hold-out, and the check that RAISES.

Design (ledger W0): 260 labelled MSD Task04 cases, seed 1337, stratified by
image-voxel-count tercile, sizes 182 / 39 / 39. The hold-out is never read
by the loop; `assert_disjoint` is the fence that proves it, and the test
suite plants a violation to prove the fence can fire.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SEED = 1337
SIZES = {"train": 182, "val": 39, "holdout": 39}
SPLITS = ("train", "val", "holdout")


class PartitionLeak(RuntimeError):
    """A case id appears in more than one split, or a split is missing."""


@dataclass
class Partition:
    seed: int
    train: list[str] = field(default_factory=list)
    val: list[str] = field(default_factory=list)
    holdout: list[str] = field(default_factory=list)
    sha256: dict[str, str] = field(default_factory=dict)  # case id -> image sha256
    strata: dict[str, int] = field(default_factory=dict)  # case id -> tercile 0..2

    def split(self, name: str) -> list[str]:
        return list(getattr(self, name))

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "stratify": "image voxel-count tercile",
            "n": {k: len(getattr(self, k)) for k in SPLITS},
            "train": self.train,
            "val": self.val,
            "holdout": self.holdout,
            "strata": self.strata,
            "sha256": self.sha256,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Partition":
        return cls(
            seed=int(d["seed"]),
            train=list(d["train"]),
            val=list(d["val"]),
            holdout=list(d["holdout"]),
            sha256=dict(d.get("sha256", {})),
            strata={k: int(v) for k, v in d.get("strata", {}).items()},
        )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_disjoint(p: Partition) -> None:
    """Raise PartitionLeak unless the three splits are pairwise disjoint,
    non-empty, and free of duplicates. This is the fence; it is called by
    every loader and by the fence test."""
    seen: dict[str, str] = {}
    for name in SPLITS:
        ids = getattr(p, name)
        if not ids:
            raise PartitionLeak(f"split {name!r} is empty")
        if len(set(ids)) != len(ids):
            raise PartitionLeak(f"split {name!r} has duplicate ids")
        for cid in ids:
            if cid in seen:
                raise PartitionLeak(f"{cid} is in both {seen[cid]!r} and {name!r}")
            seen[cid] = name


def build_partition(case_sizes: dict[str, int], image_hashes: dict[str, str],
                    seed: int = SEED, sizes: dict[str, int] = SIZES) -> Partition:
    """Stratified split by voxel-count tercile. Deterministic for a seed.

    case_sizes: case id -> number of image voxels.
    """
    ids = sorted(case_sizes)
    if len(ids) != sum(sizes.values()):
        raise ValueError(f"expected {sum(sizes.values())} cases, found {len(ids)}")
    counts = np.array([case_sizes[c] for c in ids], dtype=np.int64)
    cut = np.quantile(counts, [1 / 3, 2 / 3])
    strata = {c: int(np.searchsorted(cut, case_sizes[c], side="right")) for c in ids}
    rng = np.random.default_rng(seed)
    p = Partition(seed=seed, sha256=dict(image_hashes), strata=strata)
    # Allocate per stratum proportionally; remainders go to train.
    quota = {k: 0 for k in SPLITS}
    for t in range(3):
        members = [c for c in ids if strata[c] == t]
        rng.shuffle(members)
        n = len(members)
        n_val = round(sizes["val"] * n / len(ids))
        n_hold = round(sizes["holdout"] * n / len(ids))
        p.val += members[:n_val]
        p.holdout += members[n_val:n_val + n_hold]
        p.train += members[n_val + n_hold:]
        quota["val"] += n_val; quota["holdout"] += n_hold; quota["train"] += n - n_val - n_hold
    # Rounding can leave the totals one off the design; move singletons from
    # train so the committed sizes are exactly the designed ones.
    for name in ("val", "holdout"):
        while len(getattr(p, name)) < sizes[name]:
            getattr(p, name).append(p.train.pop())
        while len(getattr(p, name)) > sizes[name]:
            p.train.append(getattr(p, name).pop())
    for name in SPLITS:
        getattr(p, name).sort()
    assert_disjoint(p)
    return p


def load_partition(path: Path) -> Partition:
    p = Partition.from_json(json.loads(Path(path).read_text()))
    assert_disjoint(p)
    return p
