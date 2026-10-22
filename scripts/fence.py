#!/usr/bin/env python3
"""INS-3: the hold-out fence and the scorer hash. Writes manifests/w1_fence.json.

Checks, and RAISES on failure:
  1. no hold-out id appears in results.tsv or in any file under runs/ or mlruns/
  2. eval/score.py on disk hashes to manifests/frozen.json
  3. a planted hold-out id in a scratch log is caught by the same scanner

    python scripts/fence.py            # check + write manifest
    python scripts/fence.py --freeze   # (re)write manifests/frozen.json from disk;
                                       #  only at W1 or after a logged re-freeze
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hippo.partition import load_partition, sha256_file  # noqa: E402

FROZEN = ROOT / "manifests" / "frozen.json"
SCORER = ROOT / "eval" / "score.py"
SCAN_ROOTS = [ROOT / "results.tsv", ROOT / "runs", ROOT / "mlruns"]
SKIP = {".nii", ".gz", ".pt", ".pth", ".npz", ".npy", ".png", ".bin"}


class FenceBreach(RuntimeError):
    pass


def scan_for_ids(paths: list[Path], ids: set[str]) -> list[str]:
    pat = re.compile("|".join(re.escape(i) for i in sorted(ids)))
    hits = []
    for root in paths:
        files = [root] if root.is_file() else (list(root.rglob("*")) if root.is_dir() else [])
        for f in files:
            if not f.is_file() or f.suffix.lower() in SKIP:
                continue
            try:
                text = f.read_text(errors="ignore")
            except OSError:
                continue
            m = pat.search(text)
            if m:
                hits.append(f"{f}: {m.group(0)}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    if args.freeze:
        FROZEN.write_text(json.dumps({"path": "eval/score.py", "sha256": sha256_file(SCORER)}, indent=1) + "\n")
        print(f"froze eval/score.py -> {FROZEN}")
    part = load_partition(ROOT / "manifests" / "partition.json")
    holdout = set(part.holdout)
    hits = scan_for_ids(SCAN_ROOTS, holdout)
    frozen = json.loads(FROZEN.read_text()) if FROZEN.exists() else {}
    matches = bool(frozen) and sha256_file(SCORER) == frozen.get("sha256")
    with tempfile.TemporaryDirectory() as td:
        planted = Path(td) / "run.log"
        planted.write_text(f"scored {sorted(holdout)[0]} dice 0.9\n")
        planted_raised = bool(scan_for_ids([planted], holdout))
    out = {
        "holdout_ids_in_logs": len(hits),
        "scorer_sha256_matches_ledger": matches,
        "scorer_sha256": sha256_file(SCORER),
        "planted_leak_raised": planted_raised,
        "scanned": [str(p.relative_to(ROOT)) for p in SCAN_ROOTS],
    }
    (ROOT / "manifests" / "w1_fence.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    if hits:
        raise FenceBreach("hold-out ids found in logs:\n" + "\n".join(hits[:10]))
    if not matches:
        raise FenceBreach("eval/score.py does not hash to manifests/frozen.json; the run is void")
    if not planted_raised:
        raise FenceBreach("the scanner did not catch a planted id; the fence is theater")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
