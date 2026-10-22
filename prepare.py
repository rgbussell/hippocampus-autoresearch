#!/usr/bin/env python3
"""Download MSD Task04 Hippocampus, verify it, and write the frozen partition.

Fixed by design. The agent does not edit this file. Run once:

    python prepare.py            # downloads to data/, writes manifests/partition.json
    python prepare.py --check    # re-verify the archive and the manifest only

Data: Medical Segmentation Decathlon, Task04 Hippocampus (CC BY-SA 4.0).
Simpson et al., "A large annotated medical image dataset for the development
and evaluation of segmentation algorithms", arXiv:1902.09063; Antonelli et
al., Nature Communications 2022. Mirror: msd-for-monai S3 bucket.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from hippo.partition import Partition, PartitionLeak, assert_disjoint, build_partition, sha256_file  # noqa: E402

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TAR = DATA / "Task04_Hippocampus.tar"
URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task04_Hippocampus.tar"
# Pinned after the first verified download (2026-10-22). A mismatch stops everything.
TAR_SHA256 = "282d808a3e84e5a52f090d9dd4c0b0057b94a6bd51ad41569aef5ff303287771"
TAR_BYTES = 28425216
MANIFEST = ROOT / "manifests" / "partition.json"


def download() -> None:
    DATA.mkdir(exist_ok=True)
    if TAR.exists() and TAR.stat().st_size == TAR_BYTES:
        print(f"archive present: {TAR}")
        return
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, TAR)  # noqa: S310 (fixed https URL)
    print(f"downloaded {TAR.stat().st_size} bytes")


def verify_tar() -> str:
    digest = sha256_file(TAR)
    size = TAR.stat().st_size
    if size != TAR_BYTES:
        raise SystemExit(f"archive size {size} != pinned {TAR_BYTES}")
    if TAR_SHA256 != "PIN_ON_FIRST_DOWNLOAD" and digest != TAR_SHA256:
        raise SystemExit(f"archive sha256 {digest} != pinned {TAR_SHA256}")
    print(f"archive sha256 {digest}")
    return digest


def extract() -> Path:
    task = DATA / "Task04_Hippocampus"
    if (task / "imagesTr").exists():
        return task
    with tarfile.open(TAR) as tf:
        members = [m for m in tf.getmembers() if m.isfile() and "/._" not in m.name and not m.name.startswith("._")]
        tf.extractall(DATA, members=members, filter="data")
    return task


def scan(task: Path) -> tuple[dict[str, int], dict[str, str], dict[str, dict]]:
    sizes: dict[str, int] = {}
    hashes: dict[str, str] = {}
    meta: dict[str, dict] = {}
    for img in sorted((task / "imagesTr").glob("hippocampus_*.nii.gz")):
        cid = img.name.replace(".nii.gz", "")
        lab = task / "labelsTr" / img.name
        if not lab.exists():
            raise SystemExit(f"missing label for {cid}")
        im = nib.load(str(img))
        shape = tuple(int(s) for s in im.shape)
        if len(shape) == 4 and shape[-1] == 1:
            shape = shape[:3]
        sizes[cid] = int(np.prod(shape))
        hashes[cid] = sha256_file(img)
        meta[cid] = {
            "shape": shape,
            "label_sha256": sha256_file(lab),
            "zooms": [float(z) for z in im.header.get_zooms()[:3]],
        }
    return sizes, hashes, meta


def planted_leak_self_check(p: Partition) -> bool:
    """Prove the fence fires: plant a hold-out id in train and expect a raise."""
    leaked = Partition.from_json(p.to_json())
    leaked.train.append(leaked.holdout[0])
    try:
        assert_disjoint(leaked)
    except PartitionLeak:
        return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="verify only")
    args = ap.parse_args()
    if args.check:
        verify_tar()
        p = Partition.from_json(json.loads(MANIFEST.read_text()))
        assert_disjoint(p)
        print("manifest clean:", {k: len(getattr(p, k)) for k in ("train", "val", "holdout")})
        return 0
    download()
    tar_digest = verify_tar()
    task = extract()
    sizes, hashes, meta = scan(task)
    print(f"scanned {len(sizes)} labelled cases")
    p = build_partition(sizes, hashes)
    ok = planted_leak_self_check(p)
    if not ok:
        raise SystemExit("the fence did not fire on a planted leak; refusing to write a manifest")
    out = p.to_json()
    out.update(
        {
            "source": {
                "url": URL,
                "tar_sha256": tar_digest,
                "tar_bytes": TAR_BYTES,
                "license": "CC BY-SA 4.0",
                "citation": "Simpson et al. arXiv:1902.09063; Antonelli et al. Nat Commun 2022",
            },
            "labels": {"0": "background", "1": "anterior hippocampus", "2": "posterior hippocampus"},
            "cases": meta,
            "planted_leak_test_passes": ok,
        }
    )
    MANIFEST.parent.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {MANIFEST}: {out['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
