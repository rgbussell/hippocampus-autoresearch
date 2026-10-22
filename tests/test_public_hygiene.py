"""Public-readiness fence. This repo is public; nothing private may enter it.

Scans every git-tracked text file and every commit message for generic
patterns (home-directory paths, cloud credentials, cloud-storage URLs, a
committed DVC remote) plus an OPTIONAL untracked deny-list of private terms
at .hygiene-deny.txt (one regex per line; gitignored so the terms themselves
never ship). Runs in the suite and in the pre-commit hook.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DENY_FILE = ROOT / ".hygiene-deny.txt"

GENERIC = {
    "home path": r"/home/[a-z][a-z0-9_-]+/|/Users/[A-Za-z][A-Za-z0-9_-]+/",
    "removable media path": r"/media/[a-z][a-z0-9_-]+/",
    "aws access key": r"AKIA[0-9A-Z]{16}",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "cloud storage url": r"\b(gdrive|s3|gs|azure)://\S+|drive\.google\.com/\S+",
    "credential assignment": r"(client_secret|api[_-]?key|access[_-]?token|password)\s*[:=]\s*['\"][^'\"]{8,}",
}
SKIP_SUFFIX = {".png", ".gif", ".jpg", ".jpeg", ".nii", ".gz", ".tar", ".pt", ".pth", ".npz", ".npy"}


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True,
                          capture_output=True, text=True).stdout


def _patterns() -> dict[str, re.Pattern]:
    pats = {k: re.compile(v) for k, v in GENERIC.items()}
    if DENY_FILE.exists():
        for i, line in enumerate(DENY_FILE.read_text().splitlines()):
            line = line.strip()
            if line and not line.startswith("#"):
                pats[f"deny-list line {i + 1}"] = re.compile(line, re.IGNORECASE)
    return pats


def _tracked_text_files() -> list[Path]:
    out = []
    for rel in _git("ls-files", "-z").split("\0"):
        if not rel:
            continue
        p = ROOT / rel
        if p.suffix.lower() in SKIP_SUFFIX or not p.is_file():
            continue
        out.append(p)
    return out


def test_tracked_files_are_clean():
    pats = _patterns()
    hits = []
    for p in _tracked_text_files():
        if p.name == "test_public_hygiene.py":
            continue  # the patterns themselves
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        for name, pat in pats.items():
            m = pat.search(text)
            if m:
                hits.append(f"{p.relative_to(ROOT)}: {name}: {m.group(0)[:60]!r}")
    assert not hits, "\n".join(hits)


def test_commit_messages_are_clean():
    pats = _patterns()
    if not _git("rev-list", "--all", "--count").strip().strip("0"):
        pytest.skip("no commits yet")
    log = _git("log", "--format=%H%n%B%n----")
    hits = [f"{name}: {m.group(0)[:60]!r}" for name, pat in pats.items() if (m := pat.search(log))]
    assert not hits, "\n".join(hits)


def test_no_dvc_remote_committed():
    tracked = _git("ls-files").split()
    assert ".dvc/config.local" not in tracked
    if ".dvc/config" in tracked:
        cfg = (ROOT / ".dvc" / "config").read_text()
        assert "remote" not in cfg, "a DVC remote is committed; it belongs in .dvc/config.local"


def test_data_not_vendored():
    tracked = _git("ls-files").split()
    big = [t for t in tracked if t.endswith((".nii.gz", ".nii", ".tar", ".pt", ".pth"))]
    assert not big, f"data or weights are tracked: {big[:5]}"


def test_fence_can_fire(tmp_path):
    """A planted hit must be caught by the same scanner, or the test is theater."""
    pats = _patterns()
    planted = "export AWS_KEY=AKIAABCDEFGHIJKLMNOP"
    assert any(p.search(planted) for p in pats.values())
