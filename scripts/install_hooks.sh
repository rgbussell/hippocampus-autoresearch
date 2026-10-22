#!/usr/bin/env bash
# Install the pre-commit hook: hygiene + fence tests must pass before any commit.
# Hooks are per-clone and untracked; run once after cloning.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
HOOK="$ROOT/.git/hooks/pre-commit"
cat > "$HOOK" <<'HOOK'
#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="${HIPPO_PYTHON:-python3}"
PYTHONPATH=src "$PY" -m pytest -q tests/test_public_hygiene.py tests/test_partition.py tests/test_fence.py >/dev/null \
  || { echo "pre-commit: hygiene or fence test failed; run: PYTHONPATH=src pytest -q tests" >&2; exit 1; }
"$PY" scripts/fence.py >/dev/null \
  || { echo "pre-commit: the hold-out fence or the scorer hash check failed" >&2; exit 1; }
HOOK
chmod +x "$HOOK"
echo "installed $HOOK"
