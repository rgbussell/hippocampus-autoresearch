#!/usr/bin/env bash
# Install the git hooks for this clone (hooks are per-clone and untracked).
#
#   scripts/install_hooks.sh
#
# Uses the pre-commit framework (.pre-commit-config.yaml): secrets scan,
# private-key detection, large-file guard, the repo's public-hygiene and
# fence tests, the eval/ freeze guard, and ruff lint + format. Falls back to
# a plain shell hook if pre-commit is not installed.
set -euo pipefail
ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"
PY="${HIPPO_PYTHON:-python3}"

if command -v pre-commit >/dev/null 2>&1; then
  pre-commit install --install-hooks
  pre-commit install --hook-type pre-push
  echo "installed pre-commit hooks (pre-commit + pre-push)"
  exit 0
fi

echo "pre-commit not found; installing a plain shell hook (pip install pre-commit for the full set)" >&2
HOOK="$ROOT/.git/hooks/pre-commit"
cat > "$HOOK" <<HOOK
#!/usr/bin/env bash
set -euo pipefail
cd "\$(git rev-parse --show-toplevel)"
PYTHONPATH=src "$PY" -m pytest -q tests/test_public_hygiene.py tests/test_partition.py tests/test_fence.py >/dev/null \\
  || { echo "pre-commit: hygiene or fence test failed" >&2; exit 1; }
"$PY" scripts/fence.py >/dev/null || { echo "pre-commit: the hold-out fence failed" >&2; exit 1; }
scripts/check_eval_frozen.sh
HOOK
chmod +x "$HOOK"
echo "installed $HOOK"
