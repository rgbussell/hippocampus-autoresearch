#!/usr/bin/env bash
# Refuse a commit that changes eval/ unless HIPPO_REFREEZE=1 is set, which a
# person sets only for a logged re-freeze of the ruler (see METHOD.md s.3).
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
changed="$(git diff --cached --name-only -- eval/ || true)"
if [ -n "$changed" ] && [ "${HIPPO_REFREEZE:-0}" != "1" ]; then
  echo "eval/ is frozen; staged changes: $changed" >&2
  echo "a re-freeze is a logged experiment: set HIPPO_REFREEZE=1 and update manifests/frozen.json + the ledger" >&2
  exit 1
fi
