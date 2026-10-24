#!/usr/bin/env bash
# One agent experiment: train the current recipe.py, score, decide, record,
# and keep (commit recipe.py) or revert (git checkout recipe.py).
#
#   scripts/climb_step.sh "<one-line description>"
#
# Env: HIPPO_BUDGET (default 300), HIPPO_DEVICE (default cuda),
#      HIPPO_PYTHON (default python3), HIPPO_ARM (default agent),
#      HIPPO_BAND (default manifests/w2_baseline_seed_band.json).
# The decision lives in scripts/decide.py; this script only sequences it.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
DESC="${1:?one-line description required}"
PY="${HIPPO_PYTHON:-python3}"
# the pre-commit hooks call python3: make it the training interpreter, so the
# per-step commits also work under systemd (run-compute), where PATH is bare
export PATH="$(dirname "$(command -v "$PY")"):$PATH"
BUDGET="${HIPPO_BUDGET:-300}"
DEVICE="${HIPPO_DEVICE:-cuda}"
ARM="${HIPPO_ARM:-agent}"
BAND="${HIPPO_BAND:-manifests/w2_baseline_seed_band.json}"
SEED1=1337
SEED2=2027

[ -f "$BAND" ] || { echo "no noise floor at $BAND; run scripts/seed_band.py first (W2)" >&2; exit 2; }
# recipe.py must be the only dirty file apart from the harness-owned ledger
# (results.tsv, manifests/); anything else is a harness breach
DIRTY="$(git status --porcelain | grep -v '^?? ' | grep -v -E ' (recipe.py|results.tsv|manifests/[^ ]+)$' || true)"
[ -z "$DIRTY" ] || { echo "working tree has changes outside recipe.py:" >&2; echo "$DIRTY" >&2; exit 2; }

"$PY" scripts/fence.py >/dev/null
RUN="runs/$(date +%Y%m%d-%H%M%S)-${ARM}-s${SEED1}"
set +e
"$PY" train.py --seed "$SEED1" --budget "$BUDGET" --device "$DEVICE" --desc "$DESC" --arm "$ARM" --no-append --run-dir "$RUN"
RC=$?
set -e
if [ "$RC" -eq 3 ]; then
  "$PY" -m hippo.record --refused --arm "$ARM" --desc "$DESC" --seed "$SEED1"
  git checkout -- recipe.py
  git add results.tsv && git commit -q -m "refused: $DESC" || true
  echo "VERDICT refused ($DESC)"; exit 0
fi
[ "$RC" -eq 0 ] || { echo "train.py failed with $RC" >&2; exit "$RC"; }

DECISION="$("$PY" scripts/decide.py --result "$RUN/result.json" --band "$BAND" --arm "$ARM")"
if [ "$DECISION" = "confirm" ]; then
  RUN2="runs/$(date +%Y%m%d-%H%M%S)-${ARM}-s${SEED2}"
  "$PY" train.py --seed "$SEED2" --budget "$BUDGET" --device "$DEVICE" --desc "$DESC (confirm seed)" --arm "$ARM" --no-append --run-dir "$RUN2"
  DECISION="$("$PY" scripts/decide.py --result "$RUN/result.json" --confirm "$RUN2/result.json" --band "$BAND" --arm "$ARM")"
  "$PY" -m hippo.record --result "$RUN2/result.json" --verdict confirm --arm "$ARM"
fi
"$PY" -m hippo.record --result "$RUN/result.json" --verdict "$DECISION" --arm "$ARM"
if [ "$DECISION" = "kept" ]; then
  "$PY" scripts/decide.py --promote "$RUN/result.json" --arm "$ARM"
  git add recipe.py manifests/incumbent_${ARM}.json results.tsv
  git commit -q -m "keep: $DESC" -m "val_dice_mean $(jq -r .val_dice_mean "$RUN/result.json") (arm $ARM, seed $SEED1)"
else
  git checkout -- recipe.py
  git add results.tsv && git commit -q -m "revert: $DESC" -m "val_dice_mean $(jq -r .val_dice_mean "$RUN/result.json") (arm $ARM, seed $SEED1), $DECISION" || true
fi
echo "VERDICT $DECISION ($DESC)"
