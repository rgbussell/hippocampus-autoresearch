#!/usr/bin/env bash
# One agent experiment (protocol v3): train the current recipe.py with every
# seed in HIPPO_SEEDS, score each, decide on the mean, record one row per seed,
# and keep (commit recipe.py) or revert (git checkout recipe.py).
#
#   scripts/climb_step.sh "<one-line description>"
#
# Env: HIPPO_BUDGET (default: the band's budget_s), HIPPO_DEVICE (default cuda),
#      HIPPO_PYTHON (default python3), HIPPO_ARM (default agent),
#      HIPPO_SEEDS (default "1337 2027"; at least two),
#      HIPPO_BAND (default manifests/w4_baseline_seed_band.json).
# The decision lives in scripts/decide.py; this script only sequences it.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
DESC="${1:?one-line description required}"
PY="${HIPPO_PYTHON:-python3}"
# the pre-commit hooks call python3: make it the training interpreter, so the
# per-step commits also work under systemd (run-compute), where PATH is bare
export PATH="$(dirname "$(command -v "$PY")"):$PATH"
DEVICE="${HIPPO_DEVICE:-cuda}"
ARM="${HIPPO_ARM:-agent}"
BAND="${HIPPO_BAND:-manifests/w4_baseline_seed_band.json}"
SEEDS="${HIPPO_SEEDS:-1337 2027}"

[ -f "$BAND" ] || { echo "no noise floor at $BAND; run scripts/seed_band.py first" >&2; exit 2; }
BUDGET="${HIPPO_BUDGET:-$(jq -r .budget_s "$BAND")}"
set -- $SEEDS
[ "$#" -ge 2 ] || { echo "protocol v3 needs at least two seeds (HIPPO_SEEDS)" >&2; exit 2; }
# recipe.py must be the only dirty file apart from the harness-owned ledger
# (results.tsv, manifests/); anything else is a harness breach
DIRTY="$(git status --porcelain | grep -v '^?? ' | grep -v -E ' (recipe.py|results.tsv|manifests/[^ ]+)$' || true)"
[ -z "$DIRTY" ] || { echo "working tree has changes outside recipe.py:" >&2; echo "$DIRTY" >&2; exit 2; }

"$PY" scripts/fence.py >/dev/null
RESULTS=()
for SEED in "$@"; do
  RUN="runs/$(date +%Y%m%d-%H%M%S)-${ARM}-s${SEED}"
  set +e
  "$PY" train.py --seed "$SEED" --budget "$BUDGET" --device "$DEVICE" --desc "$DESC" --arm "$ARM" --no-append --run-dir "$RUN"
  RC=$?
  set -e
  if [ "$RC" -eq 3 ]; then
    "$PY" -m hippo.record --refused --arm "$ARM" --desc "$DESC" --seed "$SEED"
    git checkout -- recipe.py
    git add results.tsv && git commit -q -m "refused: $DESC" || true
    echo "VERDICT refused ($DESC)"; exit 0
  fi
  [ "$RC" -eq 0 ] || { echo "train.py failed with $RC" >&2; exit "$RC"; }
  RESULTS+=("$RUN/result.json")
done

DECISION="$("$PY" scripts/decide.py --result "${RESULTS[@]}" --band "$BAND" --arm "$ARM")"
MEAN="$(jq -s 'map(.val_dice_mean) | add / length' "${RESULTS[@]}")"
# one row per seed: the first carries the verdict, the others are marked pair
V="$DECISION"
for R in "${RESULTS[@]}"; do
  "$PY" -m hippo.record --result "$R" --verdict "$V" --arm "$ARM"
  V="pair"
done
if [ "$DECISION" = "kept" ]; then
  "$PY" scripts/decide.py --promote "${RESULTS[@]}" --arm "$ARM"
  git add recipe.py "manifests/incumbent_${ARM}.json" results.tsv
  git commit -q -m "keep: $DESC" -m "two-seed mean val_dice $MEAN (arm $ARM, seeds $SEEDS, budget ${BUDGET}s)"
else
  git checkout -- recipe.py
  git add results.tsv && git commit -q -m "revert: $DESC" -m "two-seed mean val_dice $MEAN (arm $ARM, seeds $SEEDS, budget ${BUDGET}s)" || true
fi
echo "VERDICT $DECISION ($DESC) mean=$MEAN"
