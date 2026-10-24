#!/usr/bin/env bash
# One campaign iteration (W4): the agent's experiment, then one random-edit
# experiment, on the same GPU, same budget, same seeds, same keep rule.
#
#   scripts/campaign_step.sh "<agent's one-line description>"
#
# The agent edits recipe.py, then calls this. HIPPO_BAND defaults to the W4
# noise floor; the budget comes from it. The random arm never touches
# recipe.py; its rows are committed here.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HIPPO_PYTHON="${HIPPO_PYTHON:-python3}"
export HIPPO_BAND="${HIPPO_BAND:-manifests/w4_baseline_seed_band.json}"
export PATH="$(dirname "$(command -v "$HIPPO_PYTHON")"):$PATH"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
scripts/climb_step.sh "$1"
"$HIPPO_PYTHON" scripts/random_climber.py --n 1 --band "$HIPPO_BAND"
N="$(awk -F'\t' '$2=="random" && ($11=="kept"||$11=="reverted"||$11=="refused")' results.tsv | wc -l)"
git add results.tsv manifests/incumbent_random.json 2>/dev/null || git add results.tsv
git commit -q -m "random-edit step $N" || true
echo "CAMPAIGN STEP DONE"
