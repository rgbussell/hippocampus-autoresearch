#!/usr/bin/env bash
# W3 dry run on the GPU of record: five scripted agent steps (one nnU-Net-direction
# knob each, through the real climb step) and five random-edit steps, then
# manifests/w3_dryrun.json. This is a harness test, not the campaign: the five
# agent edits are fixed here in advance so the dry run is reproducible.
#
#   scripts/w3_dryrun.sh            (HIPPO_PYTHON, HIPPO_BUDGET as in climb_step.sh)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HIPPO_PYTHON="${HIPPO_PYTHON:-python3}"
export HIPPO_BUDGET="${HIPPO_BUDGET:-300}"
export HIPPO_DEVICE="${HIPPO_DEVICE:-cuda}"
PY="$HIPPO_PYTHON"
export PATH="$(dirname "$(command -v "$PY")"):$PATH"   # hooks call python3; see climb_step.sh

set_knob() {  # set_knob <key> <python literal>   (edits RECIPE in recipe.py, the agent's one file)
  "$PY" - "$1" "$2" <<'PYEOF'
import re, sys
from pathlib import Path
key, value = sys.argv[1], sys.argv[2]
p = Path("recipe.py"); s = p.read_text()
pat = re.compile(r'^(\s*"' + re.escape(key) + r'":\s*)([^,]+)(,.*)$', re.M)
if not pat.search(s):
    raise SystemExit(f"knob {key} not found in RECIPE")
p.write_text(pat.sub(lambda m: m.group(1) + value + m.group(3), s, count=1))
PYEOF
}

# five single-knob agent steps, each a named nnU-Net component
AGENT_STEPS=(
  'loss|"dice_ce"|loss ce -> dice_ce (nnU-Net compound loss)'
  'normalization|"zscore"|normalization minmax -> zscore (nnU-Net MR normalization)'
  'fg_oversample|0.33|fg_oversample 0 -> 0.33 (nnU-Net foreground oversampling)'
  'aug_flip|True|aug_flip off -> on (mirroring augmentation)'
  'schedule|"poly"|schedule flat -> poly (nnU-Net poly LR decay)'
)
for step in "${AGENT_STEPS[@]}"; do
  IFS='|' read -r key value desc <<<"$step"
  set_knob "$key" "$value"
  scripts/climb_step.sh "$desc"
done

# five random-edit steps from the same knob space, same budget and keep rule
"$PY" scripts/random_climber.py --n 5 --budget "$HIPPO_BUDGET" --device "$HIPPO_DEVICE" --seed 7

"$PY" scripts/w3_manifest.py
git add manifests/w3_dryrun.json results.tsv
git commit -q -m "W3 dry run: five agent steps and five random-edit steps on the GPU of record" || true
echo "W3 DRY RUN DONE"
