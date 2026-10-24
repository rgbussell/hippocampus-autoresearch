# hippocampus-autoresearch

**Can a coding agent rediscover nnU-Net?**

A deliberately plain 3-D U-Net starts at 0.878 mean Dice on MSD Task04
Hippocampus after 300 s on one GPU. An agent edits one file, `recipe.py`, under a fixed 5-minute
training budget. A frozen ruler scores Dice on a frozen validation split.
Git keeps the change if the score beats the incumbent by more than the
measured noise floor, else reverts it. A random-edit climber runs alongside
with the same budget and the same keep rule. A 39-case hold-out is scored
once, by a person, at the end of the campaign. The write-up maps every kept
edit onto the nnU-Net recipe component it rediscovered, and lists the ones
it never found.

Status: harness proven on the GPU, campaign not started. W2 (seven GPU
baseline runs) put the plain baseline at 0.878 mean Dice and the keep
threshold at 0.011 Dice: the first-pass gate failed its own degeneracy check
and was replaced by an out-of-sample one (see
[docs/METHOD.md](docs/METHOD.md), section 6). W3 ran the dry run: five
scripted agent edits (Dice+CE loss, z-score normalization, foreground
oversampling, mirroring, poly schedule) through the real climb step and five
random-edit steps, rows 8 to 21 of `results.tsv`. No single-knob edit
cleared the gate; four went to a confirm seed and fell back. The campaign
(W4) is pre-registered next, and its budget is the open question.

## The fence, stated once

- `eval/` holds the scorer. It is hash-pinned in `manifests/frozen.json` and in
  the program ledger, and read-only to the agent. `scripts/fence.py` re-checks
  the hash and scans every log for hold-out ids before each run; a run whose
  scorer hash differs is void. The ruler's null suite (`eval/nulls.py`) must
  pass on the real validation references before the freeze; its output is
  committed in `manifests/w1_ruler_nulls.json`.
- `manifests/partition.json` is the split. `hippo.partition.assert_disjoint`
  raises on any overlap, and `tests/test_partition.py` plants a leak to prove
  the fence can fire.
- The hold-out ids never appear in `results.tsv` or in any run log. A test
  scans for them.
- `program.md` is the human's steering file. The agent never edits it.
- Every commit passes the pre-commit hooks in `.pre-commit-config.yaml`:
  detect-secrets against a committed baseline, private-key detection, a
  large-file guard (data and weights never enter git), the repo's own
  hygiene scanner (home paths, cloud URLs, credential assignments, commit
  messages, and an untracked deny-list of private terms), the partition and
  fence tests, the hold-out fence, a guard that refuses changes to `eval/`
  without an explicit re-freeze, and ruff lint plus format. Install once
  per clone with `scripts/install_hooks.sh`.

## Data

Medical Segmentation Decathlon, Task04 Hippocampus: 260 labelled T1-weighted
MR volumes (labels: 1 anterior, 2 posterior hippocampus) plus 130 unlabelled
test volumes. Licence CC BY-SA 4.0. Not vendored; `prepare.py` downloads the
28.4 MB archive from the msd-for-monai mirror, verifies its size and sha256,
and writes the partition (train 182 / val 39 / hold-out 39, seed 1337,
stratified by image voxel-count tercile).

Citation: Simpson AL et al., *A large annotated medical image dataset for the
development and evaluation of segmentation algorithms*, arXiv:1902.09063
(2019); Antonelli M et al., *The Medical Segmentation Decathlon*, Nature
Communications 13, 4128 (2022).

## Run

```bash
pip install -e ".[dev]"          # numpy, nibabel, pytest
python prepare.py                # download, verify, partition
PYTHONPATH=src pytest -q         # the fence and hygiene tests
scripts/install_hooks.sh         # pre-commit: hygiene + fence
```

One experiment (`[train]` extra: torch, MONAI, MLflow):

```bash
pip install -e ".[train]"
python train.py --seed 1337 --budget 300 --desc "plain baseline" --verdict baseline
python scripts/show_curves.py --latest 1          # training curve from MLflow
python scripts/seed_band.py --budget 300          # W2: noise floor, keep threshold, out-of-sample degeneracy check
```

The climb, once the noise floor exists:

```bash
# agent arm: edit recipe.py, then
scripts/climb_step.sh "one-line description of the change"   # trains, scores, keeps or reverts
# random-edit null arm, same budget and keep rule, never touches recipe.py
python scripts/random_climber.py --n 50 --budget 300
```

`program.md` is the agent's standing instruction and is edited only by a
person. `results.tsv` has one row per experiment with an `arm` column
(baseline, agent, random, human), so every climb shares one ledger.

`train.py` runs the fence, validates `recipe.py` against its declared knob
space and parameter cap, trains for the budget (wall clock, after one
warm-up step), predicts the validation split with sliding-window
inference, scores with the frozen ruler, and appends one row to
`results.tsv`. Every experiment is an MLflow run in the repo-local
`mlruns/` store (override with `MLFLOW_TRACKING_URI`): all recipe knobs
as params, commit and scorer hash as tags, training loss and learning
rate per step, validation Dice per class, and `recipe.py` plus the
per-case scores as artifacts. `results.tsv` carries the run id.
A 20-second CPU smoke run of the plain baseline scores about 0.81.

The full method, for a reader who trains models: [docs/METHOD.md](docs/METHOD.md).

## Limits

This demonstrates a loop shape and the components of a training recipe on
open data. It makes no claim about any clinical system.
