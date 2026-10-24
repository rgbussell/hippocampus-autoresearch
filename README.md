# hippocampus-autoresearch

**Can a coding agent rediscover nnU-Net?**

A deliberately plain 3-D U-Net starts at 0.878 mean Dice on MSD Task04
Hippocampus after 300 s on one GPU. An agent edits one file, `recipe.py`, under a fixed training budget (two
seeds of 10 minutes each in the campaign). A frozen ruler scores Dice on a frozen validation split.
Git keeps the change if the score beats the incumbent by more than the
measured noise floor, else reverts it. A random-edit climber runs alongside
with the same budget and the same keep rule. A 39-case hold-out is scored
once, by a person, at the end of the campaign. The write-up maps every kept
edit onto the nnU-Net recipe component it rediscovered, and lists the ones
it never found.

Status: campaign pre-registered, waiting for the GPU. W2 put the plain
baseline at 0.878 mean Dice with a 0.011 gate at 300 s; W3's dry run of five
agent and five random edits all reverted under it (rows 8 to 21 of
`results.tsv`). The campaign protocol therefore moved to two seeds per
experiment at 600 s per seed, with the noise floor re-measured on two-seed
means and a written rule for moving to 900 s. Claim: at least 0.89 two-seed
validation Dice within 25 experiments per arm, with at least three kept
edits mapping to named nnU-Net components. See
[docs/METHOD.md](docs/METHOD.md) sections 6 and 8, and
[docs/AUTORESEARCH.md](docs/AUTORESEARCH.md) for what autoresearch is and
when to use it.

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
python scripts/seed_band.py --budget 600 --calibration-pairs 3   # noise floor of two-seed means, out-of-sample check
```

The climb, once the noise floor exists:

```bash
# the agent edits recipe.py, then one campaign iteration: agent experiment + random-edit control
scripts/campaign_step.sh "one-line description of the change"
# the pieces, separately
scripts/climb_step.sh "one-line description"           # two seeds, keep or revert
python scripts/random_climber.py --n 1                 # one random-edit experiment, never touches recipe.py
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
