# Method

This document describes the loop, the fence around it, the null arms, and the
analysis, for a reader who trains models and wants to know exactly what is
being measured. The short version is in the README. This is the long one.

## 1. The question

nnU-Net's central claim (Isensee et al., Nature Methods 2021) is that a
plain U-Net wins medical segmentation challenges when its *recipe* is right:
intensity normalization, patch sampling with foreground oversampling, Dice
plus cross-entropy, deep supervision, a fixed augmentation stack, SGD with
Nesterov momentum and a polynomial learning-rate schedule, and a patch size
chosen from the data fingerprint. On MSD Task04 Hippocampus it reports a
test-set Dice of 0.90 anterior and 0.89 posterior.

We ask a narrower question than "can an agent beat nnU-Net." We ask: starting
from a U-Net with none of those decisions made well, and given only a scalar
validation score, a fixed training budget, and one editable file, which of
those decisions does a coding agent rediscover, in what order, and does it
reach them by generalizing or by fitting the ruler? And does it do better than
a random walk over the same knobs with the same keep rule?

## 2. The data

Medical Segmentation Decathlon Task04 Hippocampus (CC BY-SA 4.0). 260
labelled T1-weighted MR volumes, each roughly 35 x 50 x 35 voxels at 1 mm,
with labels 1 (anterior) and 2 (posterior hippocampus). A further 130
unlabelled volumes form the Decathlon test set and are not used by the loop.

`prepare.py` downloads the 28.4 MB archive from the msd-for-monai mirror,
checks its size and sha256 against values pinned in the script, extracts it,
hashes every image and label, and writes `manifests/partition.json`.

### The partition

| split | n | role |
|---|---|---|
| train | 182 | the only cases a training run may read |
| val | 39 | the only cases the ruler scores during a campaign |
| hold-out | 39 | scored once, by a person, after the campaign |

Seed 1337. Stratified by image voxel-count tercile so that small and large
fields of view are represented in every split. The manifest records the
stratum and the sha256 of every case, so a reader can verify that the files
on their disk are the files the results were computed on.

`hippo.partition.assert_disjoint` raises `PartitionLeak` if any id appears in
two splits, if any split has duplicates, or if any split is empty. Every
loader calls it. `prepare.py` plants a hold-out id in the training list and
refuses to write a manifest unless the check raises. The test suite does the
same. A fence that has never been seen to fire is a fence you cannot trust.

## 3. The ruler

`eval/score.py` computes, per case, the Dice coefficient for label 1 and
label 2 separately, and their mean. The campaign metric is the mean over the
validation split of the per-case two-label mean. Dice is computed on the
full volume at native resolution after sliding-window inference; no cropping
to a region of interest, no post-processing inside the scorer.

Before it is used, the ruler passes a null suite whose outputs are committed
to `manifests/w1_ruler_nulls.json`:

| null | expected | why |
|---|---|---|
| identity (prediction = reference) | mean Dice exactly 1.0 | the top of the scale is where it should be |
| empty prediction | mean Dice exactly 0.0, never NaN | the bottom of the scale is defined |
| anterior and posterior swapped | per-class Dice 0.0 | the ruler is not label-blind |
| reference shifted by 2 voxels | strictly below identity | the ruler responds to geometry |

Then the file is hashed and the hash is recorded in the program ledger. The
`eval/` directory is made read-only, and a fence test asserts, before every
experiment, that the on-disk hash matches the ledger. A run whose scorer
hash differs is void. If the ruler ever needs to change, the change is a
logged experiment with a stated reason, and every prior score is marked as
computed under the old hash.

## 4. The editable surface

`recipe.py` is the only file the agent may change. It declares:

- intensity normalization (none, min-max, z-score per volume, percentile clip);
- patch size, and the sampler (uniform, or foreground-oversampled at a ratio);
- the loss (cross-entropy, Dice, Dice + CE, with class weights if wanted);
- the optimizer, learning rate, momentum, weight decay, and schedule;
- the augmentation list and parameters;
- deep supervision on or off;
- network width, depth and normalization layer, under a parameter cap;
- batch size.

Everything else lives outside it and is fixed: the split, the budget
enforcement, inference, scoring, and logging. The parameter cap exists
because a 16 GB GPU is the hardware of record; a recipe that exceeds the cap
or runs out of memory is logged as *reverted*, not as an error.

## 5. One experiment

`train.py` runs one experiment from a clean commit:

1. Import `recipe.py`, build the model, the sampler and the loss from it.
2. Train for a fixed wall-clock budget of 300 seconds, measured after data
   loading and the first compiled step, so that every run is comparable on
   the same hardware. The budget is enforced by the harness, not the recipe.
3. Predict the validation split with sliding-window inference.
4. Score with the frozen ruler.
5. Append one row to `results.tsv`: experiment index, arm (baseline, agent,
   random, human), git commit of `recipe.py`, seed, `val_dice_mean`,
   `val_dice_ant`, `val_dice_post`, training seconds, number of optimizer
   steps, a one-line description of the change, the keep/revert verdict, and
   the MLflow run id. One writer (`hippo.record`) appends; nothing rewrites.

Each experiment is also an MLflow run. The tracking store is a file store
under `mlruns/` in the repo (gitignored; override with `MLFLOW_TRACKING_URI`),
experiment name `hippocampus-autoresearch`. Logged: every recipe knob as a
parameter, the git commit and the scorer hash as tags, the training loss per
step, the validation Dice per class, steps completed, and the elapsed time.
`scripts/show_curves.py` renders the training curves from the store for any
run id so a kept change can be inspected, not just trusted.

## 6. The keep rule and its noise floor

An autoresearch loop that keeps every positive delta is a random walk with a
ratchet whenever the delta is smaller than run-to-run noise. So the noise
floor is measured first: the plain baseline is trained with three seeds and
the half-width of the resulting Dice range is recorded in
`manifests/w2_baseline_seed_band.json`.

The keep rule is then: keep the change if `val_dice_mean` exceeds the
incumbent's by more than the half-width; if the gain is positive but within
twice the half-width, run a second seed and keep only if the mean of both
seeds clears the threshold; otherwise `git checkout recipe.py`.

The rule has a degeneracy check: re-running the incumbent recipe with the
same seed must not be kept. If it is, the rule is measuring noise and the
campaign is void.

## 7. The null arms

Three arms share the budget, the ruler, and the keep rule:

- **Do-nothing.** The incumbent recipe, re-scored. The floor.
- **Random-edit climber.** `scripts/random_climber.py` draws a mutation from
  the same knob space the agent sees (one knob per experiment, uniformly
  from its declared range or choices), trains under the same budget, applies
  the same keep rule, and runs the same number of experiments as the agent.
  Its best is reported beside the agent's. If the agent cannot beat it, the
  headline is that it cannot.
- **The one-hour human climb.** In the course version, a student replaces
  the agent for an hour and logs to the same `results.tsv` format. It is the
  control that answers "would an informed person have found these edits
  faster."

## 8. The campaign

`program.md` is the agent's standing instruction. It is written and edited
by a person, never by the agent, and it states the objective, the single
editable file, the budget, the keep rule, the log format, and the forbidden
list. A campaign runs unattended for 6 to 8 hours, about 50 experiments,
with each experiment writing only to a fresh run directory, `recipe.py`
restored by git on every revert, and the harness refusing to start if any
file other than `recipe.py` is modified. The random climber never touches
`recipe.py`: it passes its mutated recipe to `train.py` as JSON, so the two
arms share the ledger and the GPU but not the working tree. Before the first run, the claim
and the kill condition are written to the program ledger:

> Claim: within 50 experiments the agent keeps a recipe reaching at least
> 0.87 validation mean Dice from the baseline, and at least three kept edits
> correspond to named nnU-Net components.
> Kill: the random-edit climber's best matches the agent's best within the
> noise floor, or any hold-out id appears in a log.

## 9. The close

After the campaign:

1. The kept recipe is retrained with three seeds.
2. A person runs `eval/score.py` on the hold-out, once. The hold-out mean and
   the gap to validation are recorded. A gap above 0.02 Dice is reported as
   overfitting to the ruler and blocks the claim.
3. The random-edit climber's best recipe is scored on the hold-out the same
   way.
4. Every kept edit is annotated against the nnU-Net component it corresponds
   to (normalization, foreground oversampling, Dice + CE, deep supervision,
   augmentation, schedule, optimizer, patch size), and the components the
   agent never found are listed.
5. `docs/results.html` renders the Dice-versus-experiment plot for both
   climbers from `results.tsv` only, with kept points marked.

Optionally, the kept recipe's predictions on the 130 Decathlon test volumes
are submitted to the challenge site. The hold-out is the evaluator of record;
the submission is a courtesy.

## 10. What this does and does not show

It shows a loop shape: a scalar objective, a single editable surface, a
fixed budget, a frozen evaluator the optimizer cannot touch, a measured
noise floor, a sequestered hold-out, and null arms. It shows which
components of a known-good recipe a search process reaches and which it
misses, and whether the search generalizes.

It does not show clinical performance, and it does not compare across
hardware. A 300-second budget on one GPU is not a 300-second budget on
another, and results from different machines are not placed on one axis.

## References

- Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a
  self-configuring method for deep learning-based biomedical image
  segmentation. Nature Methods 18, 203–211 (2021). arXiv:1904.08128.
- Antonelli M et al. The Medical Segmentation Decathlon. Nature
  Communications 13, 4128 (2022). Simpson AL et al. arXiv:1902.09063.
- Karpathy A. autoresearch. github.com/karpathy/autoresearch (2026). The
  edit-one-file, fixed-budget, keep-or-revert pattern this repo follows.
- Askarbekuly N et al. Autoresearch with Coding Agents: Generalizers and
  Metric-Maximizers. arXiv:2607.18064 (2026). The held-out-evaluation and
  isolation rules this repo adopts.
