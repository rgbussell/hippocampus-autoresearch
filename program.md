# program.md

This file is written and edited by a person. The agent reads it and never
edits it. It is the whole standing instruction for one campaign.

## Objective

Raise `val_dice_mean` on the validation split of MSD Task04 Hippocampus by
editing the training recipe. The score is produced by the frozen ruler in
`eval/`; you do not touch the ruler, the split, the budget, or this file.

## What you may edit

Exactly one file: `recipe.py`. Change the values in `RECIPE`. Every value
must stay inside the space declared in `KNOBS`, and the network must stay
under `PARAM_CAP`. You may not add knobs, import new libraries into
`train.py`, or change anything outside `recipe.py`. A recipe outside the
space is refused and logged as `refused`; that costs you an experiment.

## One experiment

1. Read `results.tsv` and the last few lines of `runs/*/result.json`.
   Know what the incumbent is and what has already been tried.
2. Decide one change. One idea per experiment. Say what you expect and why
   in the description, in one line.
3. Edit `recipe.py`.
4. Run `scripts/campaign_step.sh "<one-line description>"`. It trains your
   recipe with both seeds for the fixed budget, scores the mean, decides,
   records one row per seed, and either commits `recipe.py` (kept) or
   restores it (reverted). It then runs one random-edit experiment for the
   control arm; those rows are not yours to read.
5. Read the printed verdict. Then go to step 1.

Do not ask for permission between experiments. Do not stop to summarize.
Do not re-run an experiment that was already reverted with the same values.
Do not try to read, score, or reason about the hold-out split.

## The keep rule

Every experiment is the mean validation Dice of two fixed seeds. A change is
kept only if that mean beats the incumbent by more than the measured noise
floor (`manifests/w4_baseline_seed_band.json`, `keep_threshold_dice`, set
as twice the largest scatter between baseline pairs and checked out of
sample). Everything else is reverted. Ties are reverted. There is no second
chance for a near miss. The floor is a measurement, not a setting you may
change.

## Budget

Each seed trains for the fixed wall-clock budget recorded in the noise-floor
manifest (`budget_s`; 600 s on the GPU of record unless the pre-registered
rule moved it to 900 s), two seeds per experiment. You cannot change it. If
a recipe runs out of GPU memory it is logged as `refused`.

## What is forbidden

- Editing anything but `recipe.py`.
- Reading `manifests/partition.json` for the hold-out list, or any file
  under `data/` by hand.
- Writing to `results.tsv` yourself. Only `scripts/climb_step.sh` writes it.
- Adding seeds, budget, or steps outside the harness.
- Reading rows with `arm=random` to choose your next edit. They are the
  control arm. Read only `baseline` and `agent` rows.
- Stopping early because the score looks good. The campaign ends when the
  person running it ends it.
