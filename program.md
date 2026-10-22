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
4. Run `scripts/climb_step.sh "<one-line description>"`. It trains for the
   fixed budget, scores, decides, records, and either commits `recipe.py`
   (kept) or restores it (reverted). It may ask for a second seed
   (`confirm`); it runs that itself.
5. Read the printed verdict. Then go to step 1.

Do not ask for permission between experiments. Do not stop to summarize.
Do not re-run an experiment that was already reverted with the same values.
Do not try to read, score, or reason about the hold-out split.

## The keep rule

A change is kept only if its validation score beats the incumbent by more
than the measured noise floor (`manifests/w2_baseline_seed_band.json`,
`keep_threshold_dice`). Gains inside twice the floor trigger a second seed
and are kept only if the two-seed mean clears the floor. Everything else is
reverted. Ties are reverted.

## Budget

Each experiment trains for the fixed wall-clock budget in
`scripts/climb_step.sh` (300 s on the GPU of record). You cannot change it.
If a recipe runs out of GPU memory it is logged as `refused`.

## What good experiments look like

Prefer changes that a segmentation engineer would recognize as named
decisions: how intensities are normalized, how patches are sampled, which
loss, which optimizer and schedule, which augmentations, whether deeper
supervision helps, how big the patch and the network should be for this
field of view. Prefer one knob per experiment early; combine only after
single knobs have been measured. When a change is reverted, write down why
you think it lost before moving on.

## What is forbidden

- Editing anything but `recipe.py`.
- Reading `manifests/partition.json` for the hold-out list, or any file
  under `data/` by hand.
- Writing to `results.tsv` yourself. Only `scripts/climb_step.sh` writes it.
- Adding seeds, budget, or steps outside the harness.
- Stopping early because the score looks good. The campaign ends when the
  person running it ends it.
