"""recipe.py: THE ONLY FILE THE AGENT EDITS.

Everything that shapes training lives here: normalization, patch sampling,
loss, optimizer, schedule, augmentation, deep supervision, network size,
batch size. Everything else (the split, the budget, inference, the scorer,
the log) lives outside and is fixed.

Two things are declared:

  RECIPE  the current values (this is what the agent changes), and
  KNOBS   the admissible space per knob (choices or a numeric range). The
          random-edit climber draws from KNOBS; the harness refuses a RECIPE
          whose values fall outside KNOBS, and refuses a network above
          PARAM_CAP parameters.

Baseline (deliberately plain): min-max normalization, uniform random 32^3
patches, plain cross-entropy, Adam at a flat 1e-3, no augmentation, no deep
supervision, batch 2, a small U-Net. Expected near 0.80 mean Dice.
"""
from __future__ import annotations

PARAM_CAP = 8_000_000  # parameters; a bigger network is refused (logged as reverted)

RECIPE = {
    # intensity normalization applied per volume before patching
    "normalization": "minmax",          # none | minmax | zscore | percentile_clip
    # patch geometry and sampling
    "patch": (32, 32, 32),              # fits the ~35x50x35 volumes with padding
    "fg_oversample": 0.0,               # fraction of patches centred on a foreground voxel
    # loss
    "loss": "ce",                       # ce | dice | dice_ce
    "class_weights": None,              # None or [w_bg, w_ant, w_post]
    # optimizer and schedule
    "optimizer": "adam",                # adam | sgd_nesterov | adamw
    "lr": 1e-3,
    "momentum": 0.99,                   # sgd only
    "weight_decay": 0.0,
    "schedule": "flat",                 # flat | poly | cosine
    # augmentation (applied to the training patch)
    "aug_flip": False,
    "aug_rot90": False,
    "aug_scale_intensity": 0.0,         # +/- fraction, 0 = off
    "aug_gaussian_noise": 0.0,          # std, 0 = off
    "aug_affine": False,                # small random affine
    # network
    "channels": (16, 32, 64),           # U-Net channels per level
    "strides": (2, 2),
    "num_res_units": 0,
    "norm": "batch",                    # batch | instance
    "deep_supervision": False,
    "dropout": 0.0,
    # batching
    "batch_size": 2,
}

KNOBS = {
    "normalization": {"choices": ["none", "minmax", "zscore", "percentile_clip"]},
    "patch": {"choices": [(32, 32, 32), (32, 48, 32), (40, 56, 40), (48, 64, 48)]},
    "fg_oversample": {"range": (0.0, 1.0)},
    "loss": {"choices": ["ce", "dice", "dice_ce"]},
    "class_weights": {"choices": [None, [0.5, 1.0, 1.0], [0.2, 1.0, 1.0]]},
    "optimizer": {"choices": ["adam", "sgd_nesterov", "adamw"]},
    "lr": {"range": (1e-4, 3e-2), "log": True},
    "momentum": {"choices": [0.9, 0.95, 0.99]},
    "weight_decay": {"choices": [0.0, 1e-5, 3e-5, 1e-4]},
    "schedule": {"choices": ["flat", "poly", "cosine"]},
    "aug_flip": {"choices": [False, True]},
    "aug_rot90": {"choices": [False, True]},
    "aug_scale_intensity": {"choices": [0.0, 0.1, 0.25]},
    "aug_gaussian_noise": {"choices": [0.0, 0.05, 0.1]},
    "aug_affine": {"choices": [False, True]},
    "channels": {"choices": [(16, 32, 64), (32, 64, 128), (32, 64, 128, 256), (16, 32, 64, 128)]},
    "strides": {"choices": [(2, 2), (2, 2, 2)]},
    "num_res_units": {"choices": [0, 1, 2]},
    "norm": {"choices": ["batch", "instance"]},
    "deep_supervision": {"choices": [False, True]},
    "dropout": {"choices": [0.0, 0.1, 0.2]},
    "batch_size": {"choices": [1, 2, 4, 8]},
}
