"""Training-time utilities: losses, augmentations, distances, schedules,
meters, and checkpoint helpers.

Implementations ported from the upstream CLAN repo
(https://github.com/jackwilkie/CLAN, Apache-2.0) with light cleanups.
"""

from . import distance
from .augmentations import (
    FeatureShuffle,
    GaussianResample,
    Jitter,
    UniformResample,
    ZeroOutNoise,
    make_augmentation,
)
from .checkpoint import load_checkpoint, make_checkpoint
from .losses import CLANLoss, clan_loss
from .meter import AverageMeter
from .schedules import LRSchedule, Schedule, WarmupCosineSchedule

__all__ = [
    "distance",
    "CLANLoss",
    "clan_loss",
    "Jitter",
    "ZeroOutNoise",
    "GaussianResample",
    "UniformResample",
    "FeatureShuffle",
    "make_augmentation",
    "AverageMeter",
    "Schedule",
    "WarmupCosineSchedule",
    "LRSchedule",
    "make_checkpoint",
    "load_checkpoint",
]
