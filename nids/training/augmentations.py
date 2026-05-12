"""Augmentations used to generate "negative views" during CLAN pretraining.

Ported from https://github.com/jackwilkie/CLAN/blob/main/loss/augmentations.py
(Apache-2.0). The current thesis scope always uses CLAN's default
``UniformResample`` negative-view generator; the other augmentation modules are
kept for tests and future ablations.

All augmentation modules are stateless (``no_grad``) — they only perturb
the input tensor and return a new tensor of the same shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from nids.config import AugmentationConfig


def _aug_matrix(x: Tensor, p_feature: float, p_sample: float) -> Tensor:
    """Binary mask (1 = augment this element)."""
    feature_mask = torch.bernoulli(torch.full_like(x, p_feature))
    if p_sample < 1.0:
        sample_probs = torch.full((x.size(0),), p_sample, device=x.device)
        sample_mask = torch.bernoulli(sample_probs).unsqueeze(-1)
        if x.dim() == 3:
            sample_mask = sample_mask.unsqueeze(-1)
        feature_mask = feature_mask * sample_mask
    return feature_mask


class Jitter(nn.Module):
    """Add Gaussian noise with variance ``variance`` to a random fraction
    of features."""

    def __init__(
        self, variance: float, mean: float = 0.0, p_feature: float = 1.0, p_sample: float = 1.0
    ) -> None:
        super().__init__()
        self.variance = variance
        self.mean = mean
        self.p_feature = p_feature
        self.p_sample = p_sample

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        mask = _aug_matrix(x, self.p_feature, self.p_sample)
        noise = (self.variance ** 0.5) * torch.randn_like(x) + self.mean
        return x + noise * mask


class ZeroOutNoise(nn.Module):
    """Randomly zero out a fraction of features."""

    def __init__(self, p_feature: float, p_sample: float = 1.0) -> None:
        super().__init__()
        self.p_feature = p_feature
        self.p_sample = p_sample

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        mask = _aug_matrix(x, self.p_feature, self.p_sample)
        return x * (1.0 - mask)


class GaussianResample(nn.Module):
    """Replace selected features with freshly-drawn Gaussian samples."""

    def __init__(
        self, mean: float, variance: float, p_feature: float, p_sample: float = 1.0
    ) -> None:
        super().__init__()
        self.mean = mean
        self.variance = variance
        self.p_feature = p_feature
        self.p_sample = p_sample

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        mask = _aug_matrix(x, self.p_feature, self.p_sample)
        noise = (self.variance ** 0.5) * torch.randn_like(x) + self.mean
        return x * (1.0 - mask) + noise * mask


class UniformResample(nn.Module):
    """Replace selected features with U(-max_val, max_val) + mean — the
    CLAN paper default."""

    def __init__(
        self, max_val: float, mean: float, p_feature: float, p_sample: float = 1.0
    ) -> None:
        super().__init__()
        self.max_val = max_val
        self.mean = mean
        self.p_feature = p_feature
        self.p_sample = p_sample

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        mask = _aug_matrix(x, self.p_feature, self.p_sample)
        noise = torch.empty_like(x).uniform_(-self.max_val, self.max_val) + self.mean
        return x * (1.0 - mask) + noise * mask


class FeatureShuffle(nn.Module):
    """Randomly permute the feature axis for selected positions."""

    def __init__(self, p_feature: float, p_sample: float = 1.0) -> None:
        super().__init__()
        self.p_feature = p_feature
        self.p_sample = p_sample

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        mask = _aug_matrix(x, self.p_feature, self.p_sample)
        perm = torch.randperm(x.size(-1), device=x.device)
        x_shuffled = x[:, perm]
        return x * (1.0 - mask) + x_shuffled * mask


def make_augmentation(cfg: AugmentationConfig) -> nn.Module:
    """Build CLAN's default uniform-resample augmentation."""
    return UniformResample(
        max_val=cfg.max_val,
        mean=0.0,
        p_feature=cfg.p_feature,
        p_sample=1.0,
    )
