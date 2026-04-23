"""Shape + behaviour tests for the CLAN augmentation modules."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nids.config import AugmentationConfig
from nids.training.augmentations import (
    FeatureShuffle,
    GaussianResample,
    Jitter,
    UniformResample,
    ZeroOutNoise,
    make_augmentation,
)


@pytest.mark.parametrize(
    "aug",
    [
        Jitter(variance=0.1, mean=0.0, p_feature=1.0, p_sample=1.0),
        ZeroOutNoise(p_feature=0.3, p_sample=1.0),
        GaussianResample(mean=0.0, variance=1.0, p_feature=0.2, p_sample=1.0),
        UniformResample(max_val=1.7, mean=0.0, p_feature=0.1, p_sample=1.0),
        FeatureShuffle(p_feature=0.5, p_sample=1.0),
    ],
)
def test_augmentation_preserves_shape(aug):
    x = torch.randn(16, 72)
    y = aug(x)
    assert y.shape == x.shape
    assert y.dtype == x.dtype


def test_zero_out_noise_produces_zeros_when_p_feature_1():
    aug = ZeroOutNoise(p_feature=1.0, p_sample=1.0)
    x = torch.randn(8, 12)
    y = aug(x)
    assert torch.all(y == 0)


def test_jitter_does_not_change_stats_when_variance_zero():
    aug = Jitter(variance=0.0, mean=0.0, p_feature=1.0, p_sample=1.0)
    x = torch.randn(8, 12)
    y = aug(x)
    assert torch.allclose(y, x, atol=1e-6)


def test_make_augmentation_dispatches_by_name():
    cfg = AugmentationConfig(
        name="uniform_resample", max_val=1.7, mean=0.0, p_feature=0.1, p_sample=1.0
    )
    aug = make_augmentation(cfg)
    assert isinstance(aug, UniformResample)


def test_make_augmentation_rejects_unknown_name():
    cfg = AugmentationConfig(name="bogus")
    with pytest.raises(ValueError, match="Unknown augmentation"):
        make_augmentation(cfg)


def test_augmentations_are_no_grad():
    """Augmentations must never contribute to autograd to avoid leaks."""
    aug = UniformResample(max_val=1.7, mean=0.0, p_feature=0.2, p_sample=1.0)
    x = torch.randn(4, 8, requires_grad=True)
    y = aug(x)
    assert not y.requires_grad
