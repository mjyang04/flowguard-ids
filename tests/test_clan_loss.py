"""Numerical tests for CLANLoss."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nids.training.losses import CLANLoss, clan_loss


def test_clan_loss_returns_non_negative_scalar():
    torch.manual_seed(0)
    z = torch.randn(32, 16)
    z_aug = torch.randn(32, 16)
    loss_module = CLANLoss(m=1.0, loss_alpha=0.5, distance_metric="cosine")
    loss, frac = loss_module(z, z_aug)
    assert loss.ndim == 0
    assert float(loss) >= 0.0
    assert 0.0 <= float(frac) <= 1.0


def test_clan_loss_rejects_invalid_alpha():
    with pytest.raises(ValueError, match="loss_alpha"):
        CLANLoss(loss_alpha=2.0)


def test_clan_loss_rejects_unknown_distance():
    with pytest.raises(ValueError, match="Invalid distance metric"):
        CLANLoss(distance_metric="manhattan")


def test_identical_batch_intra_distance_zero():
    """If all z are identical, the intra-class term is 0 and the loss
    reduces to the inter-class hinge term."""
    z = torch.ones(4, 8)
    z_aug = torch.zeros(4, 8)
    loss_module = CLANLoss(m=1.0, loss_alpha=0.5, distance_metric="cosine")
    loss, _ = loss_module(z, z_aug)
    # z_aug is all zeros: cosine(z, z_aug) is 0 after F.normalize handles the
    # zero vector as 0, so distance = (1 - 0) / 2 = 0.5. Hinge = 1 - 0.5 = 0.5.
    # alpha=0.5 -> loss = 0.5 * 0 + 0.5 * 0.5 = 0.25.
    assert float(loss) == pytest.approx(0.25, abs=1e-5)


def test_functional_and_module_agree():
    torch.manual_seed(42)
    z = torch.randn(16, 8)
    z_aug = torch.randn(16, 8)
    loss_module = CLANLoss(m=0.5, loss_alpha=0.3)
    l_mod, _ = loss_module(z, z_aug)
    from nids.training.distance import cosdist
    l_fn, _ = clan_loss(z, z_aug, m=0.5, alpha=0.3, distance_metric=cosdist)
    assert float(l_mod) == pytest.approx(float(l_fn), abs=1e-6)
