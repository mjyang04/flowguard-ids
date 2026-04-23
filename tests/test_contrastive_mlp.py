"""Shape and forward-pass tests for ContrastiveMLP."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nids.config import ModelConfig
from nids.models import ContrastiveMLP, create_model


def test_contrastive_mlp_projection_shape():
    model = ContrastiveMLP(
        d_in=72,
        neurons=[128, 128],
        d_out=64,
        n_classes=12,
        residual=True,
    )
    x = torch.randn(16, 72)
    z = model(x)
    assert z.shape == (16, 64)


def test_contrastive_mlp_finetune_head_shape():
    model = ContrastiveMLP(
        d_in=72, neurons=[64], d_out=32, n_classes=12, residual=True
    )
    x = torch.randn(8, 72)
    logits = model.forward_finetune(x)
    assert logits.shape == (8, 12)


def test_contrastive_mlp_project_to_sphere_normalises():
    model = ContrastiveMLP(
        d_in=72, neurons=[64], d_out=32, residual=True, project_to_sphere=True
    )
    x = torch.randn(4, 72)
    z = model(x)
    norms = z.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_create_model_rejects_unknown_name():
    cfg = ModelConfig(name="bogus")
    with pytest.raises(ValueError, match="Unknown model name"):
        create_model(cfg)


def test_create_model_from_config_builds_expected_shape():
    cfg = ModelConfig(
        name="contrastive_mlp",
        input_dim=72,
        neurons=(256, 256),
        embedding_dim=64,
        n_classes=12,
        residual=True,
    )
    model = create_model(cfg)
    x = torch.randn(4, 72)
    assert model(x).shape == (4, 64)
    assert model.forward_finetune(x).shape == (4, 12)
