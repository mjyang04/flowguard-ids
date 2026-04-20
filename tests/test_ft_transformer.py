"""Tests for the FT-Transformer baseline (Gorishniy et al. 2021)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nids.models.ft_transformer import FTTransformer  # noqa: E402


def test_ft_transformer_binary_forward_shape():
    model = FTTransformer(
        input_dim=55,
        num_classes=2,
        d_token=64,
        n_blocks=3,
        attention_heads=4,
        ffn_factor=2.0,
        dropout=0.1,
    )
    x = torch.randn(8, 55)
    out = model(x)
    assert out.shape == (8,)


def test_ft_transformer_multiclass_forward_shape():
    model = FTTransformer(
        input_dim=55,
        num_classes=15,
        d_token=64,
        n_blocks=3,
        attention_heads=4,
        ffn_factor=2.0,
        dropout=0.1,
    )
    x = torch.randn(4, 55)
    out = model(x)
    assert out.shape == (4, 15)


def test_ft_transformer_param_count_reasonable():
    model = FTTransformer(
        input_dim=55,
        num_classes=15,
        d_token=64,
        n_blocks=3,
        attention_heads=4,
        ffn_factor=2.0,
        dropout=0.1,
    )
    params = sum(p.numel() for p in model.parameters())
    # Expect roughly 0.15M-2M params for this configuration.
    assert 150_000 < params < 2_000_000, f"unexpected param count {params}"


def test_ft_transformer_built_through_registry():
    from nids.config import ModelConfig
    from nids.models.registry import create_model

    model_cfg = ModelConfig(
        name="ft_transformer", input_dim=55, num_classes=15, dropout=0.1
    )
    model = create_model(model_cfg)
    out = model(torch.randn(2, 55))
    assert out.shape == (2, 15)


def test_ft_transformer_backward_pass():
    """A single backward pass should populate gradients on all parameters."""
    model = FTTransformer(
        input_dim=16,
        num_classes=5,
        d_token=32,
        n_blocks=2,
        attention_heads=4,
        ffn_factor=2.0,
        dropout=0.1,
    )
    x = torch.randn(8, 16)
    y = torch.randint(0, 5, (8,))
    logits = model(x)
    assert logits.shape == (8, 5)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"missing gradients on: {missing}"
