"""Unit tests for CNNBiLSTMSETransformer (three-scale hybrid model)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nids.config import ModelConfig  # noqa: E402
from nids.models.cnn_bilstm_se_transformer import CNNBiLSTMSETransformer  # noqa: E402
from nids.models.registry import create_model  # noqa: E402


def _base_kwargs(num_classes: int = 2) -> dict:
    return dict(
        input_dim=55,
        num_classes=num_classes,
        conv_channels=[64, 128],
        conv_kernel_sizes=[3, 3],
        conv_pool_sizes=[2, 2],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dropout=0.1,
        bidirectional=True,
        use_attention=False,
        use_se=True,
        se_reduction=16,
        transformer_layers=2,
        transformer_heads=4,
        transformer_dim_feedforward=256,
    )


def test_forward_binary_output_shape():
    model = CNNBiLSTMSETransformer(**_base_kwargs(num_classes=2))
    model.eval()
    x = torch.randn(8, 55)
    with torch.no_grad():
        out = model(x)
    # Binary head uses a single logit and squeezes the trailing dim.
    assert out.shape == (8,)


def test_forward_multiclass_output_shape():
    model = CNNBiLSTMSETransformer(**_base_kwargs(num_classes=15))
    model.eval()
    x = torch.randn(4, 55)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 15)


def test_forward_is_deterministic_in_eval_mode():
    model = CNNBiLSTMSETransformer(**_base_kwargs(num_classes=10))
    model.eval()
    x = torch.randn(2, 55)
    with torch.no_grad():
        a = model(x)
        b = model(x)
    torch.testing.assert_close(a, b)


def test_backward_updates_gradients():
    model = CNNBiLSTMSETransformer(**_base_kwargs(num_classes=3))
    model.train()
    x = torch.randn(6, 55)
    y = torch.randint(0, 3, (6,))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    grads = [p.grad for p in model.transformer.parameters() if p.grad is not None]
    assert grads, "Transformer parameters did not receive any gradient"
    assert any(float(g.abs().sum()) > 0 for g in grads)


def test_use_attention_pooling_still_works():
    kwargs = _base_kwargs(num_classes=2)
    kwargs["use_attention"] = True
    model = CNNBiLSTMSETransformer(**kwargs)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(3, 55))
    assert out.shape == (3,)


def test_nhead_falls_back_when_not_divisible():
    # d_model = 128 (hidden=64, bidir=True). 128 % 3 != 0, so effective_heads
    # should drop to the nearest valid divisor from (4, 2, 1).
    kwargs = _base_kwargs(num_classes=2)
    kwargs["lstm_hidden_size"] = 64
    kwargs["transformer_heads"] = 3
    model = CNNBiLSTMSETransformer(**kwargs)
    assert model.effective_heads in (1, 2, 4)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 55))
    assert out.shape == (2,)


def test_get_metadata_reports_transformer_params():
    model = CNNBiLSTMSETransformer(**_base_kwargs(num_classes=2))
    meta = model.get_metadata()
    assert meta["model"] == "cnn_bilstm_se_transformer"
    assert meta["transformer_layers"] == 2
    assert meta["transformer_heads"] == 4
    assert meta["transformer_dim_feedforward"] == 256


def test_registry_creates_cnn_bilstm_se_transformer():
    cfg = ModelConfig(
        name="cnn_bilstm_se_transformer",
        input_dim=55,
        num_classes=15,
        conv_channels=[64, 128],
        conv_kernel_sizes=[3, 3],
        conv_pool_sizes=[2, 2],
        lstm_hidden_size=128,
        lstm_num_layers=2,
        dropout=0.1,
        bidirectional=True,
        use_se=True,
        se_reduction=16,
        use_attention=False,
        transformer_layers=2,
        transformer_heads=4,
        transformer_dim_feedforward=512,
    )
    model = create_model(cfg)
    assert isinstance(model, CNNBiLSTMSETransformer)
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(2, 55))
    assert out.shape == (2, 15)


def test_registry_rejects_unknown_name():
    cfg = ModelConfig(name="does_not_exist")
    with pytest.raises(ValueError, match="Unknown model"):
        create_model(cfg)
