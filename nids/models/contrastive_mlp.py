"""ContrastiveMLP — CLAN's encoder.

Ported from https://github.com/jackwilkie/CLAN/blob/main/model/model.py
(Apache-2.0) with minor cleanups:

* replaced the in-file ``F.normalise`` typo with ``F.normalize``
* added type-safe ``create_model(cfg)`` factory
* moved ``layer_scale`` default handling to avoid a mutable default

The residual MLP consists of stacked :class:`DenseBlock` layers followed
by an optional projection head and an optional classifier probe. During
CLAN pretraining, only ``forward_proj`` is used; ``forward_finetune`` is
used during downstream multiclass fine-tune.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from nids.config import ModelConfig


class Residual(nn.Module):
    """Residual layer that resizes the skip connection via a linear layer
    when the input and output dimensions differ."""

    def __init__(
        self,
        layer: nn.Module,
        in_dim: int,
        out_dim: Optional[int] = None,
        layer_scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.layer = layer
        out_dim = out_dim or in_dim

        if layer_scale is not None:
            self.layer_scale: nn.Parameter | float = nn.Parameter(
                torch.ones(out_dim) * layer_scale
            )
        else:
            self.layer_scale = 1.0

        self.resize = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        return (self.layer(x) * self.layer_scale) + self.resize(x)


class DenseBlock(nn.Module):
    """Linear → activation → optional norm → dropout, optionally wrapped
    in a residual connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.ReLU,
        residual: bool = False,
        layer_scale: Optional[float] = None,
        layernorm: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_dim = out_dim or in_dim
        norm_layer = layernorm or nn.Identity

        block: nn.Module = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=bias),
            activation(),
            norm_layer(out_dim),
            nn.Dropout(dropout),
        )

        if residual:
            block = Residual(block, in_dim, out_dim, layer_scale=layer_scale)

        self.block = block

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


def make_mlp(
    d_in: int,
    neurons: list[int] | tuple[int, ...],
    activation: Callable[[], nn.Module] = nn.ReLU,
    dropout: float = 0.0,
    residual: bool | list[bool] = False,
    final_layer_activation: Optional[Callable[[], nn.Module]] = nn.ReLU,
    layer_scale: Optional[float] = None,
    bias: bool = True,
) -> nn.Sequential:
    """Build a residual MLP with ``len(neurons)`` hidden layers."""
    neurons_list = [d_in] + list(neurons)

    if isinstance(residual, bool):
        residual_flags = [residual] * len(neurons_list)
    else:
        residual_flags = list(residual)
        if len(residual_flags) != len(neurons_list):
            raise ValueError(
                f"length of residual ({len(residual_flags)}) must match number of "
                f"layers ({len(neurons_list)})"
            )

    layers: list[nn.Module] = []
    for i in range(len(neurons_list) - 1):
        is_final = i == len(neurons_list) - 2
        if is_final:
            act = nn.Identity if final_layer_activation is None else final_layer_activation
            block = DenseBlock(
                neurons_list[i],
                neurons_list[i + 1],
                dropout=0.0,
                activation=act,
                residual=residual_flags[i],
                layer_scale=layer_scale,
                bias=bias,
            )
        else:
            block = DenseBlock(
                neurons_list[i],
                neurons_list[i + 1],
                dropout=dropout,
                activation=activation,
                residual=residual_flags[i],
                layer_scale=layer_scale,
                bias=bias,
            )
        layers.append(block)

    return nn.Sequential(*layers)


class ContrastiveMLP(nn.Module):
    """Residual MLP encoder with a linear projection head and a linear
    classifier probe.

    During CLAN pretraining, :meth:`forward` returns projected embeddings;
    during downstream fine-tuning, :meth:`forward_finetune` returns class
    logits computed on top of the frozen (or slightly-tuned) features.
    """

    def __init__(
        self,
        d_in: int,
        neurons: list[int] | tuple[int, ...],
        d_out: Optional[int] = None,
        n_classes: Optional[int] = None,
        dropout: float = 0.0,
        residual: bool | list[bool] = False,
        project_to_sphere: bool = False,
        final_layer_activation: Optional[Callable[[], nn.Module]] = nn.ReLU,
    ) -> None:
        super().__init__()
        self.proj_to_sphere = project_to_sphere

        neurons_list = list(neurons)
        self.mlp = make_mlp(
            d_in=d_in,
            neurons=neurons_list,
            dropout=dropout,
            residual=residual,
            final_layer_activation=final_layer_activation,
        )

        last_hidden = neurons_list[-1]
        self.proj = nn.Identity() if d_out is None else nn.Linear(last_hidden, d_out)
        self.probe = nn.Identity() if n_classes is None else nn.Linear(last_hidden, n_classes)

    def forward_features(self, x: Tensor) -> Tensor:
        return self.mlp(x)

    def forward_cls(self, x: Tensor) -> Tensor:
        return self.probe(x)

    def forward_finetune(self, x: Tensor) -> Tensor:
        return self.probe(self.forward_features(x))

    def forward_proj(self, x: Tensor) -> Tensor:
        z = self.proj(x)
        if self.proj_to_sphere:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_proj(self.forward_features(x))


def create_model(cfg: ModelConfig) -> ContrastiveMLP:
    """Factory: build a :class:`ContrastiveMLP` from an
    :class:`~nids.config.ModelConfig`."""
    if cfg.name.lower() != "contrastive_mlp":
        raise ValueError(
            f"Unknown model name: {cfg.name!r}. Only 'contrastive_mlp' is supported."
        )
    return ContrastiveMLP(
        d_in=cfg.input_dim,
        neurons=cfg.neurons,
        d_out=cfg.embedding_dim,
        n_classes=cfg.n_classes,
        dropout=cfg.dropout,
        residual=cfg.residual,
        project_to_sphere=cfg.project_to_sphere,
    )
