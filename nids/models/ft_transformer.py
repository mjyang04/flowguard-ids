"""FT-Transformer for tabular intrusion detection.

Follows Gorishniy, Rubachev, Khrulkov and Babenko (2021)
"Revisiting Deep Learning Models for Tabular Data", NeurIPS.

Each continuous feature :math:`x_j` is embedded as
``t_j = b_j + x_j * w_j``  (with ``w_j``, ``b_j`` in ``R^{d_token}``),
then a learned [CLS] token is prepended. The stack of ``n_features + 1``
tokens is processed by a standard Transformer encoder. The CLS-token
representation is projected to ``num_classes`` logits (or a single logit
in the binary case for compatibility with the existing BCE pipeline).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """Per-feature linear embedding + learnable [CLS] token."""

    def __init__(self, input_dim: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, d_token))
        self.bias = nn.Parameter(torch.empty(input_dim, d_token))
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        # Kaiming init with a == sqrt(5) matches PyTorch nn.Linear defaults.
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.bias, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.cls, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  ->  tokens: (B, F + 1, d_token)
        tokens = self.bias.unsqueeze(0) + x.unsqueeze(-1) * self.weight.unsqueeze(0)
        cls = self.cls.expand(x.shape[0], -1, -1)
        return torch.cat([cls, tokens], dim=1)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with GELU-activated FFN."""

    def __init__(self, d_token: int, n_heads: int, ffn_factor: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            d_token, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_token)
        hidden = int(d_token * ffn_factor)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_token),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn)
        h = self.norm2(x)
        x = x + self.drop(self.ffn(h))
        return x


class FTTransformer(nn.Module):
    """FT-Transformer tabular classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        d_token: int = 64,
        n_blocks: int = 3,
        attention_heads: int = 4,
        ffn_factor: float = 2.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.d_token = d_token
        self.n_blocks = n_blocks
        self.attention_heads = attention_heads
        self.ffn_factor = ffn_factor
        self.dropout_p = dropout

        self.tokenizer = FeatureTokenizer(input_dim, d_token)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_token, attention_heads, ffn_factor, dropout)
                for _ in range(n_blocks)
            ]
        )
        self.norm = nn.LayerNorm(d_token)
        out_dim = num_classes if num_classes > 2 else 1
        self.head = nn.Linear(d_token, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        for block in self.blocks:
            tokens = block(tokens)
        cls = self.norm(tokens[:, 0])
        logits = self.head(cls)
        return logits.squeeze(-1) if self.num_classes == 2 else logits

    def get_metadata(self) -> dict:
        return {
            "model": "ft_transformer",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "d_token": self.d_token,
            "n_blocks": self.n_blocks,
            "attention_heads": self.attention_heads,
            "ffn_factor": self.ffn_factor,
            "dropout": self.dropout_p,
        }
