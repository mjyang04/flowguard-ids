"""Najar et al. (2025) CNN-BiLSTM-AT lightweight baseline.

Reproduces the architecture from:
  "DDoS attack detection using CNN-BiLSTM with attention mechanism"
  Telematics and Informatics Reports 18 (2025) 100211

Key design choices from the paper:
  - 1D Conv with 16 filters, kernel_size=1, ReLU + BN
  - BiLSTM with 16 units per direction (32 total)
  - Soft attention over reshaped BiLSTM output
  - Dense(16) + Dropout(0.1) + Softmax output
  - ~5000 trainable parameters (ultra-lightweight)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseNIDSModel


class SoftAttention(nn.Module):
    """Element-wise soft attention (tanh + softmax) from Najar et al."""

    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, dim)
        weights = torch.softmax(self.score(x), dim=-1)
        return x * weights


class CNNBiLSTMAT(BaseNIDSModel):
    """CNN-BiLSTM with soft attention, following Najar et al. (2025).

    Adapted to work with the FlowGuard 55-dim feature space while
    preserving the paper's lightweight philosophy.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: int = 16,
        lstm_hidden_size: int = 16,
        dense_units: int = 16,
        dropout: float = 0.1,
        # Accept extra kwargs from registry for compatibility
        conv_kernel_sizes: list[int] | None = None,
        conv_pool_sizes: list[int] | None = None,
        lstm_num_layers: int = 1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self._conv_channels = conv_channels
        self._lstm_hidden_size = lstm_hidden_size
        self._dense_units = dense_units
        self._dropout = dropout

        lstm_out_dim = lstm_hidden_size * 2  # bidirectional

        # --- Conv1d(16, kernel=1) + BN + ReLU ---
        self.cnn = nn.Sequential(
            nn.Conv1d(1, conv_channels, kernel_size=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )

        # --- BiLSTM(16) -> 32-dim output ---
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

        # --- Soft attention ---
        self.attention = SoftAttention(lstm_out_dim)

        # --- Post-attention BN + classifier ---
        self.post_attn = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(lstm_out_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, dense_units),
            nn.ReLU(),
            nn.BatchNorm1d(dense_units),
            nn.Dropout(dropout),
            nn.Linear(dense_units, num_classes if num_classes > 2 else 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, input_dim=55)
        x = inputs.unsqueeze(1)          # (batch, 1, 55)
        x = self.cnn(x)                  # (batch, 16, 55)

        x = x.permute(2, 0, 1)          # (55, batch, 16)
        lstm_out, _ = self.lstm(x)       # (55, batch, 32)

        # Mean pool over sequence
        pooled = lstm_out.mean(dim=0)    # (batch, 32)

        # Soft attention
        attended = self.attention(pooled)  # (batch, 32)

        # Post-attention + classifier
        feat = self.post_attn(attended)   # (batch, 32)
        logits = self.classifier(feat)    # (batch, num_classes or 1)

        return logits.squeeze(-1) if self.num_classes == 2 else logits

    def get_metadata(self) -> dict:
        return {
            "model": "cnn_bilstm_at",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "conv_channels": self._conv_channels,
            "lstm_hidden_size": self._lstm_hidden_size,
            "dense_units": self._dense_units,
            "dropout": self._dropout,
        }
