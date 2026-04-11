from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from .base import BaseNIDSModel


class ConvBlock(nn.Module):
    """Conv -> BN -> ReLU -> MaxPool -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SqueezeExcitation(nn.Module):
    """SE channel attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        squeeze = x.mean(dim=-1)
        excitation = torch.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        return x * excitation.unsqueeze(-1)


class AttentionPooling(nn.Module):
    """Additive attention pooling over sequence dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, hidden)
        weights = torch.softmax(self.score(x), dim=0)
        return (weights * x).sum(dim=0)


class CNNBiLSTMSE(BaseNIDSModel):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: List[int],
        conv_kernel_sizes: List[int] | None = None,
        conv_pool_sizes: List[int] | None = None,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = False,
        use_se: bool = True,
        se_reduction: int = 16,
    ):
        super().__init__()
        conv_kernel_sizes = conv_kernel_sizes or [3] * len(conv_channels)
        conv_pool_sizes = conv_pool_sizes or [2] * len(conv_channels)
        if not (len(conv_channels) == len(conv_kernel_sizes) == len(conv_pool_sizes)):
            raise ValueError("conv_channels, conv_kernel_sizes, conv_pool_sizes must have same length")

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv_channels = conv_channels
        self.conv_kernel_sizes = conv_kernel_sizes
        self.conv_pool_sizes = conv_pool_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.dropout_p = dropout
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_se = use_se
        self.se_reduction = se_reduction

        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels, kernel_size, pool_size in zip(
            conv_channels, conv_kernel_sizes, conv_pool_sizes
        ):
            layers.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    dropout=dropout,
                )
            )
            if use_se:
                layers.append(SqueezeExcitation(out_channels, reduction=se_reduction))
            in_channels = out_channels
        self.feature_extractor = nn.Sequential(*layers)

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        self.attention = AttentionPooling(lstm_output_dim) if use_attention else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes if num_classes > 2 else 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, feature_dim)
        x = inputs.unsqueeze(1)  # (batch, 1, seq_len)
        x = self.feature_extractor(x)  # (batch, channels, seq_len')
        x = x.permute(2, 0, 1)  # (seq_len', batch, channels)
        lstm_out, _ = self.lstm(x)

        if self.attention is not None:
            pooled = self.attention(lstm_out)
        else:
            pooled = lstm_out.mean(dim=0)

        logits = self.classifier(pooled)
        return logits.squeeze(-1) if self.num_classes == 2 else logits

    def get_metadata(self) -> dict:
        return {
            "model": "cnn_bilstm_se",
            "input_dim": self.input_dim,
            "num_classes": self.num_classes,
            "conv_channels": self.conv_channels,
            "conv_kernel_sizes": self.conv_kernel_sizes,
            "conv_pool_sizes": self.conv_pool_sizes,
            "lstm_hidden_size": self.lstm_hidden_size,
            "lstm_num_layers": self.lstm_num_layers,
            "dropout": self.dropout_p,
            "bidirectional": self.bidirectional,
            "use_attention": self.use_attention,
            "use_se": self.use_se,
            "se_reduction": self.se_reduction,
        }
