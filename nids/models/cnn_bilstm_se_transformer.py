"""CNN-BiLSTM-SE with a lightweight Transformer encoder on top.

Three-scale fusion design following the hybrid Transformer+CNN-BiLSTM
architecture reported in MDPI Sensors 2025 (doi.org/10.3390/s25092725) and
MDPI Future Internet 2024 (doi.org/10.3390/fi16120481):

    CNN blocks (+SE)   -> local packet-level patterns
    BiLSTM             -> temporal dependencies
    Transformer encoder-> global long-range attention

This module reuses ``ConvBlock`` / ``SqueezeExcitation`` / ``AttentionPooling``
from :mod:`nids.models.cnn_bilstm_se` so the behaviour of the CNN and SE
stages is identical to the existing ``cnn_bilstm_se`` model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import BaseNIDSModel
from .cnn_bilstm_se import AttentionPooling, ConvBlock, SqueezeExcitation


class CNNBiLSTMSETransformer(BaseNIDSModel):
    """CNN-BiLSTM-SE trunk with a small Transformer encoder on the sequence."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: list[int],
        conv_kernel_sizes: list[int] | None = None,
        conv_pool_sizes: list[int] | None = None,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = False,
        use_se: bool = True,
        se_reduction: int = 16,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dim_feedforward: int = 512,
    ):
        super().__init__()
        conv_kernel_sizes = conv_kernel_sizes or [3] * len(conv_channels)
        conv_pool_sizes = conv_pool_sizes or [2] * len(conv_channels)
        if not (len(conv_channels) == len(conv_kernel_sizes) == len(conv_pool_sizes)):
            raise ValueError(
                "conv_channels, conv_kernel_sizes, conv_pool_sizes must have same length"
            )

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
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dim_feedforward = transformer_dim_feedforward

        # --- CNN + SE trunk ----------------------------------------------------
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

        # --- BiLSTM -----------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # --- Transformer encoder ---------------------------------------------
        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)
        # Ensure nhead divides d_model; fall back to nearest divisor if misconfigured.
        effective_heads = max(1, transformer_heads)
        if lstm_output_dim % effective_heads != 0:
            for candidate in (effective_heads, 4, 2, 1):
                if lstm_output_dim % candidate == 0:
                    effective_heads = candidate
                    break
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=lstm_output_dim,
            nhead=effective_heads,
            dim_feedforward=transformer_dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=max(1, transformer_layers)
        )
        self.effective_heads = effective_heads

        # --- Head ------------------------------------------------------------
        self.attention = AttentionPooling(lstm_output_dim) if use_attention else None
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes if num_classes > 2 else 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (batch, feature_dim) -> (batch, 1, feature_dim)
        x = inputs.unsqueeze(1)
        x = self.feature_extractor(x)  # (batch, channels, seq_len')
        x = x.permute(2, 0, 1)  # (seq_len', batch, channels)
        lstm_out, _ = self.lstm(x)  # (seq_len', batch, lstm_output_dim)

        # Transformer wants batch_first
        batch_first = lstm_out.permute(1, 0, 2)  # (batch, seq_len', d_model)
        attended = self.transformer(batch_first)
        attended_seq_first = attended.permute(1, 0, 2)  # back to (seq_len', batch, d_model)

        if self.attention is not None:
            pooled = self.attention(attended_seq_first)
        else:
            pooled = attended_seq_first.mean(dim=0)

        logits = self.classifier(pooled)
        return logits.squeeze(-1) if self.num_classes == 2 else logits

    def get_metadata(self) -> dict:
        return {
            "model": "cnn_bilstm_se_transformer",
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
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.effective_heads,
            "transformer_dim_feedforward": self.transformer_dim_feedforward,
        }
