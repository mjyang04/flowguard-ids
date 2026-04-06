from __future__ import annotations

from typing import List

from .cnn_bilstm_se import CNNBiLSTMSE


class CNNBiLSTMAttention(CNNBiLSTMSE):
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
        use_se: bool = True,
    ):
        super().__init__(
            input_dim=input_dim,
            num_classes=num_classes,
            conv_channels=conv_channels,
            conv_kernel_sizes=conv_kernel_sizes,
            conv_pool_sizes=conv_pool_sizes,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            use_attention=True,
            use_se=use_se,
        )
