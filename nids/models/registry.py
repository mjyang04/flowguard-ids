from __future__ import annotations

from nids.config import ModelConfig
from nids.models.cnn_bilstm import CNNBiLSTM
from nids.models.cnn_bilstm_at import CNNBiLSTMAT
from nids.models.cnn_bilstm_attention import CNNBiLSTMAttention
from nids.models.cnn_bilstm_se import CNNBiLSTMSE
from nids.models.ft_transformer import FTTransformer


MODEL_REGISTRY = {
    "cnn_bilstm": CNNBiLSTM,
    "cnn_bilstm_se": CNNBiLSTMSE,
    "cnn_bilstm_attention": CNNBiLSTMAttention,
    "cnn_bilstm_at": CNNBiLSTMAT,
    "ft_transformer": FTTransformer,
}


def create_model(config: ModelConfig):
    key = config.name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {config.name}. Available: {list(MODEL_REGISTRY)}")
    model_cls = MODEL_REGISTRY[key]

    # Najar et al. lightweight model uses a different constructor signature
    if key == "cnn_bilstm_at":
        return model_cls(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )

    if key == "ft_transformer":
        return model_cls(
            input_dim=config.input_dim,
            num_classes=config.num_classes,
            d_token=getattr(config, "d_token", 64),
            n_blocks=getattr(config, "n_blocks", 3),
            attention_heads=getattr(config, "attention_heads", 4),
            ffn_factor=getattr(config, "ffn_factor", 2.0),
            dropout=config.dropout,
        )

    kwargs = {
        "input_dim": config.input_dim,
        "num_classes": config.num_classes,
        "conv_channels": config.conv_channels,
        "conv_kernel_sizes": config.conv_kernel_sizes,
        "conv_pool_sizes": config.conv_pool_sizes,
        "lstm_hidden_size": config.lstm_hidden_size,
        "lstm_num_layers": config.lstm_num_layers,
        "dropout": config.dropout,
        "bidirectional": config.bidirectional,
    }
    if key == "cnn_bilstm_se":
        kwargs["use_attention"] = config.use_attention
        kwargs["use_se"] = config.use_se
        kwargs["se_reduction"] = getattr(config, "se_reduction", 16)
    elif key == "cnn_bilstm":
        kwargs["use_attention"] = config.use_attention
    elif key == "cnn_bilstm_attention":
        kwargs["use_se"] = config.use_se
        kwargs["se_reduction"] = getattr(config, "se_reduction", 16)
    return model_cls(**kwargs)
