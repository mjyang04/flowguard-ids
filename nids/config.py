from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _default_common_features() -> list[str]:
    # 55-dimensional unified NetFlow-style feature space.
    return [
        "dst_port",
        "flow_duration",
        "total_fwd_packets",
        "total_bwd_packets",
        "total_length_of_fwd_packets",
        "total_length_of_bwd_packets",
        "fwd_packet_length_max",
        "fwd_packet_length_min",
        "fwd_packet_length_mean",
        "fwd_packet_length_std",
        "bwd_packet_length_max",
        "bwd_packet_length_min",
        "bwd_packet_length_mean",
        "bwd_packet_length_std",
        "flow_bytes_per_sec",
        "flow_packets_per_sec",
        "flow_iat_mean",
        "flow_iat_std",
        "flow_iat_max",
        "flow_iat_min",
        "fwd_iat_total",
        "fwd_iat_mean",
        "fwd_iat_std",
        "fwd_iat_max",
        "fwd_iat_min",
        "bwd_iat_total",
        "bwd_iat_mean",
        "bwd_iat_std",
        "bwd_iat_max",
        "bwd_iat_min",
        "fwd_header_length",
        "bwd_header_length",
        "fwd_packets_per_sec",
        "bwd_packets_per_sec",
        "min_packet_length",
        "max_packet_length",
        "packet_length_mean",
        "packet_length_std",
        "packet_length_variance",
        "fin_flag_count",
        "syn_flag_count",
        "rst_flag_count",
        "psh_flag_count",
        "ack_flag_count",
        "urg_flag_count",
        "down_up_ratio",
        "average_packet_size",
        "avg_fwd_segment_size",
        "avg_bwd_segment_size",
        "subflow_fwd_packets",
        "subflow_fwd_bytes",
        "subflow_bwd_packets",
        "subflow_bwd_bytes",
        "active_mean",
        "idle_mean",
    ]


def _default_cicids_renaming_map() -> dict[str, str]:
    return {
        "destination_port": "dst_port",
        "flow_duration": "flow_duration",
        "total_fwd_packets": "total_fwd_packets",
        "total_backward_packets": "total_bwd_packets",
        "total_length_of_fwd_packets": "total_length_of_fwd_packets",
        "total_length_of_bwd_packets": "total_length_of_bwd_packets",
        "fwd_packet_length_max": "fwd_packet_length_max",
        "fwd_packet_length_min": "fwd_packet_length_min",
        "fwd_packet_length_mean": "fwd_packet_length_mean",
        "fwd_packet_length_std": "fwd_packet_length_std",
        "bwd_packet_length_max": "bwd_packet_length_max",
        "bwd_packet_length_min": "bwd_packet_length_min",
        "bwd_packet_length_mean": "bwd_packet_length_mean",
        "bwd_packet_length_std": "bwd_packet_length_std",
        "flow_bytes_s": "flow_bytes_per_sec",
        "flow_packets_s": "flow_packets_per_sec",
        "flow_iat_mean": "flow_iat_mean",
        "flow_iat_std": "flow_iat_std",
        "flow_iat_max": "flow_iat_max",
        "flow_iat_min": "flow_iat_min",
        "fwd_iat_total": "fwd_iat_total",
        "fwd_iat_mean": "fwd_iat_mean",
        "fwd_iat_std": "fwd_iat_std",
        "fwd_iat_max": "fwd_iat_max",
        "fwd_iat_min": "fwd_iat_min",
        "bwd_iat_total": "bwd_iat_total",
        "bwd_iat_mean": "bwd_iat_mean",
        "bwd_iat_std": "bwd_iat_std",
        "bwd_iat_max": "bwd_iat_max",
        "bwd_iat_min": "bwd_iat_min",
        "fwd_header_length": "fwd_header_length",
        "bwd_header_length": "bwd_header_length",
        "fwd_packets_s": "fwd_packets_per_sec",
        "bwd_packets_s": "bwd_packets_per_sec",
        "min_packet_length": "min_packet_length",
        "max_packet_length": "max_packet_length",
        "packet_length_mean": "packet_length_mean",
        "packet_length_std": "packet_length_std",
        "packet_length_variance": "packet_length_variance",
        "fin_flag_count": "fin_flag_count",
        "syn_flag_count": "syn_flag_count",
        "rst_flag_count": "rst_flag_count",
        "psh_flag_count": "psh_flag_count",
        "ack_flag_count": "ack_flag_count",
        "urg_flag_count": "urg_flag_count",
        "down_up_ratio": "down_up_ratio",
        "average_packet_size": "average_packet_size",
        "avg_fwd_segment_size": "avg_fwd_segment_size",
        "avg_bwd_segment_size": "avg_bwd_segment_size",
        "subflow_fwd_packets": "subflow_fwd_packets",
        "subflow_fwd_bytes": "subflow_fwd_bytes",
        "subflow_bwd_packets": "subflow_bwd_packets",
        "subflow_bwd_bytes": "subflow_bwd_bytes",
        "active_mean": "active_mean",
        "idle_mean": "idle_mean",
    }


def _default_unsw_renaming_map() -> dict[str, str]:
    return {
        "dur": "flow_duration",
        "spkts": "total_fwd_packets",
        "dpkts": "total_bwd_packets",
        "sbytes": "total_length_of_fwd_packets",
        "dbytes": "total_length_of_bwd_packets",
        "smean": "fwd_packet_length_mean",
        "dmean": "bwd_packet_length_mean",
        "rate": "flow_packets_per_sec",
        "sinpkt": "fwd_iat_mean",
        "dinpkt": "bwd_iat_mean",
        "sjit": "fwd_iat_std",
        "djit": "bwd_iat_std",
        "ct_src_dport_ltm": "dst_port",
    }


@dataclass
class DataConfig:
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    train_dataset: str = "cicids2017"
    test_dataset: str = "unsw_nb15"
    data_percentage: float = 100.0
    batch_size: int = 128
    num_workers: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    scaler_type: str = "minmax"
    label_mode: str = "binary"
    max_rows: int | None = None
    random_state: int = 42
    stratify: bool = True


@dataclass
class ModelConfig:
    name: str = "cnn_bilstm_se"
    input_dim: int = 55
    num_classes: int = 2
    conv_channels: list[int] = field(default_factory=lambda: [64, 128])
    conv_kernel_sizes: list[int] = field(default_factory=lambda: [3, 3])
    conv_pool_sizes: list[int] = field(default_factory=lambda: [2, 2])
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    use_se: bool = True
    se_reduction: int = 16
    use_attention: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"
    use_scheduler: bool = True
    scheduler: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_learning_rate: float = 1e-6
    use_early_stopping: bool = True
    early_stopping_patience: int = 5
    early_stopping_delta: float = 1e-4
    gradient_clip: float = 1.0
    amp: bool = True
    selection_metric: str = "avg_attack_recall"
    loss_type: str = "bce"  # bce | focal
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    use_auc_loss: bool = False
    auc_loss_lambda: float = 0.1
    auc_loss_margin: float = 1.0
    auc_loss_num_neg: int = 5
    use_platt_calibration: bool = False
    use_tqdm: bool = True
    show_eval_tqdm: bool = False


@dataclass
class ShapConfig:
    n_samples: int = 2000
    background_size: int = 100
    top_k: int = 30
    top_k_choices: list[int] = field(default_factory=lambda: [20, 30, 50])
    reference_train_dataset: str = "cicids2017"
    reference_test_dataset: str = "cicids2017"
    reference_model_name: str = "cnn_bilstm_se"
    cumulative_threshold: float = 0.9


@dataclass
class AlignmentConfig:
    common_features: list[str] = field(default_factory=_default_common_features)
    cicids_renaming_map: dict[str, str] = field(default_factory=_default_cicids_renaming_map)
    unsw_renaming_map: dict[str, str] = field(default_factory=_default_unsw_renaming_map)


@dataclass
class RuntimeConfig:
    output_dir: str = "artifacts"
    seed: int = 42
    device: str = "auto"


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    shap: ShapConfig = field(default_factory=ShapConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _dataclass_from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    kwargs = {}
    for field_def in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field_def.name
        if name not in data:
            continue
        value = data[name]
        field_type = field_def.type
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _dataclass_from_dict(field_type, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(config_path: str | Path | None = None) -> ExperimentConfig:
    if config_path is None:
        return ExperimentConfig()

    path = Path(config_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    return ExperimentConfig(
        data=_dataclass_from_dict(DataConfig, raw.get("data", {})),
        model=_dataclass_from_dict(ModelConfig, raw.get("model", {})),
        training=_dataclass_from_dict(TrainingConfig, raw.get("training", {})),
        shap=_dataclass_from_dict(ShapConfig, raw.get("shap", {})),
        alignment=_dataclass_from_dict(AlignmentConfig, raw.get("alignment", {})),
        runtime=_dataclass_from_dict(RuntimeConfig, raw.get("runtime", {})),
    )


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data": config.data.__dict__,
        "model": config.model.__dict__,
        "training": config.training.__dict__,
        "shap": config.shap.__dict__,
        "alignment": {
            "common_features": config.alignment.common_features,
            "cicids_renaming_map": config.alignment.cicids_renaming_map,
            "unsw_renaming_map": config.alignment.unsw_renaming_map,
        },
        "runtime": config.runtime.__dict__,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
