"""Experiment configuration for CLAN reproduction on Lycos2017.

Minimal, immutable dataclasses loaded from YAML. Every tunable hyperparameter
lives in ``configs/default.yaml``; never hardcode them in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    data_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    dataset: str = "lycos2017"
    csv_path: str = "data/raw/lycos.csv"
    batch_size: int = 256
    num_workers: int = 0
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    scaler_type: str = "minmax"
    benign_label: str = "BENIGN"
    random_state: int = 42


@dataclass(frozen=True)
class ModelConfig:
    """CLDNN encoder hyperparameters (CLAN default)."""

    name: str = "cldnn"
    input_dim: int = 78
    embedding_dim: int = 128
    conv_channels: tuple[int, ...] = (64, 128)
    conv_kernel_sizes: tuple[int, ...] = (3, 3)
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    bidirectional: bool = True
    dropout: float = 0.2
    l2_normalize: bool = True


@dataclass(frozen=True)
class LossConfig:
    """Loss configuration. ``name`` selects which SSL loss drives training."""

    name: str = "clan"  # clan | simclr | barlow_twins | byol | vicreg | simsiam | conflow | sscl_ids
    margin: float = 1.0
    temperature: float = 0.5


@dataclass(frozen=True)
class AugmentationConfig:
    """Augmentation strategy used to produce negative samples in CLAN."""

    name: str = "gaussian_noise"
    noise_std: float = 0.1
    feature_dropout_prob: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    use_scheduler: bool = True
    scheduler: str = "cosine"
    use_early_stopping: bool = True
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    amp: bool = True
    eval_metric: str = "roc_auc"


@dataclass(frozen=True)
class FinetuneConfig:
    """Few-shot fine-tune on labelled samples for multiclass evaluation."""

    shots_per_class: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)
    num_epochs: int = 20
    learning_rate: float = 1e-3
    head_hidden_dim: int = 64


@dataclass(frozen=True)
class RuntimeConfig:
    output_dir: str = "artifacts"
    seed: int = 42
    device: str = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    finetune: FinetuneConfig = field(default_factory=FinetuneConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _from_dict(cls: type[Any], data: dict[str, Any]) -> Any:
    kwargs: dict[str, Any] = {}
    for field_def in cls.__dataclass_fields__.values():  # type: ignore[attr-defined]
        name = field_def.name
        if name not in data:
            continue
        value = data[name]
        field_type = field_def.type
        if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _from_dict(field_type, value)
        elif isinstance(value, list) and _is_tuple_field(field_def):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _is_tuple_field(field_def: Any) -> bool:
    annotation = field_def.type
    return isinstance(annotation, str) and annotation.startswith("tuple")


def load_config(config_path: str | Path | None = None) -> ExperimentConfig:
    if config_path is None:
        return ExperimentConfig()

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    return ExperimentConfig(
        data=_from_dict(DataConfig, raw.get("data", {})),
        model=_from_dict(ModelConfig, raw.get("model", {})),
        loss=_from_dict(LossConfig, raw.get("loss", {})),
        augmentation=_from_dict(AugmentationConfig, raw.get("augmentation", {})),
        training=_from_dict(TrainingConfig, raw.get("training", {})),
        finetune=_from_dict(FinetuneConfig, raw.get("finetune", {})),
        runtime=_from_dict(RuntimeConfig, raw.get("runtime", {})),
    )


def save_config(config: ExperimentConfig, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_dict(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    payload = {
        "data": _to_dict(config.data),
        "model": _to_dict(config.model),
        "loss": _to_dict(config.loss),
        "augmentation": _to_dict(config.augmentation),
        "training": _to_dict(config.training),
        "finetune": _to_dict(config.finetune),
        "runtime": _to_dict(config.runtime),
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
