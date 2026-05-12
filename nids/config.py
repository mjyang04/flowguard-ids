"""Experiment configuration for CLAN reproduction on Lycos2017.

Mirrors the hyperparameters used in the upstream CLAN repo
(https://github.com/jackwilkie/CLAN) for ``train_clan.py`` /
``eval_clan.py`` / ``finetune_clan.py``. Encoder is a **ContrastiveMLP**
(4-layer residual MLP), not a CLDNN — the paper's abstract is slightly
misleading: the CLDNN in the comparison table is a *baseline*, whereas
CLAN itself uses an MLP.

Every tunable value lives in ``configs/default.yaml``; never hardcode
hyperparameters in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "lycos2017"  # 'lycos2017' | 'cicids2017'
    # Path to the dataset: a CSV file for Lycos2017 or a directory of
    # day-split zips/CSVs for CICIDS2017.
    csv_path: str = "data/raw/lycos.csv"
    drop_cols: tuple[str, ...] = (
        "flow_id",
        "src_addr",
        "src_port",
        "dst_addr",
        "dst_port",
        "ip_prot",
        "timestamp",
    )
    sample_threshold: int = 100  # attacks with < threshold samples become zero-day held-out
    test_ratio: float = 0.5
    split_seed: int = 39058032
    batch_size: int = 2048  # lowered from 8192 for 6GB VRAM; raise on HPC


@dataclass(frozen=True)
class ModelConfig:
    """ContrastiveMLP — the encoder used by CLAN."""

    neurons: tuple[int, ...] = (1024, 1024, 1024, 1024)
    embedding_dim: int = 64
    residual: bool = True


@dataclass(frozen=True)
class LossConfig:
    """CLAN objective hyperparameters."""

    margin: float = 0.5
    loss_alpha: float = 0.5  # weight on intra-class term vs inter-class term


@dataclass(frozen=True)
class AugmentationConfig:
    """Uniform-resample augmentation strength for CLAN negative views."""

    max_val: float = 1.7
    p_feature: float = 0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Pretrain hyperparameters — mirrors CLAN train_clan.py defaults."""

    num_epochs: int = 200
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1
    lr_start: float = 1e-6
    lr_end: float = 1e-6
    amp: bool = True
    print_freq: int = 50


@dataclass(frozen=True)
class FinetuneConfig:
    """Few-shot fine-tune hyperparameters — mirrors CLAN finetune_clan.py."""

    samples_per_class: tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024)
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    batch_size: int = 64
    n_sample_seeds: int = 10


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
    """Build a frozen dataclass instance from a YAML dict.

    Resolves field types via :func:`typing.get_type_hints` so the converter
    works regardless of ``from __future__ import annotations`` (which keeps
    annotations as strings) and correctly identifies nested dataclasses /
    tuple fields.
    """
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for field_def in fields(cls):
        name = field_def.name
        if name not in data:
            continue
        value = data[name]
        resolved = hints.get(name, field_def.type)
        if isinstance(resolved, type) and hasattr(resolved, "__dataclass_fields__") and isinstance(value, dict):
            kwargs[name] = _from_dict(resolved, value)
        elif isinstance(value, list) and _is_tuple_type(resolved):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _is_tuple_type(annotation: Any) -> bool:
    if annotation is tuple:
        return True
    origin = get_origin(annotation)
    if origin is tuple:
        return True
    # String fallback for unresolved annotations (e.g. forward refs).
    if isinstance(annotation, str) and annotation.startswith("tuple"):
        return True
    return False


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
