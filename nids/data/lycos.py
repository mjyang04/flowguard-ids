"""Lycos2017 CSV loader.

Upstream CLAN repo imports ``data.load_data.get_data`` / ``data.utils.sample_data``
/ ``data.loaders.tabular_dl`` but those files are not checked in. This module
re-implements ``get_data`` based on the call sites in
``/Users/mj/CLAN/train_clan.py``, ``eval_clan.py``, ``finetune_clan.py``:

    x_train, y_train, x_val, y_val, x_test, y_test, x_zd, y_zd = get_data(
        data_path, target, drop, class_zero, sample_thres, split_seed,
        test_ratio, val_ratio, anomaly_detection,
    )

Behaviour mirrored from the upstream paper:

* drop the seven metadata columns (flow_id, addresses, ports, protocol, timestamp)
* map ``class_zero`` label -> 0; every other label -> a stable positive integer
* attacks with < ``sample_thres`` samples are held out as the *zero-day* split
* stratified train/test split by label; optional val carved from train
* normalise features to zero-mean / unit-std using **training** stats
* when ``anomaly_detection=True``, drop all non-benign rows from the training set
  (the CLAN pretraining regime)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    """Return object for :func:`get_data`. Field names match the 8-tuple
    order used by the upstream CLAN scripts."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_zd: np.ndarray  # zero-day held-out
    y_zd: np.ndarray
    class_names: list[str]
    feature_names: list[str]

    def as_tuple(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        return (
            self.x_train,
            self.y_train,
            self.x_val,
            self.y_val,
            self.x_test,
            self.y_test,
            self.x_zd,
            self.y_zd,
        )


def _coerce_label(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    return str(value).strip().lower()


def _build_label_map(
    labels: np.ndarray, class_zero: str
) -> tuple[dict[str, int], list[str]]:
    """Assign ``class_zero`` -> 0 and every other label -> a stable positive int."""
    unique = sorted({lbl for lbl in labels.tolist()})
    if class_zero not in unique:
        raise ValueError(
            f"class_zero={class_zero!r} not present in labels; found {unique!r}"
        )
    mapping: dict[str, int] = {class_zero: 0}
    idx = 1
    for lbl in unique:
        if lbl == class_zero:
            continue
        mapping[lbl] = idx
        idx += 1
    class_names = [class_zero] + [lbl for lbl in unique if lbl != class_zero]
    return mapping, class_names


def _split_zero_day(
    x: np.ndarray, y: np.ndarray, sample_threshold: int, benign_class: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split rare-attack rows (count < threshold) into the zero-day holdout."""
    counts = np.bincount(y)
    rare_classes = {
        c for c, n in enumerate(counts) if c != benign_class and n < sample_threshold
    }
    if not rare_classes:
        x_zd = np.empty((0, x.shape[1]), dtype=x.dtype)
        y_zd = np.empty((0,), dtype=y.dtype)
        return x, y, x_zd, y_zd

    mask_zd = np.isin(y, list(rare_classes))
    return x[~mask_zd], y[~mask_zd], x[mask_zd], y[mask_zd]


def _fit_standardiser(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0, ddof=1)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def get_data(
    data_path: str | Path,
    *,
    target: str = "label",
    drop: Iterable[str] = (
        "flow_id",
        "src_addr",
        "src_port",
        "dst_addr",
        "dst_port",
        "ip_prot",
        "timestamp",
    ),
    class_zero: str = "benign",
    sample_thres: int = 100,
    split_seed: int = 39058032,
    test_ratio: float = 0.5,
    val_ratio: float = 0.0,
    anomaly_detection: bool = True,
    standardise: bool = True,
) -> DataSplits:
    """Load the Lycos2017 CSV and produce the 8 splits consumed by CLAN scripts."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Lycos CSV not found at {path}")

    df = pd.read_csv(path)

    # Drop metadata columns that must never reach the encoder.
    drop_set = {c.lower() for c in drop}
    keep_cols = [c for c in df.columns if c.lower() not in drop_set]
    df = df[keep_cols]

    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not in CSV; have {df.columns.tolist()}")

    y_raw = df[target].map(_coerce_label).to_numpy()
    x_df = df.drop(columns=[target])
    feature_names = x_df.columns.tolist()

    # Drop non-numeric columns defensively (e.g. residual string categoricals).
    numeric_df = x_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] != x_df.shape[1]:
        dropped = set(x_df.columns) - set(numeric_df.columns)
        raise ValueError(
            f"non-numeric feature columns encountered and dropped: {sorted(dropped)}. "
            "Add them to `drop` or convert before loading."
        )
    x = numeric_df.to_numpy(dtype=np.float32)

    # Clean up inf / NaN.
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    label_map, class_names = _build_label_map(y_raw, class_zero=class_zero)
    y = np.asarray([label_map[lbl] for lbl in y_raw], dtype=np.int64)

    # Carve out zero-day rows first.
    x_dev, y_dev, x_zd, y_zd = _split_zero_day(x, y, sample_thres, benign_class=0)

    # Stratified train/test split.
    stratify = y_dev if len(np.unique(y_dev)) > 1 else None
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x_dev,
        y_dev,
        test_size=test_ratio,
        random_state=split_seed,
        stratify=stratify,
    )

    if val_ratio > 0.0:
        stratify_val = y_train_full if len(np.unique(y_train_full)) > 1 else None
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full,
            y_train_full,
            test_size=val_ratio,
            random_state=split_seed,
            stratify=stratify_val,
        )
    else:
        x_train, y_train = x_train_full, y_train_full
        x_val = np.empty((0, x.shape[1]), dtype=x.dtype)
        y_val = np.empty((0,), dtype=y.dtype)

    if anomaly_detection:
        mask_benign = y_train == 0
        x_train = x_train[mask_benign]
        y_train = y_train[mask_benign]

    if standardise:
        if x_train.size == 0:
            raise ValueError("Training split is empty after filtering; cannot standardise.")
        mean, std = _fit_standardiser(x_train)
        x_train = (x_train - mean) / std
        if x_val.size:
            x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std
        if x_zd.size:
            x_zd = (x_zd - mean) / std

    return DataSplits(
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
        x_zd=x_zd.astype(np.float32),
        y_zd=y_zd.astype(np.int64),
        class_names=class_names,
        feature_names=feature_names,
    )
