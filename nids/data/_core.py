"""Shared preprocessing kernel used by every dataset loader.

Both :mod:`nids.data.lycos` and :mod:`nids.data.cicids` call
:func:`prepare_splits` after reading and normalising their respective CSVs.
This keeps dataset-specific quirks (column names, label casing, zip
archives) out of the splitting / standardisation logic.

Behaviour mirrors the upstream CLAN ``get_data`` (Apache-2.0,
https://github.com/jackwilkie/CLAN/blob/main/data/load_data.py) with two
additions:

* ``drop_zero_cols`` — drop feature columns that are entirely zero across
  the dataset, matching upstream's silent ``np.argwhere(np.all(...==0))``
  step that determines the final ``input_dim``.
* the sklearn ``train_test_split`` call is deterministic via ``split_seed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    """Return type for every dataset loader.

    Field order matches the 8-tuple used in upstream CLAN scripts, so the
    port remains mechanically close to the reference implementation.
    """

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    x_zd: np.ndarray
    y_zd: np.ndarray
    class_names: list[str]
    feature_names: list[str]

    @property
    def input_dim(self) -> int:
        """Number of feature columns after preprocessing — authoritative
        value to use when building the encoder input layer."""
        return self.x_train.shape[1]

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


def prepare_splits(
    df: pd.DataFrame,
    *,
    target: str,
    drop: Iterable[str],
    class_zero: str,
    sample_thres: int,
    split_seed: int,
    test_ratio: float,
    val_ratio: float,
    anomaly_detection: bool,
    standardise: bool = True,
    drop_zero_cols: bool = True,
) -> DataSplits:
    """Turn an already-loaded DataFrame into the 8-way split used by CLAN.

    ``drop`` column matching is case-insensitive so that datasets with
    idiosyncratic column casing (e.g. CICIDS2017's leading-space
    ``" Label"``) work without aliasing config overrides.
    """
    drop_set = {c.lower() for c in drop}
    keep_cols = [c for c in df.columns if c.lower() not in drop_set]
    df = df[keep_cols]

    if target not in df.columns:
        raise KeyError(f"Target column {target!r} not in CSV; have {df.columns.tolist()}")

    y_raw = df[target].map(_coerce_label).to_numpy()
    x_df = df.drop(columns=[target])
    feature_names = x_df.columns.tolist()

    numeric_df = x_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.shape[1] != x_df.shape[1]:
        dropped = set(x_df.columns) - set(numeric_df.columns)
        raise ValueError(
            f"non-numeric feature columns encountered: {sorted(dropped)}. "
            "Add them to `drop` or convert before loading."
        )
    x = numeric_df.to_numpy(dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    if drop_zero_cols and x.shape[0] > 0:
        zero_mask = np.all(x == 0, axis=0)
        if zero_mask.any():
            x = x[:, ~zero_mask]
            feature_names = [n for n, keep in zip(feature_names, ~zero_mask) if keep]

    label_map, class_names = _build_label_map(y_raw, class_zero=class_zero)
    y = np.asarray([label_map[lbl] for lbl in y_raw], dtype=np.int64)

    x_dev, y_dev, x_zd, y_zd = _split_zero_day(x, y, sample_thres, benign_class=0)

    stratify = y_dev if len(np.unique(y_dev)) > 1 else None
    x_train_full, x_test, y_train_full, y_test = train_test_split(
        x_dev, y_dev, test_size=test_ratio, random_state=split_seed, stratify=stratify,
    )

    if val_ratio > 0.0:
        stratify_val = y_train_full if len(np.unique(y_train_full)) > 1 else None
        x_train, x_val, y_train, y_val = train_test_split(
            x_train_full, y_train_full,
            test_size=val_ratio, random_state=split_seed, stratify=stratify_val,
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
