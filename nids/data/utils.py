"""Few-shot subsampling helpers for CLAN fine-tuning.

Upstream CLAN imports ``from data.utils import sample_data`` but that
file is not checked in. This module re-implements the missing helper:

    x_train_sub, y_train_sub = sample_data(
        x_train, y_train,
        num_benign=K,
        num_mal=K,
        sample_seed=None,
    )

Semantics: take ``num_benign`` rows from class 0 and ``num_mal`` rows
distributed as evenly as possible across the other classes. Sampling is
without replacement when the pool is large enough, with replacement
otherwise.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _rng(seed: Optional[int]) -> np.random.Generator:
    """Create the local random generator used by few-shot sampling."""
    return np.random.default_rng(seed)


def _sample_without_replacement(
    indices: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample row indices, falling back to replacement when the pool is small."""
    if n <= 0 or indices.size == 0:
        return np.empty((0,), dtype=indices.dtype)
    replace = n > indices.size
    return rng.choice(indices, size=n, replace=replace)


def sample_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    num_benign: int,
    num_mal: int,
    sample_seed: Optional[int] = None,
    benign_class: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Carve a balanced benign-vs-attack subset for few-shot fine-tuning.

    Benign rows come from class 0 by default. Requested attack rows are spread
    as evenly as possible across all non-benign classes present in ``y_train``.
    """
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"x/y length mismatch: {x_train.shape[0]} vs {y_train.shape[0]}")

    rng = _rng(sample_seed)
    benign_idx = np.where(y_train == benign_class)[0]
    attack_classes = [c for c in np.unique(y_train) if c != benign_class]

    per_class: dict[int, int] = {}
    if attack_classes:
        base = num_mal // len(attack_classes)
        remainder = num_mal - base * len(attack_classes)
        for i, c in enumerate(attack_classes):
            per_class[int(c)] = base + (1 if i < remainder else 0)

    sampled_idx: list[np.ndarray] = [
        _sample_without_replacement(benign_idx, num_benign, rng),
    ]
    for c, n in per_class.items():
        class_idx = np.where(y_train == c)[0]
        sampled_idx.append(_sample_without_replacement(class_idx, n, rng))

    idx = np.concatenate(sampled_idx)
    rng.shuffle(idx)
    return x_train[idx], y_train[idx]
