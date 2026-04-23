"""DataLoader factory for CLAN training.

Upstream CLAN imports ``from data.loaders import tabular_dl`` but that
file is not checked in. This module re-implements the missing factory
with the same signature:

    dl = tabular_dl(
        x, y,
        batch_size=8192,
        balanced=True,
        collate_fn=None,
        drop_last=True,
        num_workers=0,
    )

``balanced=True`` uses a :class:`WeightedRandomSampler` weighted by
inverse class frequency so that minority classes are upsampled on the
fly — important for the CLAN fine-tune few-shot regime.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def _inverse_frequency_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    class_to_weight = {c: 1.0 / n for c, n in zip(classes, counts)}
    return np.asarray([class_to_weight[int(c)] for c in y], dtype=np.float64)


def tabular_dl(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int = 8192,
    balanced: bool = True,
    collate_fn: Optional[Callable[..., Any]] = None,
    drop_last: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap ``(x, y)`` into a (optionally balanced) ``DataLoader``."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y length mismatch: {x.shape[0]} vs {y.shape[0]}")

    x_tensor = torch.from_numpy(x.astype(np.float32))
    y_tensor = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(x_tensor, y_tensor)

    if balanced and len(np.unique(y)) > 1:
        weights = torch.from_numpy(_inverse_frequency_weights(y))
        sampler: Optional[WeightedRandomSampler] = WeightedRandomSampler(
            weights=weights, num_samples=len(y), replacement=True
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
