from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


class NIDSDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def _build_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that balances classes during training.

    Each sample is assigned weight = 1 / count(its_class), so minority classes
    are sampled proportionally more often. This is the strategy recommended by
    Najar et al. (2025) which outperformed SMOTE and oversampling.
    """
    classes, counts = np.unique(labels, return_counts=True)
    class_weight = {c: 1.0 / n for c, n in zip(classes, counts)}
    sample_weights = np.array([class_weight[label] for label in labels], dtype=np.float64)
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(labels),
        replacement=True,
    )


def create_dataloaders(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 128,
    num_workers: int = 0,
    weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = NIDSDataset(X_train, y_train)
    val_ds = NIDSDataset(X_val, y_val)
    test_ds = NIDSDataset(X_test, y_test)

    if weighted_sampler:
        sampler = _build_weighted_sampler(y_train)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
