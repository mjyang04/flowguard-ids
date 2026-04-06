"""
Fixed data loading and evaluation for FlowGuard IDS autoresearch.

This file is READ-ONLY — do NOT modify it.
It defines the fixed evaluation metric, dataloaders, and constants
that train.py must use.

Usage (inside Docker container, from /workspace/autoresearch):
    python train.py > run.log 2>&1
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed — do not change)
# ---------------------------------------------------------------------------

# Preprocessed .npz produced by scripts/preprocess_cross_dataset.py
DATA_FILE = Path("/workspace/data/processed/cross_cicids2017_to_unsw_nb15.npz")

INPUT_DIM = 55       # unified feature space dimensionality
NUM_CLASSES = 2      # binary: 0=benign, 1=attack
EPOCH_BUDGET = 50    # training epochs per experiment
BATCH_SIZE = 256     # fixed for fair comparison


# ---------------------------------------------------------------------------
# Dataset (mirrors nids.data.dataset.NIDSDataset)
# ---------------------------------------------------------------------------

class NIDSDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.features = torch.from_numpy(features.astype(np.float32))
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def make_dataloaders(
    data_file: Path = DATA_FILE,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Load .npz and return (train_loader, val_loader, test_loader, y_train, y_test).

    y_train is returned so train.py can compute class weights.
    y_test is returned for final evaluation.
    """
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Run from /workspace: python scripts/preprocess_cross_dataset.py --config configs/default.yaml"
        )
    data = np.load(data_file, allow_pickle=True)
    X_train = data["X_train"].astype(np.float32)
    X_val   = data["X_val"].astype(np.float32)
    X_test  = data["X_test"].astype(np.float32)
    y_train = data["y_train"].astype(np.int64)
    y_val   = data["y_val"].astype(np.int64)
    y_test  = data["y_test"].astype(np.int64)

    pin = torch.cuda.is_available()
    kw: dict = dict(batch_size=batch_size, num_workers=0, pin_memory=pin)
    train_loader = DataLoader(NIDSDataset(X_train, y_train), shuffle=True,  **kw)
    val_loader   = DataLoader(NIDSDataset(X_val,   y_val),   shuffle=False, **kw)
    test_loader  = DataLoader(NIDSDataset(X_test,  y_test),  shuffle=False, **kw)
    return train_loader, val_loader, test_loader, y_train, y_test


# ---------------------------------------------------------------------------
# Fixed evaluation metric — DO NOT CHANGE
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nids(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
) -> dict:
    """Evaluate model. Primary metric: avg_attack_recall (higher is better)."""
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        if num_classes == 2:
            preds = (torch.sigmoid(logits) >= 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    attack_classes = [c for c in sorted(set(y_true) | set(y_pred)) if c != 0]
    attack_recalls: list[float] = []
    attack_precisions: list[float] = []
    for cls in attack_classes:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        attack_recalls.append(tp / max(1, tp + fn))
        attack_precisions.append(tp / max(1, tp + fp))

    n_benign = max(1, int((y_true == 0).sum()))
    false_alarms = int(((y_true == 0) & (y_pred != 0)).sum())

    return {
        "avg_attack_recall":       float(np.mean(attack_recalls))    if attack_recalls else 0.0,
        "attack_macro_precision":  float(np.mean(attack_precisions)) if attack_precisions else 0.0,
        "benign_false_alarm_rate": false_alarms / n_benign,
        "attack_miss_rate":        1.0 - (float(np.mean(attack_recalls)) if attack_recalls else 0.0),
        "accuracy":                float((y_pred == y_true).mean()),
    }
