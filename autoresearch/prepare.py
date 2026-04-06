"""
Fixed data loading and evaluation for FlowGuard IDS autoresearch.

This file is READ-ONLY — do NOT modify it.
It defines the fixed evaluation metric, dataloaders, and constants
that train.py must use.

Usage (in Docker container):
    python train.py > run.log 2>&1
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Constants (fixed — do not change)
# ---------------------------------------------------------------------------

# Preprocessed .npz produced by scripts/preprocess_cross_dataset.py
# Adjust if you want to target a different experiment setting.
DATA_FILE = Path("/workspace/data/processed/cross_cicids2017_to_unsw_nb15.npz")

INPUT_DIM = 55           # unified feature space dimensionality
NUM_CLASSES = 2          # binary: 0=benign, 1=attack
EPOCH_BUDGET = 50        # maximum training epochs per experiment
BATCH_SIZE = 256         # fixed batch size for fair comparison


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NIDSDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloaders(
    data_file: Path = DATA_FILE,
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load preprocessed .npz and return (train_loader, val_loader, test_loader)."""
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Run: python scripts/preprocess_cross_dataset.py --config configs/default.yaml"
        )
    data = np.load(data_file)
    pin = torch.cuda.is_available()
    kw: dict = dict(batch_size=batch_size, num_workers=0, pin_memory=pin)
    train_loader = DataLoader(NIDSDataset(data["X_train"], data["y_train"]), shuffle=True, **kw)
    val_loader   = DataLoader(NIDSDataset(data["X_val"],   data["y_val"]),   shuffle=False, **kw)
    test_loader  = DataLoader(NIDSDataset(data["X_test"],  data["y_test"]),  shuffle=False, **kw)
    return train_loader, val_loader, test_loader


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
    """Evaluate *model* on *loader* and return a dict of NIDS metrics.

    Primary metric: avg_attack_recall (higher is better).
    This is the ground-truth metric used for experiment comparison.
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits = model(X)
        if num_classes == 2:
            preds = (torch.sigmoid(logits) >= 0.5).long()
        else:
            preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(y.cpu().numpy().tolist())

    y_pred = np.array(all_preds)
    y_true = np.array(all_labels)

    unique_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    attack_classes = [c for c in unique_classes if c != 0]  # 0 = benign

    attack_recalls: list[float] = []
    attack_precisions: list[float] = []
    for cls in attack_classes:
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        recall = tp / max(1, tp + fn)
        precision = tp / max(1, tp + fp)
        attack_recalls.append(recall)
        attack_precisions.append(precision)

    benign_total = max(1, int((y_true == 0).sum()))
    benign_false_alarms = int(((y_true == 0) & (y_pred != 0)).sum())

    avg_attack_recall = float(np.mean(attack_recalls)) if attack_recalls else 0.0
    attack_macro_precision = float(np.mean(attack_precisions)) if attack_precisions else 0.0
    accuracy = float((y_pred == y_true).mean())
    benign_far = float(benign_false_alarms / benign_total)

    return {
        "avg_attack_recall": avg_attack_recall,
        "attack_macro_precision": attack_macro_precision,
        "benign_false_alarm_rate": benign_far,
        "attack_miss_rate": 1.0 - avg_attack_recall,
        "accuracy": accuracy,
    }
