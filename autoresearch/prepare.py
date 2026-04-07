"""
Fixed data loading and evaluation for FlowGuard IDS autoresearch.

This file defines the shared v2 evaluation metric, dataloaders, and constants
that train.py must use.

Usage (inside Docker container, from /workspace/autoresearch):
    python train.py > run.log 2>&1
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.evaluation.metrics import compute_nids_metrics

# ---------------------------------------------------------------------------
# Constants (fixed — do not change)
# ---------------------------------------------------------------------------

# Cross-dataset: train on CICIDS2017, test on UNSW-NB15
CROSS_DATA_FILE = Path("/workspace/data/processed/cross_cicids2017_to_unsw_nb15.npz")
# Single-dataset: train/val/test all from CICIDS2017
SINGLE_DATA_FILE = Path("/workspace/data/processed/cicids2017/data.npz")

INPUT_DIM = 55       # unified feature space dimensionality
NUM_CLASSES = 2      # binary: 0=benign, 1=attack
MAX_EPOCHS = 50      # maximum training epochs per experiment
BATCH_SIZE = 2048    # fixed for fair comparison


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


def _load_npz(data_file: Path) -> dict:
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            "Run preprocessing first — see CLAUDE.md for commands."
        )
    return dict(np.load(data_file, allow_pickle=True))


def make_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, np.ndarray]:
    """Load both datasets and return loaders for training + dual evaluation.

    Returns:
        train_loader:      CICIDS2017 train split (for training)
        val_loader:        CICIDS2017 val split (for early stopping)
        test_single_loader: CICIDS2017 test split (single-dataset eval)
        test_cross_loader:  UNSW-NB15 test split (cross-dataset eval)
        y_train:           training labels (for class weight computation)
    """
    cross = _load_npz(CROSS_DATA_FILE)
    single = _load_npz(SINGLE_DATA_FILE)

    X_train = cross["X_train"].astype(np.float32)
    X_val   = cross["X_val"].astype(np.float32)
    y_train = cross["y_train"].astype(np.int64)
    y_val   = cross["y_val"].astype(np.int64)

    # Cross-dataset test: UNSW-NB15
    X_test_cross = cross["X_test"].astype(np.float32)
    y_test_cross = cross["y_test"].astype(np.int64)

    # Single-dataset test: CICIDS2017
    X_test_single = single["X_test"].astype(np.float32)
    y_test_single = single["y_test"].astype(np.int64)

    pin = torch.cuda.is_available()
    kw: dict = dict(batch_size=batch_size, num_workers=0, pin_memory=pin)
    train_loader       = DataLoader(NIDSDataset(X_train, y_train),           shuffle=True,  **kw)
    val_loader         = DataLoader(NIDSDataset(X_val, y_val),               shuffle=False, **kw)
    test_single_loader = DataLoader(NIDSDataset(X_test_single, y_test_single), shuffle=False, **kw)
    test_cross_loader  = DataLoader(NIDSDataset(X_test_cross, y_test_cross),   shuffle=False, **kw)
    return train_loader, val_loader, test_single_loader, test_cross_loader, y_train


# ---------------------------------------------------------------------------
# Shared evaluation metric
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_nids(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int = NUM_CLASSES,
    use_amp: bool = True,
) -> dict:
    """Evaluate model with the shared v2 imbalance-aware metrics."""
    model.eval()
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        with torch.autocast(device_type=device.type, enabled=use_amp and device.type == "cuda"):
            logits = model(features)
        if num_classes == 2:
            probs = torch.sigmoid(logits).reshape(-1)
            preds = (probs >= 0.5).long()
            all_scores.append(probs.cpu().numpy())
        else:
            preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    y_score = np.concatenate(all_scores) if all_scores else None
    return compute_nids_metrics(y_true, y_pred, y_score=y_score)
