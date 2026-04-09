"""Platt scaling calibration for cross-dataset probability reliability.

Based on Xin & Xu (2025) "Cross-Dataset Transformer-IDS with Calibration
and AUC Optimization". Platt scaling fits a logistic regression on raw
model logits using held-out validation data:

    p_calibrated = sigmoid(A * logit + B)

where A and B are learned parameters that minimize NLL on the calibration
set. This adjusts predicted probabilities to better reflect true attack
frequencies, which is especially valuable in cross-dataset scenarios where
the model's confidence may be poorly calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from nids.utils.logging import get_logger

logger = get_logger("calibration")


@dataclass
class CalibrationResult:
    """Stores fitted Platt scaling parameters."""

    A: float
    B: float
    val_ece_before: float
    val_ece_after: float


def _expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob > bin_edges[i]) & (y_prob <= bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)


class PlattCalibrator:
    """Platt scaling calibrator for binary classification logits."""

    def __init__(self) -> None:
        self.A: float = 1.0
        self.B: float = 0.0
        self._fitted: bool = False

    def fit(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        lr: float = 0.01,
        max_iter: int = 1000,
    ) -> CalibrationResult:
        """Fit Platt scaling parameters A, B on validation logits.

        Args:
            logits: Raw model logits (before sigmoid), shape (n,).
            labels: Binary labels (0 or 1), shape (n,).
            lr: Learning rate for gradient descent.
            max_iter: Maximum optimization iterations.

        Returns:
            CalibrationResult with fitted parameters and ECE scores.
        """
        # Compute ECE before calibration
        probs_before = 1.0 / (1.0 + np.exp(-logits))
        ece_before = _expected_calibration_error(labels, probs_before)

        # Initialize A=1, B=0 (identity mapping)
        A = 1.0
        B = 0.0

        for _ in range(max_iter):
            z = A * logits + B
            p = 1.0 / (1.0 + np.exp(-z))
            p = np.clip(p, 1e-7, 1.0 - 1e-7)

            # Gradient of NLL w.r.t. A and B
            residual = p - labels
            grad_A = float(np.mean(residual * logits))
            grad_B = float(np.mean(residual))

            A -= lr * grad_A
            B -= lr * grad_B

        self.A = A
        self.B = B
        self._fitted = True

        # Compute ECE after calibration
        probs_after = 1.0 / (1.0 + np.exp(-(A * logits + B)))
        ece_after = _expected_calibration_error(labels, probs_after)

        logger.info(
            "Platt calibration fitted | A=%.4f B=%.4f ECE_before=%.4f ECE_after=%.4f",
            A, B, ece_before, ece_after,
        )

        return CalibrationResult(
            A=A, B=B, val_ece_before=ece_before, val_ece_after=ece_after,
        )

    def transform(self, logits: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to raw logits, returning calibrated probabilities."""
        if not self._fitted:
            raise RuntimeError("PlattCalibrator has not been fitted yet.")
        z = self.A * logits + self.B
        return 1.0 / (1.0 + np.exp(-z))

    def save(self, path: str | Path) -> None:
        """Save calibration parameters to a JSON-compatible .npz file."""
        np.savez(
            Path(path),
            A=np.array(self.A),
            B=np.array(self.B),
        )

    def load(self, path: str | Path) -> None:
        """Load calibration parameters from a .npz file."""
        data = np.load(Path(path))
        self.A = float(data["A"])
        self.B = float(data["B"])
        self._fitted = True


def collect_logits(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect raw logits and labels from a data loader.

    Args:
        model: Trained model in eval mode.
        data_loader: Validation or calibration data loader.
        device: Torch device.

    Returns:
        Tuple of (logits, labels) as numpy arrays.
    """
    model.eval()
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            outputs = model(features)
            all_logits.append(outputs.cpu().numpy().reshape(-1))
            all_labels.append(labels.numpy().reshape(-1))

    return np.concatenate(all_logits), np.concatenate(all_labels)
