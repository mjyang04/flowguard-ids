"""CLAN-specific evaluation metrics.

Ports the upstream helpers:

* ``mean_auroc`` / ``balanced_auroc`` — one-vs-benign AUROC per attack class
  (from https://github.com/jackwilkie/CLAN/blob/main/util/metrics.py, Apache-2.0)
* ``centroid_scores`` — compute centroid-based anomaly scores from embeddings
  (thin wrapper around :func:`nids.training.distance.chunked_centroid_sims`)
* ``evaluate_supervised`` — macro-precision / recall / F1 / accuracy for
  fine-tune evaluation (from CLAN's ``evaluate_metrics``)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _to_numpy(x: object) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):  # torch.Tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mean_auroc(
    scores: object,
    y_true: object,
    *,
    benign_class: int = 0,
    return_class_level: bool = False,
    include_lower_thres: bool = True,
) -> float | list[float]:
    """Per-class one-vs-benign AUROC, averaged across attack classes.

    ``include_lower_thres=True`` mirrors CLAN's behaviour of taking
    ``max(auroc, 1 - auroc)`` to cancel sign flips when the anomaly score
    happens to rank benign higher than attack for a given class.
    """
    y = _to_numpy(y_true).astype(np.int64)
    s = _to_numpy(scores).astype(np.float64)

    roc_scores: list[float] = []
    for c in np.unique(y):
        if c == benign_class:
            continue
        mask = (y == benign_class) | (y == c)
        y_bin = (y[mask] != benign_class).astype(np.int64)
        if len(np.unique(y_bin)) < 2:
            continue
        roc = roc_auc_score(y_bin, s[mask])
        roc_scores.append(max(roc, 1.0 - roc) if include_lower_thres else roc)

    if return_class_level:
        return roc_scores
    return float(np.mean(roc_scores)) if roc_scores else 0.0


def balanced_auroc(
    scores: object,
    y_true: object,
    *,
    benign_class: int = 0,
    return_class_level: bool = False,
) -> float | list[float]:
    """Alias of :func:`mean_auroc` that matches the upstream name."""
    return mean_auroc(
        scores,
        y_true,
        benign_class=benign_class,
        return_class_level=return_class_level,
    )


def centroid_scores(
    benign_features: object,
    test_features: object,
    *,
    chunk_size: Optional[int] = 1024,
) -> np.ndarray:
    """Return ``-cos_sim(centroid, test_features)`` — higher = more anomalous.

    Caller supplies training benign features to derive the centroid.
    """
    try:
        import torch
    except ModuleNotFoundError as e:
        raise RuntimeError("centroid_scores requires torch at runtime") from e

    from nids.training.distance import chunked_centroid_sims

    if not isinstance(benign_features, torch.Tensor):
        benign_features = torch.as_tensor(benign_features, dtype=torch.float32)
    if not isinstance(test_features, torch.Tensor):
        test_features = torch.as_tensor(test_features, dtype=torch.float32)

    centroid = benign_features.mean(dim=0)
    sims = chunked_centroid_sims(test_features, centroid, chunk_size=chunk_size)
    return -1.0 * sims


def evaluate_supervised(
    y_true: object, y_pred: object, *, prefix: str = ""
) -> dict[str, float]:
    """Macro precision / recall / F1 + accuracy for fine-tune evaluation."""
    y_t = _to_numpy(y_true).astype(np.int64)
    y_p = _to_numpy(y_pred).astype(np.int64)
    return {
        f"{prefix}accuracy": float(accuracy_score(y_t, y_p)),
        f"{prefix}mean_recall": float(recall_score(y_t, y_p, average="macro", zero_division=0)),
        f"{prefix}mean_precision": float(
            precision_score(y_t, y_p, average="macro", zero_division=0)
        ),
        f"{prefix}macro_f1": float(f1_score(y_t, y_p, average="macro", zero_division=0)),
    }


__all__ = [
    "mean_auroc",
    "balanced_auroc",
    "centroid_scores",
    "evaluate_supervised",
]
