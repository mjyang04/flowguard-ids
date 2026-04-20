"""Validation-time selection-metric dispatcher.

Kept torch-free so it can be unit-tested without a GPU runtime. The
trainer imports :func:`compute_selection_metric` and calls it with the
predictions/labels collected during evaluation.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


_BINARY_ONLY_METRICS = frozenset(
    {
        "recall_at_far_1pct",
        "recall_at_far_5pct",
        "pr_auc",
        "roc_auc",
        "best_f1",
    }
)

_MULTICLASS_METRICS = frozenset({"macro_f1", "weighted_f1", "accuracy", "mcc"})


def compute_selection_metric(
    name: str,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: Optional[np.ndarray],
    num_classes: int,
) -> float:
    """Return the scalar value of ``name`` for the given predictions.

    Raises
    ------
    ValueError
        If ``name`` is binary-only but ``num_classes > 2``, or if
        ``name`` is not a recognised selection metric.
    """
    if num_classes > 2 and name in _BINARY_ONLY_METRICS:
        raise ValueError(
            f"selection_metric={name!r} is binary-only but num_classes={num_classes}"
        )

    # Local import to keep this module torch-free.
    from nids.evaluation.metrics import compute_nids_metrics

    score_arg = y_score if num_classes == 2 else None
    metrics = compute_nids_metrics(y_true, y_pred, y_score=score_arg)

    if name in metrics:
        value = metrics[name]
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError(
            f"selection_metric={name!r} is not a scalar metric (got {type(value).__name__})"
        )
    if name in _MULTICLASS_METRICS or name in _BINARY_ONLY_METRICS:
        # Key is recognised but absent from metrics dict — treat as 0.0.
        return 0.0
    raise ValueError(f"Unknown selection_metric={name!r}")
