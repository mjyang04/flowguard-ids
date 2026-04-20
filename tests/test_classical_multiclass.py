"""Tests for the classical (sklearn) evaluation path in binary + multiclass.

Skipped where torch is unavailable because ``nids.models`` package init
pulls in torch-dependent siblings.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from nids.models.classical import evaluate_classical_model  # noqa: E402


def test_evaluate_classical_model_multiclass_returns_per_class():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = rng.integers(0, 4, size=200)
    # Inject per-class signal so the RF actually has separable classes.
    for cls in range(4):
        X[y == cls, cls] += 3.0

    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    m = evaluate_classical_model(model, X, y)

    assert "macro_f1" in m
    assert "weighted_f1" in m
    assert set(m["per_class_f1"].keys()) == {"0", "1", "2", "3"}
    # binary-only score metrics must not appear for a 4-class run
    assert "recall_at_far_1pct" not in m


def test_evaluate_classical_model_binary_keeps_score_metrics():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    model = RandomForestClassifier(n_estimators=50, random_state=0).fit(X, y)
    m = evaluate_classical_model(model, X, y)

    # binary score metrics should still be present
    assert "recall_at_far_1pct" in m
    assert "pr_auc" in m
