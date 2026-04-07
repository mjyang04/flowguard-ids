from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from nids.evaluation.metrics import compute_nids_metrics


def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train: np.ndarray, y_train: np.ndarray):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("xgboost is not installed") from exc

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
    )
    model.fit(X_train, y_train)
    return model


def predict_binary_scores(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X), dtype=np.float64)
        if probs.ndim == 1:
            return np.clip(probs, 0.0, 1.0)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return np.clip(probs[:, 1], 0.0, 1.0)
        if probs.ndim == 2 and probs.shape[1] == 1:
            return np.clip(probs[:, 0], 0.0, 1.0)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X), dtype=np.float64).reshape(-1)
        return 1.0 / (1.0 + np.exp(-scores))

    return None


def evaluate_classical_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds = model.predict(X_test)
    scores = predict_binary_scores(model, X_test) if len(np.unique(y_test)) == 2 else None
    return compute_nids_metrics(y_test, preds, y_score=scores)
