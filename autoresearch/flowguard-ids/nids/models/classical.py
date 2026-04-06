from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


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


def evaluate_classical_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    return {
        "accuracy": report.get("accuracy", 0.0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0.0),
    }
