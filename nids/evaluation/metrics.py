from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def compute_nids_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, benign_class: int = 0
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    unique_classes = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    attack_classes = [c for c in unique_classes if c != benign_class]

    attack_recalls: list[float] = []
    attack_precisions: list[float] = []
    for cls in attack_classes:
        key = str(cls)
        if key in report:
            attack_recalls.append(report[key]["recall"])
            attack_precisions.append(report[key]["precision"])

    benign_total = max(1, int((y_true == benign_class).sum()))
    benign_false_alarms = int(((y_true == benign_class) & (y_pred != benign_class)).sum())

    avg_attack_recall = float(np.mean(attack_recalls)) if attack_recalls else 0.0
    attack_macro_precision = float(np.mean(attack_precisions)) if attack_precisions else 0.0

    return {
        "accuracy": float(report.get("accuracy", 0.0)),
        "macro_f1": float(report.get("macro avg", {}).get("f1-score", 0.0)),
        "avg_attack_recall": avg_attack_recall,
        "attack_macro_precision": attack_macro_precision,
        "benign_false_alarm_rate": float(benign_false_alarms / benign_total),
        "attack_miss_rate": float(1.0 - avg_attack_recall),
        "confusion_matrix": cm.tolist(),
    }
