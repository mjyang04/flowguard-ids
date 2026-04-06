from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def select_top_k_features(
    feature_names: list[str], importance: np.ndarray, k: int = 30
):
    sorted_idx = np.argsort(importance)[::-1]
    top_k_idx = sorted_idx[:k]
    selected_features = [feature_names[i] for i in top_k_idx]
    selected_importance = importance[top_k_idx]
    return selected_features, selected_importance, top_k_idx


def select_by_cumulative_importance(
    feature_names: list[str], importance: np.ndarray, threshold: float = 0.9
):
    sorted_idx = np.argsort(importance)[::-1]
    total_importance = float(importance.sum()) if importance.sum() > 0 else 1.0

    cumulative = 0.0
    selected_idx = []
    for idx in sorted_idx:
        selected_idx.append(idx)
        cumulative += float(importance[idx])
        if cumulative / total_importance >= threshold:
            break

    selected_features = [feature_names[i] for i in selected_idx]
    return selected_features, np.array(selected_idx, dtype=int)


def save_feature_selection_results(
    output_dir: str | Path,
    feature_names: list[str],
    selected_features: list[str],
    importance: np.ndarray,
    selected_idx: np.ndarray,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "all_features": feature_names,
        "selected_features": selected_features,
        "selected_indices": selected_idx.tolist(),
        "importance": importance.tolist(),
        "top_k": len(selected_features),
    }
    (out_dir / "feature_selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(out_dir / "selected_feature_indices.npy", selected_idx)


def apply_feature_indices(X: np.ndarray, selected_idx: np.ndarray) -> np.ndarray:
    return X[:, selected_idx]
