from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.features.feature_selector import (
    apply_feature_indices,
    save_feature_selection_results,
    select_by_cumulative_importance,
    select_top_k_features,
)
from nids.features.importance import compute_feature_importance
from nids.utils.io import save_json
from nids.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select features from SHAP values")
    parser.add_argument("--shap-values", default=None, help="Path to shap_values.npy")
    parser.add_argument("--selected-idx", default=None, help="Path to precomputed selected indices (*.npy)")
    parser.add_argument("--feature-names", required=True, help="Path to feature_names.npy")
    parser.add_argument("--data-file", default=None, help="Optional input npz to generate reduced features")
    parser.add_argument("--output-dir", default="artifacts/feature_selection")
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--mode", choices=["topk", "cumulative"], default="topk")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("feature_selection")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = np.load(args.feature_names, allow_pickle=True).tolist()
    if args.selected_idx:
        selected_idx = np.load(args.selected_idx, allow_pickle=True).astype(np.int64)
        selected_features = [feature_names[int(i)] for i in selected_idx.tolist()]
        importance = None
        (output_dir / "selected_feature_indices.npy").write_bytes(Path(args.selected_idx).read_bytes())
        (output_dir / "feature_selection.json").write_text(
            json.dumps(
                {
                    "all_features": feature_names,
                    "selected_features": selected_features,
                    "selected_indices": [int(i) for i in selected_idx.tolist()],
                    "importance": None,
                    "top_k": len(selected_features),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    else:
        if not args.shap_values:
            raise ValueError("Either --shap-values or --selected-idx must be provided.")
        shap_values = np.load(args.shap_values, allow_pickle=True)
        if shap_values.dtype == object and shap_values.shape == ():
            shap_values = shap_values.item()
        importance = compute_feature_importance(shap_values)

        if args.mode == "topk":
            selected_features, _, selected_idx = select_top_k_features(
                feature_names, importance, k=min(args.top_k, len(feature_names))
            )
        else:
            selected_features, selected_idx = select_by_cumulative_importance(
                feature_names, importance, threshold=args.threshold
            )

        save_feature_selection_results(
            output_dir=output_dir,
            feature_names=feature_names,
            selected_features=selected_features,
            importance=importance,
            selected_idx=selected_idx,
        )

    if args.data_file:
        payload = np.load(args.data_file, allow_pickle=True)
        reduced = {}
        for k in payload.files:
            v = payload[k]
            if k.startswith("X_"):
                reduced[k] = apply_feature_indices(v, selected_idx)
            elif k == "feature_names":
                reduced[k] = np.array(selected_features)
            else:
                reduced[k] = v
        np.savez_compressed(output_dir / "reduced_data.npz", **reduced)

    save_json(
        {
            "mode": "precomputed" if args.selected_idx else args.mode,
            "top_k": len(selected_features),
            "selected_features": selected_features,
            "selected_indices": [int(i) for i in selected_idx.tolist()],
        },
        output_dir / "summary.json",
    )
    logger.info("Feature selection completed: %s", output_dir)


if __name__ == "__main__":
    main()
