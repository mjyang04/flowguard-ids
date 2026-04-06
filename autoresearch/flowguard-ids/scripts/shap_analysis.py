from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import load_config
from nids.features.feature_selector import save_feature_selection_results, select_top_k_features
from nids.features.shap_analysis import SHAPAnalyzer
from nids.models.registry import create_model
from nids.utils.io import save_json
from nids.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SHAP analysis for trained NIDS model")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--model-name",
        default=None,
        choices=["cnn_bilstm", "cnn_bilstm_se", "cnn_bilstm_attention"],
        help="Optional override of model architecture name used to load weights",
    )
    parser.add_argument("--data-file", default=None, help="Path to *.npz containing X_train/y_train")
    parser.add_argument("--output-dir", default="artifacts/shap")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("shap")
    cfg = load_config(args.config)
    if args.model_name:
        cfg.model.name = args.model_name
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_file = (
        Path(args.data_file)
        if args.data_file
        else Path(cfg.data.processed_dir)
        / f"cross_{cfg.data.train_dataset}_to_{cfg.data.test_dataset}.npz"
    )
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    payload = np.load(data_file, allow_pickle=True)
    X_train = payload["X_train"].astype(np.float32)
    y_train = payload["y_train"].astype(np.int64)
    feature_names = payload.get("feature_names")
    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(X_train.shape[1])])
    feature_names = feature_names.tolist()
    np.save(output_dir / "feature_names.npy", np.array(feature_names))

    cfg.model.input_dim = int(X_train.shape[1])
    n_classes = int(len(np.unique(y_train)))
    cfg.model.num_classes = 2 if n_classes <= 2 else n_classes

    model = create_model(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    analyzer = SHAPAnalyzer(model=model, device=device)

    X_samples = analyzer.sample_data(X_train, y_train, n_samples=cfg.shap.n_samples)
    np.save(output_dir / "shap_samples.npy", X_samples)
    shap_values = analyzer.compute_shap_values(
        X_samples, background_size=cfg.shap.background_size
    )
    np.save(output_dir / "shap_values.npy", np.array(shap_values, dtype=object), allow_pickle=True)

    importance = analyzer.compute_feature_importance(shap_values)
    sorted_idx = np.argsort(importance)[::-1].astype(np.int64)
    np.save(output_dir / "feature_importance.npy", importance.astype(np.float32))
    np.save(output_dir / "sorted_feature_idx.npy", sorted_idx)
    selected_features, selected_importance, selected_idx = select_top_k_features(
        feature_names, importance, k=min(cfg.shap.top_k, len(feature_names))
    )
    ranking_items = [
        {
            "rank": int(rank + 1),
            "feature": str(feature_names[idx]),
            "index": int(idx),
            "importance": float(importance[idx]),
        }
        for rank, idx in enumerate(sorted_idx.tolist())
    ]
    save_json(
        {
            "reference_model_name": cfg.model.name,
            "n_samples": int(len(X_samples)),
            "background_size": int(min(cfg.shap.background_size, len(X_samples))),
            "ranking": ranking_items,
        },
        output_dir / "feature_ranking.json",
    )
    topk_payload = {}
    for top_k in cfg.shap.top_k_choices:
        selected = sorted_idx[: min(int(top_k), len(sorted_idx))]
        np.save(output_dir / f"top{top_k}_idx.npy", selected.astype(np.int64))
        topk_payload[f"top{top_k}"] = selected.astype(np.int64).tolist()
    save_json(topk_payload, output_dir / "topk_indices.json")
    save_feature_selection_results(
        output_dir=output_dir,
        feature_names=feature_names,
        selected_features=selected_features,
        importance=importance,
        selected_idx=selected_idx,
    )
    save_json(
        {
            "top_k_features": selected_features,
            "top_k_importance": selected_importance.tolist(),
        },
        output_dir / "top_k_summary.json",
    )
    logger.info("SHAP analysis saved to %s", output_dir)


if __name__ == "__main__":
    main()
