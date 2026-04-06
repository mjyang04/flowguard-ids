from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import load_config
from nids.data.dataset import NIDSDataset
from nids.evaluation.evaluator import evaluate_model
from nids.evaluation.latency import measure_inference_latency
from nids.models.registry import create_model
from nids.utils.io import save_json
from nids.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained NIDS model")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True, help="Path to model weights (.pt)")
    parser.add_argument("--test-data", default=None, help="Path to *.npz data artifact")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("evaluate")
    cfg = load_config(args.config)
    model_path = Path(args.model)

    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        test_data_path = (
            Path(cfg.data.processed_dir)
            / f"cross_{cfg.data.train_dataset}_to_{cfg.data.test_dataset}.npz"
        )
    if not test_data_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_data_path}")

    payload = np.load(test_data_path, allow_pickle=True)
    X_test = payload["X_test"].astype(np.float32)
    y_test = payload["y_test"].astype(np.int64)
    cfg.model.input_dim = int(X_test.shape[1])
    n_classes = int(len(np.unique(y_test)))
    cfg.model.num_classes = 2 if n_classes <= 2 else n_classes

    model = create_model(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)

    test_loader = DataLoader(
        NIDSDataset(X_test, y_test),
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        device=device,
        num_classes=cfg.model.num_classes,
        criterion=None,
    )
    latency = measure_inference_latency(model, test_loader, device=device, n_batches=10)
    report = {
        "model_path": str(model_path),
        "test_data": str(test_data_path),
        "metrics": metrics,
        "latency": latency,
    }
    save_json(report, args.output)
    logger.info("Evaluation report saved to %s", args.output)


if __name__ == "__main__":
    main()
