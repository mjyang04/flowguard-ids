"""Evaluate a pretrained CLAN encoder via centroid-based anomaly scoring.

Usage:
    python scripts/eval.py --config configs/default.yaml \\
        [--checkpoint weights/clan.pt.tar] [--device cuda]

Port of https://github.com/jackwilkie/CLAN/blob/main/eval_clan.py (Apache-2.0).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from pprint import pformat

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nids.config import ExperimentConfig, load_config  # noqa: E402
from nids.data import get_data  # noqa: E402
from nids.evaluation import balanced_auroc, centroid_scores, mean_auroc  # noqa: E402
from nids.models import create_model  # noqa: E402
from nids.training import load_checkpoint  # noqa: E402
from nids.utils import ensure_dir, get_logger, save_json, seed_everything  # noqa: E402

logger = get_logger(__name__)


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


@torch.no_grad()
def embed_all(model, x: np.ndarray, device: torch.device, chunk_size: int = 1024) -> torch.Tensor:
    """Run the encoder over ``x`` in chunks and return concatenated embeddings."""
    model.eval()
    x_t = torch.as_tensor(x, dtype=torch.float32)
    out: list[torch.Tensor] = []
    for start in range(0, x_t.size(0), chunk_size):
        end = min(start + chunk_size, x_t.size(0))
        z = model(x_t[start:end].to(device))
        out.append(z.detach().cpu())
    return torch.cat(out, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser("CLAN evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chunk-size", type=int, default=1024)
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    if args.device:
        cfg = replace(cfg, runtime=replace(cfg.runtime, device=args.device))
    checkpoint_path = args.checkpoint or cfg.training.checkpoint_path

    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    logger.info("device=%s checkpoint=%s", device, checkpoint_path)

    splits = get_data(
        cfg.data.csv_path,
        target=cfg.data.target_col,
        drop=cfg.data.drop_cols,
        class_zero=cfg.data.benign_label,
        sample_thres=cfg.data.sample_threshold,
        split_seed=cfg.data.split_seed,
        test_ratio=cfg.data.test_ratio,
        val_ratio=cfg.data.val_ratio,
        anomaly_detection=True,
    )

    # Merge zero-day set into the test set for the final AUROC report.
    x_test = np.concatenate([splits.x_test, splits.x_zd], axis=0)
    y_test = np.concatenate([splits.y_test, splits.y_zd], axis=0)

    model = create_model(cfg.model).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)

    benign_features = embed_all(model, splits.x_train, device, args.chunk_size)
    test_features = embed_all(model, x_test, device, args.chunk_size)

    scores = centroid_scores(benign_features, test_features, chunk_size=args.chunk_size)
    overall = mean_auroc(scores, y_test, benign_class=0)
    per_class = balanced_auroc(scores, y_test, benign_class=0, return_class_level=True)

    non_benign_names = [c for c in splits.class_names if c != cfg.data.benign_label]
    per_class_named = {
        name: float(auroc) for name, auroc in zip(non_benign_names, per_class)
    }

    report = {
        "mean_auroc": float(overall),
        "per_class_auroc": per_class_named,
        "checkpoint": str(checkpoint_path),
    }

    output_dir = ensure_dir(Path(cfg.runtime.output_dir) / cfg.loss.name)
    save_json(report, output_dir / "eval_report.json")
    logger.info("eval report:\n%s", pformat(report))


if __name__ == "__main__":
    main()
