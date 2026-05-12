"""Centroid-based AUROC evaluation of a pretrained CLAN encoder.

Usage:
    python scripts/eval.py --config configs/lycos.yaml
    python scripts/eval.py --config configs/cicids.yaml --seed 43

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
from nids.data import get_data_by_name  # noqa: E402
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
    parser.add_argument("--config", type=str, default="configs/lycos.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    if args.seed is not None:
        cfg = replace(cfg, runtime=replace(cfg.runtime, seed=args.seed))

    run_dir = Path(cfg.runtime.output_dir) / cfg.data.dataset / "clan" / f"seed{cfg.runtime.seed}"
    checkpoint_path = run_dir / "clan.pt.tar"

    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    logger.info("device=%s checkpoint=%s", device, checkpoint_path)

    splits = get_data_by_name(
        cfg.data.dataset,
        cfg.data.csv_path,
        drop=cfg.data.drop_cols,
        sample_thres=cfg.data.sample_threshold,
        split_seed=cfg.data.split_seed,
        test_ratio=cfg.data.test_ratio,
        anomaly_detection=True,
    )

    x_test = np.concatenate([splits.x_test, splits.x_zd], axis=0)
    y_test = np.concatenate([splits.y_test, splits.y_zd], axis=0)

    model = create_model(cfg.model, input_dim=splits.input_dim).to(device)
    load_checkpoint(checkpoint_path, model, map_location=device)

    benign_features = embed_all(model, splits.x_train, device)
    test_features = embed_all(model, x_test, device)

    scores = centroid_scores(benign_features, test_features)
    overall = mean_auroc(scores, y_test, benign_class=0)
    per_class = balanced_auroc(scores, y_test, benign_class=0, return_class_level=True)

    non_benign_names = [c for c in splits.class_names if c != "benign"]
    per_class_named = {
        name: float(auroc) for name, auroc in zip(non_benign_names, per_class)
    }

    report = {
        "dataset": cfg.data.dataset,
        "seed": cfg.runtime.seed,
        "mean_auroc": float(overall),
        "per_class_auroc": per_class_named,
        "checkpoint": str(checkpoint_path),
        "input_dim": int(splits.input_dim),
        "n_train_benign": int(splits.x_train.shape[0]),
        "n_test": int(x_test.shape[0]),
    }

    ensure_dir(run_dir)
    save_json(report, run_dir / "eval_report.json")
    logger.info("eval report:\n%s", pformat(report))


if __name__ == "__main__":
    main()
