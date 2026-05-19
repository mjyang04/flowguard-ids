"""Few-shot multiclass fine-tune of a pretrained CLAN encoder — one run.

Usage:
    python scripts/finetune.py --config configs/lycos.yaml --shots 16 --sample-seed 0

For the paper-faithful "average over 10 runs" protocol use
``scripts/finetune_sweep.py`` which loops this over multiple sample seeds.

Port of https://github.com/jackwilkie/CLAN/blob/main/finetune_clan.py (Apache-2.0).

Known paper-vs-code discrepancy preserved here:
  The paper (Wilkie et al., IEEE CSR 2025, §V-A) reports a fine-tune
  learning rate of 10^-6, but the upstream argparse default is 10^-3.
  Using 10^-6 over 100 epochs yields essentially frozen weights,
  inconsistent with the paper's reported 8-shot macro-F1 of 0.496. We
  adopt the code value (10^-3, configurable via cfg.finetune.learning_rate)
  and interpret the paper figure as a typographical error. See thesis
  Chapter 4 "Implementation Notes" for the discussion.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from pprint import pformat

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nids.config import ExperimentConfig, load_config  # noqa: E402
from nids.data import get_data_by_name, sample_data, tabular_dl  # noqa: E402
from nids.evaluation import evaluate_supervised  # noqa: E402
from nids.models import create_model  # noqa: E402
from nids.training import (  # noqa: E402
    AverageMeter,
    load_checkpoint,
)
from nids.utils import ensure_dir, get_logger, save_json, seed_everything  # noqa: E402

logger = get_logger(__name__)


def resolve_device(preference: str) -> torch.device:
    """Resolve ``auto`` to CUDA when available, otherwise return the requested device."""
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def finetune_one_epoch(loader, model, criterion, optimizer, epoch, device):
    """Run one supervised fine-tuning epoch and return average CE loss."""
    model.train()
    losses = AverageMeter()
    start = time.time()
    for _, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "non-finite fine-tune loss; check input scaling and checkpoint"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), x.size(0))
    logger.info("ft_epoch=%d loss=%.4f elapsed=%.1fs", epoch, losses.avg, time.time() - start)
    return losses.avg


def run_finetune(
    cfg: ExperimentConfig,
    *,
    shots: int,
    sample_seed: int,
    device: torch.device,
    checkpoint_path: str | Path,
) -> dict:
    """Single fine-tune run. Returns the metrics dict."""
    splits = get_data_by_name(
        cfg.data.dataset,
        cfg.data.csv_path,
        drop=cfg.data.drop_cols,
        sample_thres=cfg.data.sample_threshold,
        split_seed=cfg.data.split_seed,
        test_ratio=cfg.data.test_ratio,
        anomaly_detection=False,
    )

    x_train, y_train = sample_data(
        splits.x_train,
        splits.y_train,
        num_benign=shots,
        num_mal=shots,
        sample_seed=sample_seed,
    )

    train_loader = tabular_dl(
        x_train, y_train,
        batch_size=cfg.finetune.batch_size,
        balanced=True,
        drop_last=False,
    )

    encoder = create_model(cfg.model, input_dim=splits.input_dim).to(device)
    load_checkpoint(checkpoint_path, encoder, map_location=device)

    n_classes = len(splits.class_names)
    head = nn.Linear(cfg.model.embedding_dim, n_classes).to(device)
    model = nn.Sequential(encoder, head).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.finetune.learning_rate,
        weight_decay=cfg.finetune.weight_decay,
    )

    for epoch in range(1, cfg.finetune.num_epochs + 1):
        finetune_one_epoch(train_loader, model, criterion, optimizer, epoch, device)

    model.eval()
    with torch.no_grad():
        x_test_t = torch.as_tensor(splits.x_test, dtype=torch.float32, device=device)
        y_pred = torch.argmax(model(x_test_t), dim=-1).cpu().numpy()
    metrics = evaluate_supervised(splits.y_test, y_pred)
    metrics["shots_per_class"] = shots
    metrics["sample_seed"] = sample_seed
    metrics["dataset"] = cfg.data.dataset
    metrics["n_classes"] = int(n_classes)
    return metrics


def main() -> None:
    """Parse CLI arguments and run one few-shot fine-tune evaluation."""
    parser = argparse.ArgumentParser("CLAN fine-tune (single run)")
    parser.add_argument("--config", type=str, default="configs/lycos.yaml")
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--pretrain-seed", type=int, default=None,
                        help="which pretraining seed's checkpoint to load")
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    if args.pretrain_seed is not None:
        cfg = replace(cfg, runtime=replace(cfg.runtime, seed=args.pretrain_seed))

    seed_everything(args.sample_seed)  # sample_seed governs ft reproducibility
    device = resolve_device(cfg.runtime.device)

    run_dir = Path(cfg.runtime.output_dir) / cfg.data.dataset / "clan" / f"seed{cfg.runtime.seed}"
    checkpoint_path = run_dir / "clan.pt.tar"

    metrics = run_finetune(
        cfg, shots=args.shots, sample_seed=args.sample_seed,
        device=device, checkpoint_path=checkpoint_path,
    )
    logger.info("finetune report:\n%s", pformat(metrics))

    ensure_dir(run_dir)
    save_json(metrics, run_dir / f"finetune_shots{args.shots}_seed{args.sample_seed}.json")


if __name__ == "__main__":
    main()
