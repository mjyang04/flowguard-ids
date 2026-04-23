"""Few-shot multiclass fine-tune of a pretrained CLAN encoder.

Usage:
    python scripts/finetune.py --config configs/default.yaml --shots 16

Port of https://github.com/jackwilkie/CLAN/blob/main/finetune_clan.py
(Apache-2.0).
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
from nids.data import get_data, sample_data, tabular_dl  # noqa: E402
from nids.evaluation import evaluate_supervised  # noqa: E402
from nids.models import create_model  # noqa: E402
from nids.training import AverageMeter, load_checkpoint, make_checkpoint  # noqa: E402
from nids.utils import ensure_dir, get_logger, save_json, seed_everything  # noqa: E402

logger = get_logger(__name__)


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def finetune_one_epoch(loader, model, criterion, optimizer, epoch, device):
    model.train()
    losses = AverageMeter()
    start = time.time()
    for _, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), x.size(0))
    logger.info("ft_epoch=%d loss=%.4f elapsed=%.1fs", epoch, losses.avg, time.time() - start)
    return losses.avg


def main() -> None:
    parser = argparse.ArgumentParser("CLAN fine-tune")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--shots", type=int, required=True, help="samples per class")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    if args.device:
        cfg = replace(cfg, runtime=replace(cfg.runtime, device=args.device))
    checkpoint_path = args.checkpoint or cfg.training.checkpoint_path

    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)

    # Supervised split — keep all classes, not only benign.
    splits = get_data(
        cfg.data.csv_path,
        target=cfg.data.target_col,
        drop=cfg.data.drop_cols,
        class_zero=cfg.data.benign_label,
        sample_thres=cfg.data.sample_threshold,
        split_seed=cfg.data.split_seed,
        test_ratio=cfg.data.test_ratio,
        val_ratio=cfg.data.val_ratio,
        anomaly_detection=False,
    )

    x_train, y_train = sample_data(
        splits.x_train,
        splits.y_train,
        num_benign=args.shots,
        num_mal=args.shots,
        sample_seed=args.sample_seed,
    )

    train_loader = tabular_dl(
        x_train,
        y_train,
        batch_size=cfg.finetune.batch_size,
        balanced=cfg.data.balanced_sampling,
        drop_last=False,
    )

    encoder = create_model(cfg.model).to(device)
    load_checkpoint(checkpoint_path, encoder, map_location=device)

    head = nn.Linear(cfg.model.embedding_dim, cfg.model.n_classes).to(device)
    model = nn.Sequential(encoder, head).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.finetune.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.finetune.learning_rate,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.finetune.weight_decay,
    )

    for epoch in range(1, cfg.finetune.num_epochs + 1):
        finetune_one_epoch(train_loader, model, criterion, optimizer, epoch, device)

    model.eval()
    with torch.no_grad():
        x_test_t = torch.as_tensor(splits.x_test, dtype=torch.float32, device=device)
        y_pred = torch.argmax(model(x_test_t), dim=-1).cpu().numpy()
    metrics = evaluate_supervised(splits.y_test, y_pred)
    metrics["shots_per_class"] = args.shots
    logger.info("finetune report:\n%s", pformat(metrics))

    output_dir = ensure_dir(Path(cfg.runtime.output_dir) / cfg.loss.name)
    save_json(metrics, output_dir / f"finetune_report_shots{args.shots}.json")
    make_checkpoint(
        model=model,
        path=cfg.finetune.checkpoint_path,
        optimizer=optimizer,
        stats=metrics,
    )


if __name__ == "__main__":
    main()
