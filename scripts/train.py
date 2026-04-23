"""Pretrain a CLAN-style self-supervised encoder on Lycos2017.

Usage:
    python scripts/train.py --config configs/default.yaml [--device cuda]

Port of https://github.com/jackwilkie/CLAN/blob/main/train_clan.py
(Apache-2.0). The argparse layer has been replaced with our YAML config;
everything tunable is in ``configs/default.yaml``.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nids.config import ExperimentConfig, RuntimeConfig, load_config, save_config  # noqa: E402
from nids.data import get_data, tabular_dl  # noqa: E402
from nids.models import create_model  # noqa: E402
from nids.training import (  # noqa: E402
    AverageMeter,
    CLANLoss,
    LRSchedule,
    WarmupCosineSchedule,
    make_augmentation,
    make_checkpoint,
)
from nids.utils import get_logger, seed_everything  # noqa: E402

logger = get_logger(__name__)


def resolve_device(preference: str) -> torch.device:
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)


def train_one_epoch(
    loader, model, augmentation, criterion, optimizer, scheduler, epoch, cfg, device
):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    frac_pos_meter = AverageMeter()

    start = time.time()
    for i, (x, _) in enumerate(loader):
        data_time.update(time.time() - start)

        x = x.to(device)
        x_aug = augmentation(x)
        bsz = x.size(0)

        x_cat = torch.cat([x, x_aug], dim=0)
        z_cat = model(x_cat)
        z, z_aug = torch.split(z_cat, [x.size(0), x_aug.size(0)], dim=0)

        loss, frac_pos = criterion(z, z_aug)
        losses.update(loss.item(), bsz)
        frac_pos_meter.update(frac_pos.item())

        optimizer.zero_grad()
        loss.backward()
        if cfg.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
        optimizer.step()
        scheduler.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if (i + 1) % cfg.training.print_freq == 0:
            logger.info(
                "epoch=%d step=%d/%d loss=%.4f frac_pos=%.3f bt=%.3f dt=%.3f",
                epoch, i + 1, len(loader), losses.avg, frac_pos_meter.avg,
                batch_time.avg, data_time.avg,
            )

    return losses.avg


def main() -> None:
    parser = argparse.ArgumentParser("CLAN pretraining")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default=None, help="override cfg.runtime.device")
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    if args.device:
        cfg = replace(cfg, runtime=replace(cfg.runtime, device=args.device))

    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    logger.info("device=%s", device)

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
    logger.info(
        "train=%d test=%d zd=%d",
        splits.x_train.shape[0], splits.x_test.shape[0], splits.x_zd.shape[0],
    )

    train_loader = tabular_dl(
        splits.x_train,
        splits.y_train,
        batch_size=cfg.data.batch_size,
        balanced=cfg.data.balanced_sampling,
        drop_last=True,
        num_workers=cfg.data.num_workers,
    )

    model = create_model(cfg.model).to(device)
    augmentation = make_augmentation(cfg.augmentation).to(device)
    criterion = CLANLoss(
        m=cfg.loss.margin,
        loss_alpha=cfg.loss.loss_alpha,
        squared=cfg.loss.squared,
        distance_metric=cfg.loss.distance_metric,
        eps=cfg.loss.eps,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr_start,
        betas=(cfg.training.adam_beta1, cfg.training.adam_beta2),
        weight_decay=cfg.training.weight_decay,
    )

    total_steps = cfg.training.num_epochs * len(train_loader)
    warmup_steps = int(cfg.training.warmup_ratio * total_steps)
    base_schedule = WarmupCosineSchedule(
        start_val=cfg.training.lr_start,
        end_val=cfg.training.lr_end,
        ref_val=cfg.training.learning_rate,
        T_max=total_steps,
        warmup_steps=warmup_steps,
    )
    scheduler = LRSchedule(optimizer, base_schedule)

    output_dir = Path(cfg.runtime.output_dir) / cfg.loss.name
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "resolved_config.yaml")

    avg_loss = 0.0
    for epoch in range(1, cfg.training.num_epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            train_loader, model, augmentation, criterion, optimizer, scheduler,
            epoch, cfg, device,
        )
        logger.info("epoch=%d avg_loss=%.4f elapsed=%.1fs", epoch, avg_loss, time.time() - t0)

    checkpoint_path = Path(cfg.training.checkpoint_path)
    make_checkpoint(
        model=model,
        path=checkpoint_path,
        optimizer=optimizer,
        scheduler=scheduler,
        stats={"final_loss": float(avg_loss), "epochs": cfg.training.num_epochs},
    )
    logger.info("saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
