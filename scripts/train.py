"""Pretrain a CLAN-style self-supervised encoder.

Usage:
    python scripts/train.py --config configs/lycos.yaml  [--device cuda]
    python scripts/train.py --config configs/cicids.yaml [--device cuda]

Port of https://github.com/jackwilkie/CLAN/blob/main/train_clan.py
(Apache-2.0) with four additions for the Lycos2017 vs CICIDS2017 study:

1. Dataset dispatch via ``cfg.data.dataset`` (``lycos2017`` | ``cicids2017``).
2. Runtime ``input_dim`` inferred from the loaded splits (the upstream
   ``d_in=72`` hard-code fails on CICIDS2017 because the raw CIC feature
   set has a different zero-column footprint).
3. Automatic mixed precision (``autocast`` + ``GradScaler``) gated by
   ``cfg.training.amp`` — essential for RTX 3060 6 GB to fit
   ``batch_size=2048`` with the 4×1024 MLP.
4. Run directory layout ``artifacts/<dataset>/<loss>/seed<S>/`` so that
   downstream scripts can locate each run without scanning.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nids.config import ExperimentConfig, load_config, save_config  # noqa: E402
from nids.data import get_data_by_name, tabular_dl  # noqa: E402
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
    loader, model, augmentation, criterion, optimizer, scheduler,
    epoch, cfg, device, scaler,
):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    frac_pos_meter = AverageMeter()
    use_amp = scaler is not None

    start = time.time()
    for i, (x, _) in enumerate(loader):
        data_time.update(time.time() - start)

        x = x.to(device, non_blocking=True)
        x_aug = augmentation(x)
        bsz = x.size(0)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                x_cat = torch.cat([x, x_aug], dim=0)
                z_cat = model(x_cat)
                z, z_aug = torch.split(z_cat, [x.size(0), x_aug.size(0)], dim=0)
                loss, frac_pos = criterion(z, z_aug)
            scaler.scale(loss).backward()
            if cfg.training.gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            x_cat = torch.cat([x, x_aug], dim=0)
            z_cat = model(x_cat)
            z, z_aug = torch.split(z_cat, [x.size(0), x_aug.size(0)], dim=0)
            loss, frac_pos = criterion(z, z_aug)
            loss.backward()
            if cfg.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.gradient_clip)
            optimizer.step()

        scheduler.step()

        losses.update(loss.item(), bsz)
        frac_pos_meter.update(frac_pos.item())

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
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="override cfg.runtime.seed")
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    if args.device:
        cfg = replace(cfg, runtime=replace(cfg.runtime, device=args.device))
    if args.seed is not None:
        cfg = replace(cfg, runtime=replace(cfg.runtime, seed=args.seed))

    seed_everything(cfg.runtime.seed)
    device = resolve_device(cfg.runtime.device)
    logger.info("device=%s dataset=%s seed=%d", device, cfg.data.dataset, cfg.runtime.seed)

    splits = get_data_by_name(
        cfg.data.dataset,
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
        "train=%d test=%d zd=%d input_dim=%d classes=%d",
        splits.x_train.shape[0], splits.x_test.shape[0], splits.x_zd.shape[0],
        splits.input_dim, len(splits.class_names),
    )

    train_loader = tabular_dl(
        splits.x_train, splits.y_train,
        batch_size=cfg.data.batch_size,
        balanced=cfg.data.balanced_sampling,
        drop_last=True,
        num_workers=cfg.data.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = create_model(cfg.model, input_dim=splits.input_dim).to(device)
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

    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    if use_amp:
        logger.info("mixed precision enabled (fp16 autocast + GradScaler)")

    run_dir = Path(cfg.runtime.output_dir) / cfg.data.dataset / cfg.loss.name / f"seed{cfg.runtime.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, run_dir / "resolved_config.yaml")

    avg_loss = 0.0
    for epoch in range(1, cfg.training.num_epochs + 1):
        t0 = time.time()
        avg_loss = train_one_epoch(
            train_loader, model, augmentation, criterion, optimizer, scheduler,
            epoch, cfg, device, scaler,
        )
        logger.info("epoch=%d avg_loss=%.4f elapsed=%.1fs", epoch, avg_loss, time.time() - t0)

    checkpoint_path = run_dir / "clan.pt.tar"
    make_checkpoint(
        model=model,
        path=checkpoint_path,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        stats={
            "final_loss": float(avg_loss),
            "epochs": cfg.training.num_epochs,
            "input_dim": splits.input_dim,
            "dataset": cfg.data.dataset,
            "seed": cfg.runtime.seed,
        },
    )
    logger.info("saved checkpoint to %s", checkpoint_path)


if __name__ == "__main__":
    main()
