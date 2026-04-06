"""
FlowGuard IDS autoresearch training script. Single-GPU, single-file.
Adapt this file freely — it is the only file you should edit.

Usage: python train.py
       python train.py > run.log 2>&1
"""

from __future__ import annotations

import math
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import EPOCH_BUDGET, INPUT_DIM, NUM_CLASSES, BATCH_SIZE, evaluate_nids, make_dataloaders

# ---------------------------------------------------------------------------
# Hyperparameters — edit freely
# ---------------------------------------------------------------------------

# Architecture
CONV_CHANNELS: List[int] = [64, 128]       # CNN channel widths per block
CONV_KERNEL_SIZES: List[int] = [3, 3]      # kernel size per conv block
CONV_POOL_SIZES: List[int] = [2, 2]        # max-pool factor per conv block
USE_SE: bool = True                         # Squeeze-Excitation channel attention
USE_ATTENTION: bool = False                 # additive attention pooling (vs mean)
LSTM_HIDDEN: int = 128                      # LSTM hidden size
LSTM_LAYERS: int = 2                        # number of LSTM layers
BIDIRECTIONAL: bool = True                  # bidirectional LSTM
DROPOUT: float = 0.3                        # dropout rate

# Optimization
LEARNING_RATE: float = 2e-3
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE_OVERRIDE: int | None = None      # set to override prepare.BATCH_SIZE
GRADIENT_CLIP: float = 1.0
USE_AMP: bool = True                        # automatic mixed precision
OPTIMIZER_NAME: str = "adam"                # "adam" or "sgd" or "adamw"

# Scheduler
USE_SCHEDULER: bool = True
SCHEDULER_TYPE: str = "plateau"            # "plateau" or "cosine" or "none"
SCHEDULER_PATIENCE: int = 3
SCHEDULER_FACTOR: float = 0.5
MIN_LR: float = 1e-6

# Class imbalance
USE_CLASS_WEIGHTS: bool = True             # weight loss by inverse class frequency

# ---------------------------------------------------------------------------
# Model — copy-paste friendly, no external imports from nids package
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Conv1d -> BN -> ReLU -> MaxPool -> Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class SqueezeExcitation(nn.Module):
    """SE channel-attention block."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        scale = self.fc(x.mean(dim=-1))          # (B, C)
        return x * scale.unsqueeze(-1)


class AttentionPooling(nn.Module):
    """Additive attention over the sequence dimension."""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, B, H)
        w = torch.softmax(self.score(x), dim=0)  # (L, B, 1)
        return (w * x).sum(dim=0)                # (B, H)


class CNNBiLSTMSE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        conv_channels: List[int],
        conv_kernel_sizes: List[int],
        conv_pool_sizes: List[int],
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
        bidirectional: bool,
        use_se: bool,
        use_attention: bool,
    ):
        super().__init__()
        cnn_layers: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, p in zip(conv_channels, conv_kernel_sizes, conv_pool_sizes):
            cnn_layers.append(ConvBlock(in_ch, out_ch, k, p, dropout))
            if use_se:
                cnn_layers.append(SqueezeExcitation(out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.attn = AttentionPooling(lstm_out_dim) if use_attention else None
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim // 2, num_classes if num_classes > 2 else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        x = x.unsqueeze(1)                  # (B, 1, F)
        x = self.cnn(x)                     # (B, C, L')
        x = x.permute(2, 0, 1)             # (L', B, C)
        lstm_out, _ = self.lstm(x)          # (L', B, H)
        pooled = self.attn(lstm_out) if self.attn is not None else lstm_out.mean(0)
        logits = self.head(pooled)          # (B, cls) or (B,)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

batch_size = BATCH_SIZE_OVERRIDE or BATCH_SIZE
train_loader, val_loader, test_loader = make_dataloaders(batch_size=batch_size)
print(f"Train: {len(train_loader.dataset):,}  Val: {len(val_loader.dataset):,}  Test: {len(test_loader.dataset):,}")

model = CNNBiLSTMSE(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    conv_channels=CONV_CHANNELS,
    conv_kernel_sizes=CONV_KERNEL_SIZES,
    conv_pool_sizes=CONV_POOL_SIZES,
    lstm_hidden=LSTM_HIDDEN,
    lstm_layers=LSTM_LAYERS,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
    use_se=USE_SE,
    use_attention=USE_ATTENTION,
).to(device)

num_params = count_params(model)
print(f"Parameters: {num_params / 1e6:.3f}M")

# Class weights
if USE_CLASS_WEIGHTS:
    y_train = train_loader.dataset.y.numpy()  # type: ignore[attr-defined]
    classes, counts = np.unique(y_train, return_counts=True)
    weights = 1.0 / counts.astype(np.float32)
    weights /= weights.mean()
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
else:
    weight_tensor = None

if NUM_CLASSES == 2:
    pos_weight = None
    if USE_CLASS_WEIGHTS:
        y_tr = train_loader.dataset.y.numpy()  # type: ignore[attr-defined]
        n_neg = int((y_tr == 0).sum())
        n_pos = int((y_tr == 1).sum())
        pos_weight = torch.tensor([n_neg / max(1, n_pos)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

# Optimizer
if OPTIMIZER_NAME == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_NAME == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER_NAME == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9,
                                 weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER_NAME}")

# Scheduler
scheduler = None
if USE_SCHEDULER and SCHEDULER_TYPE != "none":
    if SCHEDULER_TYPE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=SCHEDULER_FACTOR,
            patience=SCHEDULER_PATIENCE, min_lr=MIN_LR,
        )
    elif SCHEDULER_TYPE == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=EPOCH_BUDGET, eta_min=MIN_LR,
        )

scaler = torch.amp.GradScaler("cuda") if USE_AMP and device.type == "cuda" else None

print(f"Optimizer: {OPTIMIZER_NAME}  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}")
print(f"Epochs: {EPOCH_BUDGET}  batch={batch_size}  amp={scaler is not None}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
best_val_recall = 0.0
best_epoch = 0
total_training_seconds = 0.0

for epoch in range(1, EPOCH_BUDGET + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = model(X)
                loss = criterion(logits, y.float() if NUM_CLASSES == 2 else y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = criterion(logits, y.float() if NUM_CLASSES == 2 else y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

        epoch_loss += loss.item()

    dt = time.time() - t0
    total_training_seconds += dt
    avg_loss = epoch_loss / max(1, len(train_loader))

    val_metrics = evaluate_nids(model, val_loader, device, NUM_CLASSES)
    val_recall = val_metrics["avg_attack_recall"]

    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_recall)
        else:
            scheduler.step()

    if val_recall > best_val_recall:
        best_val_recall = val_recall
        best_epoch = epoch

    lr_now = optimizer.param_groups[0]["lr"]
    print(
        f"\repoch {epoch:03d}/{EPOCH_BUDGET} | "
        f"loss={avg_loss:.4f} | "
        f"val_recall={val_recall:.4f} | "
        f"best={best_val_recall:.4f} (ep{best_epoch}) | "
        f"lr={lr_now:.1e} | "
        f"dt={dt:.1f}s    ",
        end="",
        flush=True,
    )

print()  # newline after \r log

# ---------------------------------------------------------------------------
# Final evaluation on test set
# ---------------------------------------------------------------------------

test_metrics = evaluate_nids(model, test_loader, device, NUM_CLASSES)
t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0

print("---")
print(f"val_avg_attack_recall:    {best_val_recall:.6f}")
print(f"test_avg_attack_recall:   {test_metrics['avg_attack_recall']:.6f}")
print(f"test_false_alarm_rate:    {test_metrics['benign_false_alarm_rate']:.6f}")
print(f"test_attack_precision:    {test_metrics['attack_macro_precision']:.6f}")
print(f"test_accuracy:            {test_metrics['accuracy']:.6f}")
print(f"training_seconds:         {total_training_seconds:.1f}")
print(f"total_seconds:            {t_end - t_start:.1f}")
print(f"peak_vram_mb:             {peak_vram_mb:.1f}")
print(f"best_epoch:               {best_epoch}")
print(f"num_epochs:               {EPOCH_BUDGET}")
print(f"num_params_M:             {num_params / 1e6:.3f}")
