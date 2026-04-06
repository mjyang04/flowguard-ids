"""
FlowGuard IDS autoresearch training script.
Edit this file freely — it is the only file you should modify.

Baseline: CNN-BiLSTM-SE with default.yaml settings.
Goal: maximize val_avg_attack_recall while keeping both test metrics strong.

Usage (inside Docker container, from /workspace/autoresearch):
    python train.py
    python train.py > run.log 2>&1
"""

from __future__ import annotations

import copy
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight

from prepare import (
    BATCH_SIZE,
    MAX_EPOCHS,
    INPUT_DIM,
    NUM_CLASSES,
    evaluate_nids,
    make_dataloaders,
)

# ---------------------------------------------------------------------------
# Hyperparameters — edit freely
# ---------------------------------------------------------------------------

# Architecture  (baseline matches configs/default.yaml)
CONV_CHANNELS: List[int]    = [64, 128]
CONV_KERNEL_SIZES: List[int] = [3, 3]
CONV_POOL_SIZES: List[int]  = [2, 2]
USE_SE: bool       = True
USE_ATTENTION: bool = False
LSTM_HIDDEN: int   = 128
LSTM_LAYERS: int   = 2
BIDIRECTIONAL: bool = True
DROPOUT: float     = 0.3

# Optimisation  (baseline matches configs/default.yaml)
LEARNING_RATE: float  = 1e-3
WEIGHT_DECAY: float   = 1e-4
GRADIENT_CLIP: float  = 1.0
USE_AMP: bool         = True
OPTIMIZER: str        = "adamw"     # "adam" | "adamw" | "sgd"

# Early stopping (mirrors nids.training.callbacks.EarlyStopping)
EARLY_STOPPING_PATIENCE: int   = 5
EARLY_STOPPING_DELTA: float    = 1e-4

# Scheduler
SCHEDULER: str          = "plateau"  # "plateau" | "cosine" | "none"
SCHEDULER_PATIENCE: int = 2
SCHEDULER_FACTOR: float = 0.5
MIN_LR: float           = 1e-6

# Class imbalance  (matches trainer default: class_weights strategy)
IMBALANCE_STRATEGY: str = "class_weights"  # "class_weights" | "none"

# Batch size override (None = use prepare.BATCH_SIZE)
BATCH_SIZE_OVERRIDE: int | None = None

# ---------------------------------------------------------------------------
# Model (self-contained copy of nids.models.cnn_bilstm_se.CNNBiLSTMSE)
# Modify this section for architecture experiments.
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int, dropout: float):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding=kernel // 2)
        self.bn   = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(F.relu(self.bn(self.conv(x)))))


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        s = torch.relu(self.fc1(x.mean(dim=-1)))
        return x * torch.sigmoid(self.fc2(s)).unsqueeze(-1)


class AttentionPooling(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (L, B, H)
        w = torch.softmax(self.score(x), dim=0)
        return (w * x).sum(dim=0)


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
        cnn: list[nn.Module] = []
        in_ch = 1
        for out_ch, k, p in zip(conv_channels, conv_kernel_sizes, conv_pool_sizes):
            cnn.append(ConvBlock(in_ch, out_ch, k, p, dropout))
            if use_se:
                cnn.append(SqueezeExcitation(out_ch))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn)

        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=False,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.attn = AttentionPooling(out_dim) if use_attention else None
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim // 2, num_classes if num_classes > 2 else 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                   # (B, F) -> (B, 1, F)
        x = self.cnn(x)                      # (B, C, L')
        x = x.permute(2, 0, 1)              # (L', B, C)
        out, _ = self.lstm(x)               # (L', B, H)
        pooled = self.attn(out) if self.attn else out.mean(0)
        logits = self.classifier(pooled)
        return logits.squeeze(-1) if logits.shape[-1] == 1 else logits


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

bs = BATCH_SIZE_OVERRIDE or BATCH_SIZE
train_loader, val_loader, test_single_loader, test_cross_loader, y_train_arr = (
    make_dataloaders(batch_size=bs)
)
print(f"Train: {len(train_loader.dataset):,}  "
      f"Val: {len(val_loader.dataset):,}  "
      f"Test(single): {len(test_single_loader.dataset):,}  "
      f"Test(cross): {len(test_cross_loader.dataset):,}")

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

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e6:.3f}M")

# Loss — mirrors nids.training.trainer._build_criterion
if NUM_CLASSES == 2:
    pos_weight = None
    if IMBALANCE_STRATEGY == "class_weights":
        weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train_arr)
        pos_weight = torch.tensor([weights[1] / weights[0]], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    weight_tensor = None
    if IMBALANCE_STRATEGY == "class_weights":
        classes = np.unique(y_train_arr)
        weights = compute_class_weight("balanced", classes=classes, y=y_train_arr)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

# Optimizer
if OPTIMIZER == "adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == "adamw":
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
elif OPTIMIZER == "sgd":
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE,
                                momentum=0.9, weight_decay=WEIGHT_DECAY)
else:
    raise ValueError(f"Unknown optimizer: {OPTIMIZER}")

# Scheduler — mirrors nids.training.trainer (ReduceLROnPlateau on val recall)
scheduler = None
if SCHEDULER == "plateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE, min_lr=MIN_LR,
    )
elif SCHEDULER == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS, eta_min=MIN_LR,
    )

scaler = torch.amp.GradScaler("cuda") if USE_AMP and device.type == "cuda" else None

print(f"Optimizer: {OPTIMIZER}  lr={LEARNING_RATE}  wd={WEIGHT_DECAY}  "
      f"scheduler={SCHEDULER}  imbalance={IMBALANCE_STRATEGY}")

# ---------------------------------------------------------------------------
# Training loop — mirrors nids.training.trainer.Trainer.fit
# ---------------------------------------------------------------------------

t_train_start = time.time()
best_val_recall = 0.0
best_epoch = 0
best_state_dict = None
total_train_secs = 0.0
es_counter = 0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    t0 = time.time()

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                logits = model(features)
                loss = criterion(logits, labels.float() if NUM_CLASSES == 2 else labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(features)
            loss = criterion(logits, labels.float() if NUM_CLASSES == 2 else labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()

        epoch_loss += loss.item()

    dt = time.time() - t0
    total_train_secs += dt
    avg_loss = epoch_loss / max(1, len(train_loader))

    val_metrics = evaluate_nids(model, val_loader, device, NUM_CLASSES, use_amp=USE_AMP)
    val_recall = val_metrics["avg_attack_recall"]

    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_recall)
        else:
            scheduler.step()

    if val_recall > best_val_recall + EARLY_STOPPING_DELTA:
        best_val_recall = val_recall
        best_epoch = epoch
        best_state_dict = copy.deepcopy(model.state_dict())
        es_counter = 0
    else:
        es_counter += 1

    lr_now = optimizer.param_groups[0]["lr"]
    print(
        f"\repoch {epoch:03d}/{MAX_EPOCHS} | loss={avg_loss:.4f} | "
        f"val_recall={val_recall:.4f} | best={best_val_recall:.4f}(ep{best_epoch}) | "
        f"lr={lr_now:.1e} | es={es_counter}/{EARLY_STOPPING_PATIENCE} | dt={dt:.1f}s    ",
        end="", flush=True,
    )

    if es_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch}")
        break

print()

# ---------------------------------------------------------------------------
# Restore best model and evaluate on both test sets
# ---------------------------------------------------------------------------

if best_state_dict is not None:
    model.load_state_dict(best_state_dict)
    print(f"Restored best model from epoch {best_epoch}")

test_single = evaluate_nids(model, test_single_loader, device, NUM_CLASSES, use_amp=USE_AMP)
test_cross  = evaluate_nids(model, test_cross_loader, device, NUM_CLASSES, use_amp=USE_AMP)
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if device.type == "cuda" else 0.0
t_end = time.time()

print("---")
print(f"val_avg_attack_recall:           {best_val_recall:.6f}")
print(f"test_single_avg_attack_recall:   {test_single['avg_attack_recall']:.6f}")
print(f"test_single_false_alarm_rate:    {test_single['benign_false_alarm_rate']:.6f}")
print(f"test_single_accuracy:            {test_single['accuracy']:.6f}")
print(f"test_cross_avg_attack_recall:    {test_cross['avg_attack_recall']:.6f}")
print(f"test_cross_false_alarm_rate:     {test_cross['benign_false_alarm_rate']:.6f}")
print(f"test_cross_accuracy:             {test_cross['accuracy']:.6f}")
print(f"training_seconds:                {total_train_secs:.1f}")
print(f"total_seconds:                   {t_end - t_start:.1f}")
print(f"peak_vram_mb:                    {peak_vram_mb:.1f}")
print(f"best_epoch:                      {best_epoch}")
print(f"num_epochs:                      {epoch}")
print(f"num_params_M:                    {num_params / 1e6:.3f}")
