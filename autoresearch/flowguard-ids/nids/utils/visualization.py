from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, values_format="d", colorbar=False)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_curves(history: list[dict], output_path: str | Path) -> None:
    epochs = [x["epoch"] for x in history]
    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]
    val_metric = [x["val_metric"] for x in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, val_metric, label="val_metric")
    axes[1].set_xlabel("epoch")
    axes[1].set_title("Validation Metric")
    axes[1].legend()

    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_nids_key_metrics(metrics: dict, output_path: str | Path) -> None:
    keys = [
        "avg_attack_recall",
        "attack_macro_precision",
        "benign_false_alarm_rate",
        "attack_miss_rate",
    ]
    labels = [
        "Attack Recall",
        "Attack Precision",
        "False Alarm Rate",
        "Attack Miss Rate",
    ]
    values = [float(metrics.get(k, 0.0)) for k in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("NIDS Key Metrics")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.02, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_split_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    output_path: str | Path,
    benign_class: int = 0,
) -> None:
    y_train = np.asarray(y_train)
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    splits = [("Train", y_train), ("Val", y_val), ("Test", y_test)]

    benign_counts = [int((labels == benign_class).sum()) for _, labels in splits]
    attack_counts = [int((labels != benign_class).sum()) for _, labels in splits]

    fig, ax = plt.subplots(figsize=(7, 4))
    names = [name for name, _ in splits]
    ax.bar(names, benign_counts, label="Benign", color="#1f77b4")
    ax.bar(names, attack_counts, bottom=benign_counts, label="Attack", color="#d62728")
    ax.set_ylabel("Samples")
    ax.set_title("Dataset Split Distribution")
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
