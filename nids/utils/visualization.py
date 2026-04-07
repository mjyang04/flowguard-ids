from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

PUBLICATION_DPI = 400
PUBLICATION_RC = {
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "svg.fonttype": "none",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "lines.linewidth": 1.8,
}
PALETTE = [
    "#0F4C81",
    "#E07A5F",
    "#3D9970",
    "#7A5195",
    "#EF5675",
    "#2F4B7C",
]
METRIC_LABELS = {
    "avg_attack_recall": "Attack Recall",
    "attack_macro_precision": "Attack Precision",
    "benign_false_alarm_rate": "False Alarm Rate",
    "attack_miss_rate": "Attack Miss Rate",
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
}


def save_figure_bundle(fig: plt.Figure, output_path: str | Path, dpi: int = PUBLICATION_DPI) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = path.with_suffix("")
    formats = {"svg"}
    if path.suffix:
        formats.add(path.suffix.lower().lstrip("."))
    else:
        formats.add("png")

    for fmt in sorted(formats):
        target = stem.with_suffix(f".{fmt}")
        save_kwargs = {
            "format": fmt,
            "bbox_inches": "tight",
            "facecolor": "white",
        }
        if fmt != "svg":
            save_kwargs["dpi"] = dpi
        fig.savefig(target, **save_kwargs)
    plt.close(fig)


def _style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, linestyle="--", linewidth=0.6, alpha=0.22)
    ax.set_axisbelow(True)


def _format_dataset_pair_label(dataset_pair: str) -> str:
    return dataset_pair.replace("_to_", " -> ").replace("_", "-")


def plot_confusion_matrix(cm: np.ndarray, labels: list[str], output_path: str | Path) -> None:
    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(7.0, 5.8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(ax=ax, values_format="d", colorbar=False, cmap="Blues")
        ax.set_title("Confusion Matrix")
        ax.grid(False)
        fig.tight_layout()
        save_figure_bundle(fig, output_path)


def plot_training_curves(history: list[dict], output_path: str | Path) -> None:
    epochs = [x["epoch"] for x in history]
    train_loss = [np.nan if x["train_loss"] is None else x["train_loss"] for x in history]
    val_loss = [np.nan if x["val_loss"] is None else x["val_loss"] for x in history]
    val_metric = [np.nan if x["val_metric"] is None else x["val_metric"] for x in history]
    val_metric_array = np.asarray(val_metric, dtype=np.float64)
    if np.all(np.isnan(val_metric_array)):
        best_idx = 0
    else:
        best_idx = int(np.nanargmax(val_metric_array))
    best_epoch = epochs[best_idx]

    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

        axes[0].plot(epochs, train_loss, marker="o", markersize=3.5, color=PALETTE[0], label="Train Loss")
        axes[0].plot(epochs, val_loss, marker="s", markersize=3.5, color=PALETTE[1], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training and Validation Loss")
        _style_axis(axes[0])
        axes[0].legend()

        axes[1].plot(epochs, val_metric, marker="o", markersize=3.5, color=PALETTE[2], label="Val Metric")
        axes[1].axvline(best_epoch, color=PALETTE[3], linestyle="--", linewidth=1.0, label=f"Best Epoch = {best_epoch}")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Score")
        axes[1].set_title("Validation Metric")
        _style_axis(axes[1])
        axes[1].legend()

        fig.tight_layout()
        save_figure_bundle(fig, output_path)


def plot_nids_key_metrics(metrics: dict, output_path: str | Path) -> None:
    keys = [
        "avg_attack_recall",
        "attack_macro_precision",
        "benign_false_alarm_rate",
        "attack_miss_rate",
        "pr_auc",
        "roc_auc",
    ]
    labels = [METRIC_LABELS[k] for k in keys]
    values = [float(metrics.get(k, 0.0)) for k in keys]

    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(10.8, 4.2))
        bars = ax.bar(labels, values, color=PALETTE[: len(labels)], edgecolor="#2B2B2B", linewidth=0.4)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Score")
        ax.set_title("Key Detection Metrics")
        _style_axis(ax)
        ax.tick_params(axis="x", rotation=15)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.02,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        fig.tight_layout()
        save_figure_bundle(fig, output_path)


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

    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        names = [name for name, _ in splits]
        ax.bar(names, benign_counts, label="Benign", color=PALETTE[0], edgecolor="#2B2B2B", linewidth=0.4)
        ax.bar(
            names,
            attack_counts,
            bottom=benign_counts,
            label="Attack",
            color=PALETTE[1],
            edgecolor="#2B2B2B",
            linewidth=0.4,
        )
        ax.set_ylabel("Samples")
        ax.set_title("Dataset Split Distribution")
        _style_axis(ax)
        ax.legend()
        fig.tight_layout()
        save_figure_bundle(fig, output_path)


def plot_model_comparison(summary_rows: Iterable[dict], output_path: str | Path) -> None:
    rows = list(summary_rows)
    if not rows:
        return

    dataset_pairs = sorted({str(row["dataset_pair"]) for row in rows})
    models = sorted({str(row["model"]) for row in rows})
    lookup = {(str(row["dataset_pair"]), str(row["model"])): row for row in rows}
    metric_specs = [
        ("avg_attack_recall", "Attack Recall"),
        ("pr_auc", "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("benign_false_alarm_rate", "False Alarm Rate"),
    ]

    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), sharex=True)
        axes = axes.flatten()
        x = np.arange(len(models), dtype=np.float64)
        width = min(0.72 / max(1, len(dataset_pairs)), 0.28)

        for ax, (metric_key, title) in zip(axes, metric_specs):
            for idx, dataset_pair in enumerate(dataset_pairs):
                values = [
                    float(lookup.get((dataset_pair, model), {}).get(metric_key, np.nan))
                    for model in models
                ]
                offsets = x + (idx - (len(dataset_pairs) - 1) / 2.0) * width
                ax.bar(
                    offsets,
                    values,
                    width=width,
                    color=PALETTE[idx % len(PALETTE)],
                    edgecolor="#2B2B2B",
                    linewidth=0.4,
                    label=_format_dataset_pair_label(dataset_pair),
                )
            ax.set_title(title)
            ax.set_ylim(0.0, 1.05)
            _style_axis(ax)

        axes[-2].set_xticks(x)
        axes[-2].set_xticklabels(models, rotation=15, ha="right")
        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(models, rotation=15, ha="right")
        axes[0].set_ylabel("Score")
        axes[2].set_ylabel("Score")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), bbox_to_anchor=(0.5, 1.02))
        fig.suptitle("Model Comparison Across Dataset Settings", y=1.07, fontsize=12, fontweight="semibold")
        fig.tight_layout()
        save_figure_bundle(fig, output_path)


def plot_latency_tradeoff(summary_rows: Iterable[dict], output_path: str | Path) -> None:
    rows = [row for row in summary_rows if float(row.get("mean_latency_ms", 0.0)) > 0.0]
    if not rows:
        return

    dataset_pairs = sorted({str(row["dataset_pair"]) for row in rows})

    with plt.rc_context(PUBLICATION_RC):
        fig, ax = plt.subplots(figsize=(7.6, 5.2))
        for idx, dataset_pair in enumerate(dataset_pairs):
            pair_rows = [row for row in rows if str(row["dataset_pair"]) == dataset_pair]
            x = np.array([float(row["mean_latency_ms"]) for row in pair_rows], dtype=np.float64)
            y = np.array([float(row.get("avg_attack_recall", 0.0)) for row in pair_rows], dtype=np.float64)
            sizes = np.array([float(row.get("pr_auc", 0.0)) for row in pair_rows], dtype=np.float64)
            ax.scatter(
                x,
                y,
                s=90.0 + sizes * 120.0,
                color=PALETTE[idx % len(PALETTE)],
                alpha=0.85,
                edgecolors="#2B2B2B",
                linewidths=0.5,
                label=_format_dataset_pair_label(dataset_pair),
            )
            for row in pair_rows:
                ax.annotate(
                    str(row["model"]),
                    (float(row["mean_latency_ms"]), float(row.get("avg_attack_recall", 0.0))),
                    xytext=(5, 5),
                    textcoords="offset points",
                    fontsize=8,
                )

        ax.set_xscale("log")
        ax.set_xlabel("Mean Latency (ms, log scale)")
        ax.set_ylabel("Attack Recall")
        ax.set_title("Latency vs Detection Performance")
        ax.set_ylim(0.0, 1.02)
        _style_axis(ax, grid_axis="both")
        ax.legend(loc="lower right")
        fig.tight_layout()
        save_figure_bundle(fig, output_path)


def plot_shap_top_features(ranking_rows: Iterable[dict], output_path: str | Path, top_n: int = 10) -> None:
    rows = list(ranking_rows)
    if not rows:
        return

    references = sorted({str(row["reference_id"]) for row in rows})
    n_refs = len(references)
    with plt.rc_context(PUBLICATION_RC):
        fig, axes = plt.subplots(n_refs, 1, figsize=(9.0, max(3.5, 2.9 * n_refs)), squeeze=False)
        for ax, reference_id in zip(axes.flatten(), references):
            ref_rows = [row for row in rows if str(row["reference_id"]) == reference_id]
            ref_rows = sorted(ref_rows, key=lambda item: int(item["rank"]))[:top_n]
            labels = [str(item["feature"]) for item in ref_rows][::-1]
            values = [float(item["importance"]) for item in ref_rows][::-1]
            ax.barh(labels, values, color=PALETTE[0], edgecolor="#2B2B2B", linewidth=0.4)
            ax.set_xlabel("Mean |SHAP Value|")
            ax.set_title(f"Top SHAP Features: {reference_id}")
            _style_axis(ax)
        fig.tight_layout()
        save_figure_bundle(fig, output_path)
