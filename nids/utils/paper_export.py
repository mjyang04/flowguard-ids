from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from nids.utils.io import load_json
from nids.utils.visualization import (
    plot_latency_tradeoff,
    plot_model_comparison,
    plot_shap_top_features,
)

SUMMARY_COLUMNS = [
    "dataset_pair",
    "train_dataset",
    "test_dataset",
    "model",
    "imbalance_strategy",
    "selection_metric",
    "selection_metric_value",
    "best_epoch",
    "best_metric",
    "fit_seconds",
    "history_points",
    "input_dim",
    "num_classes",
    "feature_count",
    "total_params",
    "trainable_params",
    "accuracy",
    "macro_f1",
    "avg_attack_recall",
    "attack_macro_precision",
    "benign_false_alarm_rate",
    "attack_miss_rate",
    "pr_auc",
    "roc_auc",
    "best_f1",
    "best_f1_threshold",
    "recall_at_far_1pct",
    "threshold_at_far_1pct",
    "recall_at_far_5pct",
    "threshold_at_far_5pct",
    "mean_latency_ms",
    "p50_latency_ms",
    "p99_latency_ms",
    "throughput",
    "run_name",
    "run_dir",
    "report_path",
    "config_path",
    "manifest_path",
]
BEST_COLUMNS = [
    "dataset_pair",
    "rank_within_dataset_pair",
    "model",
    "selection_metric",
    "selection_metric_value",
    "avg_attack_recall",
    "pr_auc",
    "roc_auc",
    "benign_false_alarm_rate",
    "mean_latency_ms",
    "run_name",
    "run_dir",
    "report_path",
]
EFFICIENCY_COLUMNS = [
    "dataset_pair",
    "model",
    "avg_attack_recall",
    "pr_auc",
    "mean_latency_ms",
    "p50_latency_ms",
    "p99_latency_ms",
    "throughput",
    "run_name",
]


def _safe_float(value: Any, default: float = np.nan) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _extract_train_test(
    manifest: dict[str, Any],
    report: dict[str, Any],
    report_path: Path,
) -> tuple[str, str]:
    train_dataset = str(manifest.get("train_dataset") or "")
    test_dataset = str(manifest.get("test_dataset") or "")
    if train_dataset and test_dataset:
        return train_dataset, test_dataset

    for candidate in [report.get("data_file"), report.get("test_data"), str(report_path)]:
        if not candidate:
            continue
        match = re.search(r"(?:cross_)?([a-zA-Z0-9_]+)_to_([a-zA-Z0-9_]+)\.npz", str(candidate))
        if match:
            return str(match.group(1)), str(match.group(2))

    for part in report_path.parts:
        if "_to_" not in part:
            continue
        left, right = part.split("_to_", maxsplit=1)
        if left and right:
            return left, right

    return "unknown", "unknown"


def _collect_report_rows(artifacts_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report_path in sorted(artifacts_root.rglob("report.json")):
        report = load_json(report_path)
        metrics = report.get("test_metrics") or report.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue

        run_dir = report_path.parent
        manifest_path = run_dir / "run_manifest.json"
        config_path = run_dir / "resolved_config.yaml"
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        config = _load_yaml(config_path)
        train_dataset, test_dataset = _extract_train_test(manifest, report, report_path)
        dataset_pair = f"{train_dataset}_to_{test_dataset}"

        training_summary = report.get("training_summary") or {}
        latency = report.get("latency") or {}
        history = training_summary.get("history") or []
        selection_metric = str(
            config.get("training", {}).get("selection_metric")
            or manifest.get("selection_metric")
            or "avg_attack_recall"
        )
        selection_metric_value = _safe_float(
            metrics.get(selection_metric, training_summary.get("best_metric"))
        )
        feature_names = report.get("feature_names") or []

        rows.append(
            {
                "dataset_pair": dataset_pair,
                "train_dataset": train_dataset,
                "test_dataset": test_dataset,
                "model": str(report.get("model") or manifest.get("model") or "unknown"),
                "imbalance_strategy": str(
                    report.get("imbalance_strategy") or manifest.get("imbalance_strategy") or "unknown"
                ),
                "selection_metric": selection_metric,
                "selection_metric_value": selection_metric_value,
                "best_epoch": int(training_summary.get("best_epoch", 0) or 0),
                "best_metric": _safe_float(training_summary.get("best_metric")),
                "fit_seconds": _safe_float(training_summary.get("fit_seconds")),
                "history_points": int(len(history)),
                "input_dim": int(report.get("input_dim", 0) or 0),
                "num_classes": int(report.get("num_classes", 0) or 0),
                "feature_count": int(len(feature_names)),
                "total_params": int(report.get("total_params", 0) or 0),
                "trainable_params": int(report.get("trainable_params", 0) or 0),
                "accuracy": _safe_float(metrics.get("accuracy")),
                "macro_f1": _safe_float(metrics.get("macro_f1")),
                "avg_attack_recall": _safe_float(metrics.get("avg_attack_recall")),
                "attack_macro_precision": _safe_float(metrics.get("attack_macro_precision")),
                "benign_false_alarm_rate": _safe_float(metrics.get("benign_false_alarm_rate")),
                "attack_miss_rate": _safe_float(metrics.get("attack_miss_rate")),
                "pr_auc": _safe_float(metrics.get("pr_auc")),
                "roc_auc": _safe_float(metrics.get("roc_auc")),
                "best_f1": _safe_float(metrics.get("best_f1")),
                "best_f1_threshold": _safe_float(metrics.get("best_f1_threshold")),
                "recall_at_far_1pct": _safe_float(metrics.get("recall_at_far_1pct")),
                "threshold_at_far_1pct": _safe_float(metrics.get("threshold_at_far_1pct")),
                "recall_at_far_5pct": _safe_float(metrics.get("recall_at_far_5pct")),
                "threshold_at_far_5pct": _safe_float(metrics.get("threshold_at_far_5pct")),
                "mean_latency_ms": _safe_float(latency.get("mean_latency_ms")),
                "p50_latency_ms": _safe_float(latency.get("p50_latency_ms")),
                "p99_latency_ms": _safe_float(latency.get("p99_latency_ms")),
                "throughput": _safe_float(latency.get("throughput")),
                "run_name": str(manifest.get("run_name") or run_dir.name),
                "run_dir": str(run_dir),
                "report_path": str(report_path),
                "config_path": str(config_path) if config_path.exists() else "",
                "manifest_path": str(manifest_path) if manifest_path.exists() else "",
            }
        )

    return rows


def _rank_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        ranked = df.copy()
        ranked["rank_within_dataset_pair"] = pd.Series(dtype="int64")
        return ranked

    ranked = df.assign(
        _selection_metric_value=df["selection_metric_value"].fillna(-np.inf),
        _avg_attack_recall=df["avg_attack_recall"].fillna(-np.inf),
        _pr_auc=df["pr_auc"].fillna(-np.inf),
        _roc_auc=df["roc_auc"].fillna(-np.inf),
        _far=df["benign_false_alarm_rate"].fillna(np.inf),
        _latency=df["mean_latency_ms"].replace(0.0, np.nan).fillna(np.inf),
    ).sort_values(
        by=[
            "dataset_pair",
            "_selection_metric_value",
            "_avg_attack_recall",
            "_pr_auc",
            "_roc_auc",
            "_far",
            "_latency",
            "model",
            "run_name",
        ],
        ascending=[True, False, False, False, False, True, True, True, True],
        kind="mergesort",
    )
    ranked["rank_within_dataset_pair"] = ranked.groupby("dataset_pair").cumcount() + 1
    return ranked.drop(
        columns=[
            "_selection_metric_value",
            "_avg_attack_recall",
            "_pr_auc",
            "_roc_auc",
            "_far",
            "_latency",
        ]
    )


def _collect_shap_ranking_rows(artifacts_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ranking_path in sorted(artifacts_root.rglob("feature_ranking.json")):
        payload = load_json(ranking_path)
        ranking = payload.get("ranking") or []
        if not isinstance(ranking, list):
            continue

        reference_pair = ranking_path.parent.parent.name if ranking_path.parent.parent != artifacts_root else "unknown"
        reference_model = ranking_path.parent.name
        reference_id = f"{reference_pair} | {reference_model}"
        for item in ranking:
            rows.append(
                {
                    "reference_pair": reference_pair,
                    "reference_model": reference_model,
                    "reference_id": reference_id,
                    "rank": int(item.get("rank", 0) or 0),
                    "feature": str(item.get("feature") or ""),
                    "index": int(item.get("index", 0) or 0),
                    "importance": _safe_float(item.get("importance")),
                    "feature_ranking_path": str(ranking_path),
                }
            )
    return rows


def _collect_shap_topk_rows(artifacts_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for summary_path in sorted(artifacts_root.rglob("top_k_summary.json")):
        payload = load_json(summary_path)
        features = payload.get("top_k_features") or []
        importance = payload.get("top_k_importance") or []
        if not isinstance(features, list) or not isinstance(importance, list):
            continue

        reference_pair = summary_path.parent.parent.name if summary_path.parent.parent != artifacts_root else "unknown"
        reference_model = summary_path.parent.name
        for idx, (feature, score) in enumerate(zip(features, importance, strict=False), start=1):
            rows.append(
                {
                    "reference_pair": reference_pair,
                    "reference_model": reference_model,
                    "rank": idx,
                    "feature": str(feature),
                    "importance": _safe_float(score),
                    "top_k_summary_path": str(summary_path),
                }
            )
    return rows


def _write_csv(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(path, index=False, encoding="utf-8")
        return
    df.loc[:, columns].to_csv(path, index=False, encoding="utf-8")


def _build_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["dataset_pair"])

    metric_columns = [
        "avg_attack_recall",
        "pr_auc",
        "roc_auc",
        "benign_false_alarm_rate",
        "mean_latency_ms",
    ]
    pivot = df.pivot_table(index="dataset_pair", columns="model", values=metric_columns, aggfunc="first")
    pivot = pivot.sort_index(axis=1, level=[0, 1])
    pivot.columns = [f"{metric}__{model}" for metric, model in pivot.columns]
    return pivot.reset_index()


def export_paper_results(artifacts_root: str | Path, output_dir: str | Path) -> dict[str, str]:
    artifacts_root = Path(artifacts_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(_collect_report_rows(artifacts_root))
    if summary_df.empty:
        summary_df = pd.DataFrame(columns=SUMMARY_COLUMNS)
    else:
        summary_df = _rank_summary(summary_df).reset_index(drop=True)

    summary_csv = output_dir / "paper_results_summary.csv"
    best_csv = output_dir / "paper_best_runs.csv"
    efficiency_csv = output_dir / "paper_efficiency_summary.csv"
    pivot_csv = output_dir / "paper_metrics_pivot.csv"
    _write_csv(summary_df, summary_csv, SUMMARY_COLUMNS + ["rank_within_dataset_pair"])

    best_df = summary_df.loc[summary_df.get("rank_within_dataset_pair", pd.Series(dtype=int)) == 1].copy()
    _write_csv(best_df, best_csv, BEST_COLUMNS)
    _write_csv(summary_df, efficiency_csv, EFFICIENCY_COLUMNS)

    pivot_df = _build_pivot_table(summary_df)
    pivot_df.to_csv(pivot_csv, index=False, encoding="utf-8")

    shap_ranking_df = pd.DataFrame(_collect_shap_ranking_rows(artifacts_root))
    shap_topk_df = pd.DataFrame(_collect_shap_topk_rows(artifacts_root))
    shap_ranking_csv = output_dir / "paper_shap_feature_ranking.csv"
    shap_topk_csv = output_dir / "paper_shap_top_k.csv"
    _write_csv(
        shap_ranking_df,
        shap_ranking_csv,
        [
            "reference_pair",
            "reference_model",
            "reference_id",
            "rank",
            "feature",
            "index",
            "importance",
            "feature_ranking_path",
        ],
    )
    _write_csv(
        shap_topk_df,
        shap_topk_csv,
        [
            "reference_pair",
            "reference_model",
            "rank",
            "feature",
            "importance",
            "top_k_summary_path",
        ],
    )

    if not summary_df.empty:
        plot_model_comparison(summary_df.to_dict(orient="records"), figures_dir / "model_comparison.png")
        plot_latency_tradeoff(summary_df.to_dict(orient="records"), figures_dir / "latency_tradeoff.png")
    if not shap_ranking_df.empty:
        top_rows = (
            shap_ranking_df.sort_values(["reference_id", "rank"], ascending=[True, True])
            .groupby("reference_id", group_keys=False)
            .head(10)
        )
        plot_shap_top_features(top_rows.to_dict(orient="records"), figures_dir / "shap_top_features.png")

    return {
        "summary_csv": str(summary_csv),
        "best_csv": str(best_csv),
        "efficiency_csv": str(efficiency_csv),
        "pivot_csv": str(pivot_csv),
        "shap_ranking_csv": str(shap_ranking_csv),
        "shap_topk_csv": str(shap_topk_csv),
        "figures_dir": str(figures_dir),
    }
