from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from nids.utils.visualization import plot_training_curves

from nids.utils.paper_export import export_paper_results


def _write_json(path: Path, payload: dict) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _create_run(
    root: Path,
    *,
    train_dataset: str,
    test_dataset: str,
    model: str,
    run_name: str,
    avg_attack_recall: float,
    pr_auc: float,
    roc_auc: float,
    benign_false_alarm_rate: float,
    mean_latency_ms: float,
    seed: int | None = None,
    recall_at_far_1pct: float | None = None,
) -> Path:
    base = root / f"seed{seed}" if seed is not None else root
    run_dir = base / f"{train_dataset}_to_{test_dataset}" / model / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_dir / "report.json",
        {
            "model": model,
            "imbalance_strategy": "class_weights",
            "data_file": f"data/processed/cross_{train_dataset}_to_{test_dataset}.npz",
            "input_dim": 55,
            "num_classes": 2,
            "total_params": 150000,
            "trainable_params": 150000,
            "feature_names": ["f0", "f1", "f2"],
            "training_summary": {
                "best_metric": pr_auc,
                "best_epoch": 7,
                "best_model_path": str(run_dir / "best_model.pt"),
                "history": [
                    {"epoch": 1, "train_loss": 0.8, "val_loss": 0.7, "val_metric": 0.75, "learning_rate": 1e-3},
                    {"epoch": 2, "train_loss": 0.6, "val_loss": 0.5, "val_metric": 0.82, "learning_rate": 1e-3},
                ],
                "fit_seconds": 12.5,
            },
            "test_metrics": {
                "accuracy": 0.95,
                "macro_f1": 0.93,
                "avg_attack_recall": avg_attack_recall,
                "attack_macro_precision": 0.94,
                "benign_false_alarm_rate": benign_false_alarm_rate,
                "attack_miss_rate": 1.0 - avg_attack_recall,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
                "best_f1": 0.94,
                "best_f1_threshold": 0.52,
                "recall_at_far_1pct": recall_at_far_1pct if recall_at_far_1pct is not None else 0.86,
                "threshold_at_far_1pct": 0.73,
                "recall_at_far_5pct": 0.91,
                "threshold_at_far_5pct": 0.61,
                "confusion_matrix": [[95, 5], [7, 93]],
            },
            "latency": {
                "mean_latency_ms": mean_latency_ms,
                "p50_latency_ms": mean_latency_ms * 0.95,
                "p99_latency_ms": mean_latency_ms * 1.15,
                "throughput": 500.0,
            },
        },
    )
    manifest_payload = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "started_at": "2026-04-07T10:00:00",
        "finished_at": "2026-04-07T10:10:00",
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
        "model": model,
        "imbalance_strategy": "class_weights",
        "report_path": str(run_dir / "report.json"),
    }
    if seed is not None:
        manifest_payload["seed"] = int(seed)
    _write_json(run_dir / "run_manifest.json", manifest_payload)
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump({"training": {"selection_metric": "avg_attack_recall"}}, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "training_history.csv").write_text(
        "epoch,train_loss,val_loss,val_metric,learning_rate\n"
        "1,0.8,0.7,0.75,0.001\n"
        "2,0.6,0.5,0.82,0.001\n",
        encoding="utf-8",
    )
    return run_dir


def test_plot_training_curves_writes_svg_and_png(tmp_path: Path) -> None:
    history = [
        {"epoch": 1, "train_loss": 0.9, "val_loss": 0.8, "val_metric": 0.75},
        {"epoch": 2, "train_loss": 0.7, "val_loss": 0.6, "val_metric": 0.84},
    ]

    output_path = tmp_path / "training_curves.png"
    plot_training_curves(history, output_path)

    assert output_path.exists()
    assert output_path.with_suffix(".svg").exists()


def test_export_paper_results_generates_csv_and_figures(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    _create_run(
        artifacts_root,
        train_dataset="cicids2017",
        test_dataset="unsw_nb15",
        model="cnn_bilstm_se",
        run_name="20260407_100000_cnn_bilstm_se_class_weights",
        avg_attack_recall=0.92,
        pr_auc=0.95,
        roc_auc=0.97,
        benign_false_alarm_rate=0.03,
        mean_latency_ms=1.8,
    )
    _create_run(
        artifacts_root,
        train_dataset="cicids2017",
        test_dataset="unsw_nb15",
        model="random_forest",
        run_name="20260407_100000_random_forest_class_weights",
        avg_attack_recall=0.88,
        pr_auc=0.91,
        roc_auc=0.94,
        benign_false_alarm_rate=0.05,
        mean_latency_ms=0.6,
    )

    shap_dir = artifacts_root / "shap" / "cicids2017_to_cicids2017" / "cnn_bilstm_se"
    _write_json(
        shap_dir / "feature_ranking.json",
        {
            "reference_model_name": "cnn_bilstm_se",
            "n_samples": 128,
            "background_size": 32,
            "ranking": [
                {"rank": 1, "feature": "flow_duration", "index": 12, "importance": 0.42},
                {"rank": 2, "feature": "packet_len_mean", "index": 7, "importance": 0.31},
            ],
        },
    )
    _write_json(
        shap_dir / "top_k_summary.json",
        {
            "top_k_features": ["flow_duration", "packet_len_mean"],
            "top_k_importance": [0.42, 0.31],
        },
    )

    output_dir = tmp_path / "paper_exports"
    export_paper_results(artifacts_root=artifacts_root, output_dir=output_dir)

    summary_csv = output_dir / "paper_results_summary.csv"
    best_csv = output_dir / "paper_best_runs.csv"
    shap_csv = output_dir / "paper_shap_feature_ranking.csv"
    comparison_svg = output_dir / "figures" / "model_comparison.svg"
    tradeoff_png = output_dir / "figures" / "latency_tradeoff.png"

    assert summary_csv.exists()
    assert best_csv.exists()
    assert shap_csv.exists()
    assert comparison_svg.exists()
    assert tradeoff_png.exists()
    assert "<svg" in comparison_svg.read_text(encoding="utf-8")

    with summary_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2
    assert rows[0]["dataset_pair"] == "cicids2017_to_unsw_nb15"
    assert rows[0]["selection_metric"] == "avg_attack_recall"

    with best_csv.open(encoding="utf-8", newline="") as f:
        best_rows = list(csv.DictReader(f))

    assert len(best_rows) == 1
    assert best_rows[0]["model"] == "cnn_bilstm_se"
    assert best_rows[0]["rank_within_dataset_pair"] == "1"


def test_export_paper_results_aggregates_multi_seed(tmp_path: Path) -> None:
    artifacts_root = tmp_path / "artifacts"
    recalls = {42: 0.82, 43: 0.85, 44: 0.88}
    for seed, recall in recalls.items():
        _create_run(
            artifacts_root,
            train_dataset="cicids2017",
            test_dataset="unsw_nb15",
            model="cnn_bilstm_se",
            run_name=f"20260407_100000_cnn_bilstm_se_class_weights_seed{seed}",
            avg_attack_recall=recall,
            pr_auc=0.95,
            roc_auc=0.97,
            benign_false_alarm_rate=0.03,
            mean_latency_ms=1.8,
            seed=seed,
            recall_at_far_1pct=recall,
        )

    output_dir = tmp_path / "paper_exports"
    result = export_paper_results(artifacts_root=artifacts_root, output_dir=output_dir)

    seed_agg_csv = Path(result["seed_aggregated_csv"])
    assert seed_agg_csv.exists()

    with seed_agg_csv.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 1
    row = rows[0]
    assert row["model"] == "cnn_bilstm_se"
    assert int(row["n_seeds"]) == 3
    mean_recall = float(row["recall_at_far_1pct_mean"])
    std_recall = float(row["recall_at_far_1pct_std"])
    assert mean_recall == pytest.approx(sum(recalls.values()) / 3, abs=1e-6)
    assert std_recall > 0

    summary_csv = Path(result["summary_csv"])
    with summary_csv.open(encoding="utf-8", newline="") as f:
        summary_rows = list(csv.DictReader(f))
    seeds_seen = sorted({int(r["seed"]) for r in summary_rows})
    assert seeds_seen == [42, 43, 44]
