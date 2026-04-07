from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import ExperimentConfig, load_config, save_config
from nids.data.dataset import create_dataloaders
from nids.data.preprocessing import apply_smote, compute_class_weights
from nids.evaluation.metrics import compute_nids_metrics
from nids.models.classical import predict_binary_scores, train_random_forest, train_xgboost
from nids.models.registry import create_model
from nids.training.trainer import Trainer
from nids.utils.io import save_json
from nids.utils.logging import get_logger
from nids.utils.process import run_command
from nids.utils.reproducibility import seed_everything
from nids.utils.visualization import (
    plot_confusion_matrix,
    plot_nids_key_metrics,
    plot_split_distribution,
    plot_training_curves,
)

TOPK_MODEL_ALIASES = {"cnn_bilstm_se_fs": "cnn_bilstm_se_topk"}
TRAINABLE_DEEP_MODELS = ["cnn_bilstm", "cnn_bilstm_se", "cnn_bilstm_se_topk"]
CLASSICAL_MODELS = ["random_forest", "xgboost"]
TRAINABLE_MODELS = TRAINABLE_DEEP_MODELS + CLASSICAL_MODELS
CLI_MODEL_CHOICES = TRAINABLE_MODELS + list(TOPK_MODEL_ALIASES.keys())
IMBALANCE_STRATEGIES = ["auto", "class_weights", "smote", "none"]

MODEL_NAME_FOR_RUN = {
    "cnn_bilstm": "CNN-BiLSTM",
    "cnn_bilstm_se": "CNN-BiLSTM-SE",
    "cnn_bilstm_se_topk": "CNN-BiLSTM-SE-TopK",
    "random_forest": "RandomForest",
    "xgboost": "XGBoost",
}

STRATEGY_NAME_FOR_RUN = {
    "class_weights": "ClassWeights",
    "class_weight_balanced": "ClassWeights",
    "smote": "SMOTE",
    "none": "NoRebalance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NIDS models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-dataset", default=None, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--test-dataset", default=None, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--cross-dataset", action="store_true")
    parser.add_argument("--model", choices=CLI_MODEL_CHOICES, default=None)
    parser.add_argument("--models", default=None, help="Comma-separated model names")
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument(
        "--one-click",
        action="store_true",
        help="One-click training: all models + auto preprocessing + skip finished models",
    )
    parser.add_argument("--auto-preprocess", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full retraining of all selected models (disable skip-existing)",
    )
    parser.add_argument(
        "--imbalance-strategy",
        default="auto",
        choices=IMBALANCE_STRATEGIES,
        help="Imbalance handling strategy tag/control (auto/class_weights/smote/none)",
    )
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Optional extra tag appended to run directory name (for traceability)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted deep-model training from checkpoint_last.pt when available",
    )
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Explicit run directory to resume (single-model mode recommended)",
    )
    parser.add_argument(
        "--auto-feature-selection",
        action="store_true",
        help="Auto-run SHAP + Top-K feature selection when training cnn_bilstm_se_topk if reduced data is missing",
    )
    parser.add_argument("--shap-dir", default="artifacts/shap")
    parser.add_argument("--feature-selection-dir", default="artifacts/feature_selection")
    parser.add_argument("--reduced-data", default=None, help="Optional reduced npz path for cnn_bilstm_se_topk")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K used for feature selection")
    parser.add_argument(
        "--no-paper-artifacts",
        action="store_true",
        help="Disable automatic generation of paper-friendly figures/tables",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def _resolve_data_file(cfg: ExperimentConfig, args: argparse.Namespace) -> Path:
    if args.data_file:
        return Path(args.data_file)

    train_ds = args.train_dataset or cfg.data.train_dataset
    test_ds = args.test_dataset or cfg.data.test_dataset
    processed_dir = Path(cfg.data.processed_dir)
    if args.cross_dataset or train_ds != test_ds:
        return processed_dir / f"cross_{train_ds}_to_{test_ds}.npz"
    return processed_dir / train_ds / "data.npz"


def _load_npz(path: Path) -> dict:
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _canonical_model_name(name: str) -> str:
    return TOPK_MODEL_ALIASES.get(name, name)


def _resolve_models(args: argparse.Namespace, cfg: ExperimentConfig) -> list[str]:
    if args.all_models or args.ablation:
        return TRAINABLE_MODELS
    if args.models:
        picked = [_canonical_model_name(x.strip()) for x in args.models.split(",") if x.strip()]
        invalid = [x for x in picked if x not in TRAINABLE_MODELS]
        if invalid:
            raise ValueError(f"Unknown model(s): {invalid}. Supported: {TRAINABLE_MODELS}")
        # Deduplicate while preserving order (handles alias inputs).
        return list(dict.fromkeys(picked))
    if args.model:
        return [_canonical_model_name(args.model)]
    cfg_model_name = _canonical_model_name(cfg.model.name)
    if cfg_model_name == "cnn_bilstm_attention":
        raise ValueError(
            "cnn_bilstm_attention is kept in code but excluded from current training set. "
            "Use one of: cnn_bilstm, cnn_bilstm_se, cnn_bilstm_se_topk, random_forest, xgboost."
        )
    if cfg_model_name not in TRAINABLE_MODELS:
        raise ValueError(f"Unsupported default model in config: {cfg.model.name}")
    return [cfg_model_name]


def _sanitize_tag(text: str) -> str:
    allowed = []
    for ch in text.strip():
        if ch.isalnum():
            allowed.append(ch)
        elif ch in {"-", "_"}:
            allowed.append(ch)
        elif ch in {" ", "."}:
            allowed.append("-")
    normalized = "".join(allowed).strip("-_")
    return normalized or "run"


def _resolve_effective_imbalance_strategy(model_name: str, requested: str) -> str:
    if model_name in TRAINABLE_DEEP_MODELS:
        if requested == "auto":
            return "class_weights"
        return requested
    if model_name == "random_forest":
        if requested in {"auto", "class_weights"}:
            return "class_weight_balanced"
        if requested == "smote":
            return "smote"
        return "none"
    if model_name == "xgboost":
        if requested == "smote":
            return "smote"
        return "none"
    return "none"


def _build_run_name(model_name: str, strategy: str, run_tag: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_part = _sanitize_tag(MODEL_NAME_FOR_RUN.get(model_name, model_name))
    strategy_part = _sanitize_tag(STRATEGY_NAME_FOR_RUN.get(strategy, strategy))
    parts = [timestamp, model_part, strategy_part]
    if run_tag:
        parts.append(_sanitize_tag(run_tag))
    return "_".join(parts)


def _find_latest_report(model_root: Path) -> Path | None:
    runs_dir = model_root / "runs"
    if runs_dir.exists():
        candidates = sorted(
            [p for p in runs_dir.glob("*/report.json") if p.is_file()],
            key=lambda x: x.parent.name,
        )
        if candidates:
            return candidates[-1]
    legacy_report = model_root / "report.json"
    if legacy_report.exists():
        return legacy_report
    return None


def _find_latest_best_model(model_root: Path, suffix: str = ".pt") -> Path | None:
    runs_dir = model_root / "runs"
    if runs_dir.exists():
        candidates = sorted(
            [p for p in runs_dir.glob(f"*/best_model{suffix}") if p.is_file()],
            key=lambda x: x.parent.name,
        )
        if candidates:
            return candidates[-1]
    legacy_model = model_root / f"best_model{suffix}"
    if legacy_model.exists():
        return legacy_model
    return None


def _find_latest_checkpoint_run(model_root: Path) -> Path | None:
    runs_dir = model_root / "runs"
    if not runs_dir.exists():
        return None
    candidates = sorted(
        [p.parent for p in runs_dir.glob("*/checkpoint_last.pt") if p.is_file()],
        key=lambda x: x.name,
    )
    for run_dir in reversed(candidates):
        if not (run_dir / "report.json").exists():
            return run_dir
    return None


def _experiment_group_name(train_ds: str, test_ds: str) -> str | None:
    mapping = {
        ("cicids2017", "cicids2017"): "same_cicids",
        ("cicids2017", "unsw_nb15"): "cross_cic_to_unsw",
        ("unsw_nb15", "cicids2017"): "cross_unsw_to_cic",
    }
    return mapping.get((train_ds, test_ds))


def _resolve_shared_shap_dir(args: argparse.Namespace, cfg: ExperimentConfig) -> Path:
    if args.shap_dir == "artifacts/shap":
        return (
            Path(args.shap_dir)
            / "shared"
            / f"{cfg.shap.reference_train_dataset}_to_{cfg.shap.reference_test_dataset}"
            / cfg.shap.reference_model_name
        )
    return Path(args.shap_dir)


def _resolve_topk_paths(
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    train_ds: str,
    test_ds: str,
    top_k: int,
) -> tuple[Path, Path, Path, Path]:
    shared_shap_dir = _resolve_shared_shap_dir(args, cfg)
    if args.feature_selection_dir == "artifacts/feature_selection":
        fs_dir = (
            Path(args.feature_selection_dir)
            / f"{train_ds}_to_{test_ds}"
            / f"cnn_bilstm_se_topk_top{top_k}"
        )
    else:
        fs_dir = Path(args.feature_selection_dir) / f"top{top_k}"

    if args.reduced_data:
        reduced_data = Path(args.reduced_data)
    else:
        reduced_data = fs_dir / "reduced_data.npz"
    selected_idx = shared_shap_dir / f"top{top_k}_idx.npy"
    return shared_shap_dir, fs_dir, reduced_data, selected_idx


def _find_reference_model_path(
    cfg: ExperimentConfig,
    base_output_dir: Path,
    logger,
) -> Path | None:
    ref_train = cfg.shap.reference_train_dataset
    ref_test = cfg.shap.reference_test_dataset
    ref_model = cfg.shap.reference_model_name

    candidates: list[Path] = []
    exp_group = _experiment_group_name(ref_train, ref_test)
    if exp_group is not None:
        candidates.append(base_output_dir.parent / exp_group / ref_model)
        candidates.append(Path(cfg.runtime.output_dir) / "experiments" / exp_group / ref_model)
    candidates.append(Path(cfg.runtime.output_dir) / f"{ref_train}_to_{ref_test}" / ref_model)

    seen: set[Path] = set()
    for candidate_root in candidates:
        if candidate_root in seen:
            continue
        seen.add(candidate_root)
        model_path = _find_latest_best_model(candidate_root, suffix=".pt")
        if model_path is not None:
            logger.info("Using shared SHAP reference model: %s", model_path)
            return model_path
    return None


def _shared_topk_assets_valid(
    selected_idx_path: Path,
    feature_names_path: Path,
    top_k: int,
) -> bool:
    if not selected_idx_path.exists() or not feature_names_path.exists():
        return False
    try:
        selected_idx = np.load(selected_idx_path, allow_pickle=True)
        feature_names = np.load(feature_names_path, allow_pickle=True)
    except Exception:
        return False
    expected = min(int(top_k), int(len(feature_names)))
    return selected_idx.ndim == 1 and len(selected_idx) == expected and expected > 0


def _reduced_data_valid(reduced_data_path: Path, top_k: int) -> bool:
    if not reduced_data_path.exists():
        return False
    try:
        payload = np.load(reduced_data_path, allow_pickle=True)
        feature_names = payload["feature_names"]
        if "X_train" not in payload:
            return False
        x_train = payload["X_train"]
    except Exception:
        return False
    expected = int(top_k)
    return (
        expected > 0
        and x_train.ndim == 2
        and len(feature_names) == expected
        and x_train.shape[1] == expected
    )



def _ensure_topk_data_artifact(
    args: argparse.Namespace,
    cfg: ExperimentConfig,
    base_data_file: Path,
    base_output_dir: Path,
    train_ds: str,
    test_ds: str,
    logger,
    reports: dict,
) -> Path:
    top_k = args.top_k if args.top_k is not None else cfg.shap.top_k
    shap_dir, fs_dir, reduced_data, selected_idx = _resolve_topk_paths(args, cfg, train_ds, test_ds, top_k)
    if _reduced_data_valid(reduced_data, top_k):
        return reduced_data

    if not args.auto_feature_selection:
        raise FileNotFoundError(
            f"Reduced data for cnn_bilstm_se_topk not found: {reduced_data}. "
            "Run SHAP + feature_selection first or enable --auto-feature-selection."
        )

    feature_names = shap_dir / "feature_names.npy"
    if not _shared_topk_assets_valid(selected_idx, feature_names, top_k):
        ref_train = cfg.shap.reference_train_dataset
        ref_test = cfg.shap.reference_test_dataset
        ref_data_file = Path(cfg.data.processed_dir) / f"cross_{ref_train}_to_{ref_test}.npz"
        if not ref_data_file.exists():
            if not args.auto_preprocess:
                raise FileNotFoundError(
                    f"Shared SHAP reference data not found: {ref_data_file}. "
                    "Enable --auto-preprocess or preprocess the reference pair first."
                )
            run_command(
                [
                    sys.executable,
                    "scripts/preprocess_cross_dataset.py",
                    "--config",
                    args.config,
                    "--train-dataset",
                    ref_train,
                    "--test-dataset",
                    ref_test,
                ],
                logger,
            )

        se_run_dir = reports.get(cfg.shap.reference_model_name, {}).get("run_dir")
        se_model_path = None
        if se_run_dir:
            candidate = Path(se_run_dir) / "best_model.pt"
            if candidate.exists():
                se_model_path = candidate
        if se_model_path is None:
            se_model_path = _find_reference_model_path(cfg, base_output_dir, logger)
        if se_model_path is None:
            raise FileNotFoundError(
                "Cannot build shared SHAP ranking because the reference cnn_bilstm_se model was not found."
            )
        shap_dir.mkdir(parents=True, exist_ok=True)
        run_command(
            [
                sys.executable,
                "scripts/shap_analysis.py",
                "--model",
                str(se_model_path),
                "--config",
                args.config,
                "--model-name",
                cfg.shap.reference_model_name,
                "--data-file",
                str(ref_data_file),
                "--output-dir",
                str(shap_dir),
            ],
            logger,
        )

    if not _shared_topk_assets_valid(selected_idx, feature_names, top_k):
        sorted_feature_idx = shap_dir / "sorted_feature_idx.npy"
        if sorted_feature_idx.exists():
            sorted_idx = np.load(sorted_feature_idx, allow_pickle=True).astype(np.int64)
            chosen = sorted_idx[: min(int(top_k), len(sorted_idx))]
            selected_idx.parent.mkdir(parents=True, exist_ok=True)
            np.save(selected_idx, chosen)
        else:
            raise FileNotFoundError(f"Shared Top-K index file was not generated: {selected_idx}")
    fs_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            sys.executable,
            "scripts/feature_selection.py",
            "--selected-idx",
            str(selected_idx),
            "--feature-names",
            str(feature_names),
            "--data-file",
            str(base_data_file),
            "--output-dir",
            str(fs_dir),
        ],
        logger,
    )

    if not reduced_data.exists():
        raise FileNotFoundError(f"Reduced data still missing after feature selection: {reduced_data}")
    return reduced_data


def _save_history_csv(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = ["epoch", "train_loss", "val_loss", "val_metric", "learning_rate"]
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in history:
            writer.writerow({k: item.get(k) for k in fieldnames})


def _save_run_artifacts(
    output_dir: Path,
    metrics: dict,
    history: list[dict],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm_list = metrics.get("confusion_matrix", [])
    cm = np.array(cm_list, dtype=np.int64) if cm_list else np.array([])
    if cm.size > 0:
        if cm.shape[0] == 2:
            labels = ["Benign", "Attack"]
        else:
            labels = [f"Class_{i}" for i in range(cm.shape[0])]
        plot_confusion_matrix(cm, labels=labels, output_path=figures_dir / "confusion_matrix.png")

    plot_nids_key_metrics(metrics, output_path=figures_dir / "nids_key_metrics.png")
    plot_split_distribution(
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        output_path=figures_dir / "split_distribution.png",
        benign_class=0,
    )

    _save_history_csv(history, output_dir / "training_history.csv")
    np.savez_compressed(output_dir / "test_predictions.npz", y_true=y_true, y_pred=y_pred)
    save_json(
        {
            "avg_attack_recall": float(metrics.get("avg_attack_recall", 0.0)),
            "attack_macro_precision": float(metrics.get("attack_macro_precision", 0.0)),
            "benign_false_alarm_rate": float(metrics.get("benign_false_alarm_rate", 0.0)),
            "attack_miss_rate": float(metrics.get("attack_miss_rate", 0.0)),
            "macro_f1": float(metrics.get("macro_f1", 0.0)),
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "pr_auc": float(metrics.get("pr_auc", 0.0)),
            "roc_auc": float(metrics.get("roc_auc", 0.0)),
            "best_f1": float(metrics.get("best_f1", 0.0)),
            "best_f1_threshold": float(metrics.get("best_f1_threshold", 1.0)),
            "recall_at_far_1pct": float(metrics.get("recall_at_far_1pct", 0.0)),
            "threshold_at_far_1pct": float(metrics.get("threshold_at_far_1pct", 1.0)),
            "recall_at_far_5pct": float(metrics.get("recall_at_far_5pct", 0.0)),
            "threshold_at_far_5pct": float(metrics.get("threshold_at_far_5pct", 1.0)),
        },
        output_dir / "key_metrics.json",
    )
    save_json(
        {
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "test_samples": int(len(y_test)),
            "train_attack_ratio": float((y_train != 0).mean()) if len(y_train) else 0.0,
            "val_attack_ratio": float((y_val != 0).mean()) if len(y_val) else 0.0,
            "test_attack_ratio": float((y_test != 0).mean()) if len(y_test) else 0.0,
        },
        output_dir / "data_profile.json",
    )


def _register_run(model_root: Path, run_dir: Path, manifest: dict) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    (model_root / "latest_run.txt").write_text(run_dir.name, encoding="utf-8")
    (model_root / "latest_report.txt").write_text(str(run_dir / "report.json"), encoding="utf-8")
    save_json(manifest, model_root / "latest_run.json")

    index_path = model_root / "runs_index.json"
    if index_path.exists():
        try:
            import json

            records = json.loads(index_path.read_text(encoding="utf-8"))
            if not isinstance(records, list):
                records = []
        except Exception:
            records = []
    else:
        records = []

    records.append(
        {
            "run_name": run_dir.name,
            "run_dir": str(run_dir),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "model": manifest.get("model"),
            "imbalance_strategy": manifest.get("imbalance_strategy"),
            "report_path": str(run_dir / "report.json"),
        }
    )
    save_json(records, index_path)


def _ensure_data_artifact(
    cfg: ExperimentConfig,
    args: argparse.Namespace,
    train_ds: str,
    test_ds: str,
    logger,
) -> Path:
    data_file = _resolve_data_file(cfg, args)
    if data_file.exists():
        return data_file

    if args.data_file and not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    if not args.auto_preprocess:
        raise FileNotFoundError(
            f"Data file not found: {data_file}. Use --auto-preprocess or run preprocess scripts first."
        )

    logger.info("Data artifact missing, auto preprocessing...")
    base = [sys.executable]
    if args.cross_dataset or train_ds != test_ds:
        cmd = base + [
            "scripts/preprocess_cross_dataset.py",
            "--config",
            args.config,
            "--train-dataset",
            train_ds,
            "--test-dataset",
            test_ds,
        ]
        if args.max_rows is not None:
            cmd += ["--max-rows", str(args.max_rows)]
    else:
        cmd = base + [
            "scripts/preprocess.py",
            "--dataset",
            train_ds,
            "--config",
            args.config,
        ]
        if args.max_rows is not None:
            cmd += ["--max-rows", str(args.max_rows)]

    run_command(cmd, logger)
    if not data_file.exists():
        raise RuntimeError(f"Auto preprocess succeeded but artifact missing: {data_file}")
    return data_file


def _measure_classical_latency(
    model,
    X_test: np.ndarray,
    batch_size: int = 512,
    n_batches: int = 10,
) -> dict:
    latencies = []
    n_samples = len(X_test)
    steps = min(n_batches, max(1, int(np.ceil(n_samples / batch_size))))
    total_seen = 0
    for i in range(steps):
        start_idx = i * batch_size
        end_idx = min(n_samples, (i + 1) * batch_size)
        if start_idx >= end_idx:
            break
        xb = X_test[start_idx:end_idx]
        t0 = time.perf_counter()
        _ = model.predict(xb)
        elapsed = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed)
        total_seen += len(xb)

    if not latencies:
        return {"mean_latency_ms": 0.0, "p50_latency_ms": 0.0, "p99_latency_ms": 0.0, "throughput": 0.0}
    total_time_s = sum(latencies) / 1000.0
    return {
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "throughput": float(total_seen / total_time_s) if total_time_s > 0 else 0.0,
    }


def run_classical_training(
    cfg: ExperimentConfig,
    data_file: Path,
    output_dir: Path,
    model_name: str,
    imbalance_strategy: str,
    report_model_name: str | None = None,
    save_paper_artifacts: bool = True,
) -> dict:
    logger = get_logger("train")
    logger.info(
        "Start classical training | model=%s train_dataset=%s test_dataset=%s",
        model_name,
        cfg.data.train_dataset,
        cfg.data.test_dataset,
    )
    payload = _load_npz(data_file)
    X_train = payload["X_train"].astype(np.float32)
    X_val = payload["X_val"].astype(np.float32)
    X_test = payload["X_test"].astype(np.float32)
    y_train = payload["y_train"].astype(np.int64)
    y_val = payload["y_val"].astype(np.int64)
    y_test = payload["y_test"].astype(np.int64)
    feature_names = payload.get("feature_names")
    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(X_train.shape[1])])

    cfg.model.input_dim = int(X_train.shape[1])
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    cfg.model.num_classes = 2 if n_classes <= 2 else n_classes
    cfg.model.name = model_name

    if imbalance_strategy == "smote":
        X_train, y_train = apply_smote(X_train, y_train)

    t0 = time.perf_counter()
    if model_name == "random_forest":
        model = train_random_forest(X_train, y_train, random_state=cfg.runtime.seed)
    elif model_name == "xgboost":
        model = train_xgboost(X_train, y_train)
    else:
        raise ValueError(f"Unsupported classical model: {model_name}")
    fit_seconds = time.perf_counter() - t0

    y_pred = model.predict(X_test)
    y_score = predict_binary_scores(model, X_test) if cfg.model.num_classes == 2 else None
    test_metrics = compute_nids_metrics(y_test, y_pred, y_score=y_score)
    latency = _measure_classical_latency(model, X_test, batch_size=cfg.data.batch_size)
    selected_metric_name = cfg.training.selection_metric
    selected_metric_value = float(
        test_metrics.get(selected_metric_name, test_metrics.get("avg_attack_recall", 0.0))
    )

    model_path = output_dir / "best_model.pkl"
    with model_path.open("wb") as f:
        pickle.dump(model, f)

    report = {
        "model": report_model_name or cfg.model.name,
        "imbalance_strategy": imbalance_strategy,
        "data_file": str(data_file),
        "input_dim": cfg.model.input_dim,
        "num_classes": cfg.model.num_classes,
        "feature_names": feature_names.tolist(),
        "training_summary": {
            "best_metric": selected_metric_value,
            "best_epoch": 1,
            "best_model_path": str(model_path),
            "history": [{"epoch": 1, "train_loss": None, "val_loss": None, "val_metric": None}],
            "fit_seconds": fit_seconds,
        },
        "test_metrics": test_metrics,
        "latency": latency,
    }
    save_json(report, output_dir / "report.json")
    save_config(cfg, output_dir / "resolved_config.yaml")
    _save_history_csv(report["training_summary"]["history"], output_dir / "training_history.csv")
    if save_paper_artifacts:
        _save_run_artifacts(
            output_dir=output_dir,
            metrics=test_metrics,
            history=report["training_summary"]["history"],
            y_true=y_test,
            y_pred=y_pred,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )
    logger.info("Training done (classical). report=%s", output_dir / "report.json")
    return report


def run_training(
    cfg: ExperimentConfig,
    data_file: Path,
    output_dir: Path,
    model_name: str,
    imbalance_strategy: str,
    report_model_name: str | None = None,
    resume_checkpoint: Path | None = None,
    save_paper_artifacts: bool = True,
) -> dict:
    if model_name in CLASSICAL_MODELS:
        return run_classical_training(
            cfg,
            data_file,
            output_dir,
            model_name,
            imbalance_strategy=imbalance_strategy,
            report_model_name=report_model_name,
            save_paper_artifacts=save_paper_artifacts,
        )

    logger = get_logger("train")
    logger.info(
        "Start deep training | model=%s train_dataset=%s test_dataset=%s batch_size=%s epochs=%s lr=%.6f",
        model_name,
        cfg.data.train_dataset,
        cfg.data.test_dataset,
        cfg.data.batch_size,
        cfg.training.num_epochs,
        cfg.training.learning_rate,
    )
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}. Run preprocess scripts first."
        )

    payload = _load_npz(data_file)
    X_train = payload["X_train"].astype(np.float32)
    X_val = payload["X_val"].astype(np.float32)
    X_test = payload["X_test"].astype(np.float32)
    y_train = payload["y_train"].astype(np.int64)
    y_val = payload["y_val"].astype(np.int64)
    y_test = payload["y_test"].astype(np.int64)
    feature_names = payload.get("feature_names")
    if feature_names is None:
        feature_names = np.array([f"f{i}" for i in range(X_train.shape[1])])

    cfg.model.input_dim = int(X_train.shape[1])
    n_classes = int(len(np.unique(np.concatenate([y_train, y_val, y_test]))))
    cfg.model.num_classes = 2 if n_classes <= 2 else n_classes
    cfg.model.name = model_name

    if imbalance_strategy == "smote":
        X_train, y_train = apply_smote(X_train, y_train)

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    model = create_model(cfg.model)
    class_weights = compute_class_weights(y_train) if imbalance_strategy == "class_weights" else None
    trainer = Trainer(config=cfg.training, output_dir=output_dir)
    summary = trainer.fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=cfg.model.num_classes,
        class_weights=class_weights,
        resume_checkpoint=resume_checkpoint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(summary.best_model_path, map_location=device, weights_only=True))
    model.to(device)
    test_result = trainer.evaluate(
        model=model,
        data_loader=test_loader,
        criterion=None,
        device=device,
        num_classes=cfg.model.num_classes,
    )
    latency = trainer.measure_latency(model, test_loader)

    summary.test_metrics = test_result.metrics
    report = {
        "model": report_model_name or cfg.model.name,
        "imbalance_strategy": imbalance_strategy,
        "data_file": str(data_file),
        "input_dim": cfg.model.input_dim,
        "num_classes": cfg.model.num_classes,
        "feature_names": feature_names.tolist(),
        "training_summary": asdict(summary),
        "test_metrics": test_result.metrics,
        "latency": latency,
    }

    save_json(report, output_dir / "report.json")
    save_config(cfg, output_dir / "resolved_config.yaml")
    if summary.history:
        plot_training_curves(summary.history, output_dir / "training_curves.png")
        _save_history_csv(summary.history, output_dir / "training_history.csv")
    if save_paper_artifacts:
        _save_run_artifacts(
            output_dir=output_dir,
            metrics=test_result.metrics,
            history=summary.history,
            y_true=test_result.labels,
            y_pred=test_result.predictions,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
        )

    logger.info("Training done. report=%s", output_dir / "report.json")
    return report


def main() -> None:
    args = parse_args()
    if args.one_click:
        args.all_models = True
        args.auto_preprocess = True
        args.auto_feature_selection = True
        if not args.force:
            args.skip_existing = True
    if args.force:
        args.skip_existing = False
    if args.all_models and not args.force:
        args.skip_existing = True

    cfg = load_config(args.config)
    seed_everything(cfg.runtime.seed)

    train_ds = args.train_dataset or cfg.data.train_dataset
    test_ds = args.test_dataset or cfg.data.test_dataset
    logger = get_logger("train")
    data_file = _ensure_data_artifact(cfg, args, train_ds, test_ds, logger)
    models = _resolve_models(args, cfg)

    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = Path(cfg.runtime.output_dir) / f"{train_ds}_to_{test_ds}"
    base_output_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    explicit_resume_run_dir = Path(args.resume_run_dir).resolve() if args.resume_run_dir else None
    if explicit_resume_run_dir is not None and len(models) != 1:
        raise ValueError("--resume-run-dir is only supported in single-model mode.")

    for requested_model_name in models:
        model_name = _canonical_model_name(requested_model_name)
        if args.output_dir and len(models) == 1:
            model_root = base_output_dir
        else:
            model_root = base_output_dir / model_name
        model_root.mkdir(parents=True, exist_ok=True)

        existing_report = _find_latest_report(model_root)
        if args.skip_existing and existing_report is not None:
            logger.info("Skip existing model=%s at %s", model_name, existing_report)
            continue

        effective_strategy = _resolve_effective_imbalance_strategy(model_name, args.imbalance_strategy)
        logger.info(
            "Resolved imbalance strategy | model=%s requested=%s effective=%s",
            model_name,
            args.imbalance_strategy,
            effective_strategy,
        )
        train_model_name = model_name
        train_data_file = data_file
        report_model_name = model_name
        if model_name == "cnn_bilstm_se_topk":
            train_model_name = "cnn_bilstm_se"
            train_data_file = _ensure_topk_data_artifact(
                args=args,
                cfg=cfg,
                base_data_file=data_file,
                base_output_dir=base_output_dir,
                train_ds=train_ds,
                test_ds=test_ds,
                logger=logger,
                reports=reports,
            )

        run_name = ""
        run_output_dir = model_root
        resume_checkpoint = None
        resumed_from_checkpoint = False

        if args.resume and train_model_name in TRAINABLE_DEEP_MODELS:
            resumable_run_dir = None
            if explicit_resume_run_dir is not None:
                resumable_run_dir = explicit_resume_run_dir
            else:
                resumable_run_dir = _find_latest_checkpoint_run(model_root)
            if resumable_run_dir is not None:
                candidate_ckpt = (resumable_run_dir / "checkpoint_last.pt").resolve()
                if candidate_ckpt.exists():
                    run_output_dir = candidate_ckpt.parent
                    run_name = resumable_run_dir.name
                    resume_checkpoint = candidate_ckpt
                    resumed_from_checkpoint = True
                    logger.info("Resume enabled | model=%s checkpoint=%s", model_name, candidate_ckpt)
                elif explicit_resume_run_dir is not None:
                    raise FileNotFoundError(
                        f"Requested resume run dir has no checkpoint_last.pt: {resumable_run_dir}"
                    )

        if args.resume and train_model_name in CLASSICAL_MODELS:
            logger.info("Resume is not applicable to classical model=%s; training will start as new run.", model_name)

        if not resumed_from_checkpoint:
            run_name = _build_run_name(model_name, effective_strategy, args.run_tag)
            run_output_dir = model_root / "runs" / run_name
            if run_output_dir.exists():
                suffix = 1
                while True:
                    candidate = model_root / "runs" / f"{run_name}_{suffix:02d}"
                    if not candidate.exists():
                        run_output_dir = candidate
                        run_name = candidate.name
                        break
                    suffix += 1
            run_output_dir.mkdir(parents=True, exist_ok=True)

        cfg_local = load_config(args.config)
        cfg_local.data.train_dataset = train_ds
        cfg_local.data.test_dataset = test_ds
        started_at = datetime.now().isoformat(timespec="seconds")
        report = run_training(
            cfg_local,
            data_file=train_data_file,
            output_dir=run_output_dir,
            model_name=train_model_name,
            imbalance_strategy=effective_strategy,
            report_model_name=report_model_name,
            resume_checkpoint=resume_checkpoint,
            save_paper_artifacts=not args.no_paper_artifacts,
        )
        finished_at = datetime.now().isoformat(timespec="seconds")
        manifest = {
            "run_name": run_name,
            "run_dir": str(run_output_dir),
            "started_at": started_at,
            "finished_at": finished_at,
            "command": " ".join(sys.argv),
            "train_dataset": train_ds,
            "test_dataset": test_ds,
            "model": model_name,
            "train_model_backbone": train_model_name,
            "imbalance_strategy": effective_strategy,
            "data_file": str(train_data_file),
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
            "report_path": str(run_output_dir / "report.json"),
        }
        save_json(manifest, run_output_dir / "run_manifest.json")
        _register_run(model_root, run_output_dir, manifest)
        reports[model_name] = {
            "run_dir": str(run_output_dir),
            "report_path": str(run_output_dir / "report.json"),
            "imbalance_strategy": effective_strategy,
            "test_metrics": report.get("test_metrics", {}),
        }

    if len(models) > 1:
        save_json(reports, base_output_dir / "multi_model_report.json")


if __name__ == "__main__":
    main()
