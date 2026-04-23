"""Two-stage cascade evaluator.

Loads a trained Stage-1 (binary) model and a trained Stage-2 (multiclass)
model, runs both on the multiclass test split, merges the predictions via
``nids.evaluation.two_stage``, and reports side-by-side metrics against the
single-stage multiclass baseline.

Typical invocation:

    python scripts/eval_two_stage.py \\
        --stage1-run artifacts/seed42/cicids2017_to_cicids2017/cnn_bilstm_se/runs/<ts>_CNN-BiLSTM-SE_SMOTE_seed42 \\
        --stage2-run artifacts/seed42/cicids2017_to_cicids2017_multiclass/cnn_bilstm_se/runs/<ts>_CNN-BiLSTM-SE_SMOTE_seed42 \\
        --data-file data/processed/cicids2017/data_multiclass.npz \\
        --output-dir artifacts/seed42/cicids2017_to_cicids2017_multiclass/two_stage/cnn_bilstm_se
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import ExperimentConfig, load_config
from nids.evaluation.metrics import compute_nids_metrics
from nids.evaluation.two_stage import run_two_stage
from nids.models.registry import create_model
from nids.utils.io import save_json
from nids.utils.logging import get_logger
from nids.utils.run_layout import find_latest_best_model, find_latest_report
from nids.utils.visualization import plot_confusion_matrix


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two-stage cascade IDS")
    parser.add_argument(
        "--stage1-run",
        required=True,
        help="Stage-1 run directory (binary) or model_root (auto-resolves latest run)",
    )
    parser.add_argument(
        "--stage2-run",
        required=True,
        help="Stage-2 run directory (multiclass) or model_root (auto-resolves latest run)",
    )
    parser.add_argument(
        "--data-file",
        required=True,
        help="Path to the multiclass NPZ artifact (X_test/y_test must be multiclass)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write two_stage_report.json and figures",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Stage-1 attack-probability threshold (default 0.5)",
    )
    parser.add_argument(
        "--benign-class",
        type=int,
        default=0,
        help="Integer label reserved for benign traffic (default 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Inference batch size for deep models",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Run directory helpers
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunArtifacts:
    """Resolved paths for a single trained-model run."""

    run_dir: Path
    config_path: Path
    model_path: Path
    is_classical: bool


def _resolve_run_dir(candidate: Path) -> Path:
    """Accept either a run directory or a model_root (holding ``runs/``)."""
    candidate = candidate.resolve()
    if (candidate / "resolved_config.yaml").exists():
        return candidate
    latest_report = find_latest_report(candidate)
    if latest_report is not None:
        return latest_report.parent
    raise FileNotFoundError(
        f"Cannot locate a run directory under {candidate}: "
        "neither resolved_config.yaml nor runs/*/report.json was found."
    )


def _resolve_artifacts(candidate: str | Path) -> RunArtifacts:
    run_dir = _resolve_run_dir(Path(candidate))
    config_path = run_dir / "resolved_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"resolved_config.yaml missing in {run_dir}")

    pt_path = run_dir / "best_model.pt"
    pkl_path = run_dir / "best_model.pkl"
    if pt_path.exists():
        return RunArtifacts(run_dir=run_dir, config_path=config_path, model_path=pt_path, is_classical=False)
    if pkl_path.exists():
        return RunArtifacts(run_dir=run_dir, config_path=config_path, model_path=pkl_path, is_classical=True)

    # Fallback: search runs/ subtree
    latest_pt = find_latest_best_model(run_dir.parent, suffix=".pt")
    if latest_pt is not None:
        return RunArtifacts(
            run_dir=latest_pt.parent,
            config_path=latest_pt.parent / "resolved_config.yaml",
            model_path=latest_pt,
            is_classical=False,
        )
    latest_pkl = find_latest_best_model(run_dir.parent, suffix=".pkl")
    if latest_pkl is not None:
        return RunArtifacts(
            run_dir=latest_pkl.parent,
            config_path=latest_pkl.parent / "resolved_config.yaml",
            model_path=latest_pkl,
            is_classical=True,
        )
    raise FileNotFoundError(f"No best_model.pt or best_model.pkl under {run_dir}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_deep_probs(
    model: torch.nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_classes: int,
) -> np.ndarray:
    """Run a deep model in eval mode and return class probabilities.

    Returns an ``(N, num_classes)`` float64 array.
    """
    model.eval()
    probs_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            logits = model(batch)
            if num_classes <= 2 and logits.shape[-1] == 1:
                # Single-logit BCE head
                p_attack = torch.sigmoid(logits.squeeze(-1))
                p = torch.stack([1.0 - p_attack, p_attack], dim=-1)
            else:
                p = torch.softmax(logits, dim=-1)
            probs_chunks.append(p.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(probs_chunks, axis=0)


def _infer_classical_probs(model, X: np.ndarray, num_classes: int) -> np.ndarray:
    """Run a classical model and return class probabilities.

    If the estimator exposes ``predict_proba`` we use it; otherwise we fall
    back to hard predictions turned into a one-hot matrix.
    """
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X), dtype=np.float64)
        if probs.ndim == 1:
            probs = np.stack([1.0 - probs, probs], axis=-1)
        return probs
    preds = np.asarray(model.predict(X), dtype=np.int64)
    k = max(int(num_classes), int(preds.max()) + 1)
    onehot = np.zeros((preds.shape[0], k), dtype=np.float64)
    onehot[np.arange(preds.shape[0]), preds] = 1.0
    return onehot


def _load_stage_model(
    artifacts: RunArtifacts,
    X_shape_dim: int,
    logger: logging.Logger,
) -> tuple[object, ExperimentConfig, bool]:
    cfg = load_config(artifacts.config_path)
    if int(cfg.model.input_dim) != X_shape_dim:
        logger.warning(
            "Model %s was trained on input_dim=%d, but NPZ feature dim is %d; "
            "coercing cfg.model.input_dim and hoping feature spaces align.",
            artifacts.run_dir.name,
            cfg.model.input_dim,
            X_shape_dim,
        )
        cfg.model.input_dim = int(X_shape_dim)

    if artifacts.is_classical:
        with artifacts.model_path.open("rb") as f:
            clf = pickle.load(f)
        return clf, cfg, True

    model = create_model(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(artifacts.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model, cfg, False


def _stage_predict(
    loaded_model,
    cfg: ExperimentConfig,
    X: np.ndarray,
    is_classical: bool,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Produce (probs, preds) for a single stage."""
    num_classes = int(cfg.model.num_classes)
    if is_classical:
        probs = _infer_classical_probs(loaded_model, X, num_classes=num_classes)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probs = _infer_deep_probs(
            loaded_model,
            X,
            device=device,
            batch_size=batch_size,
            num_classes=num_classes,
        )
    preds = np.argmax(probs, axis=-1).astype(np.int64)
    return probs, preds


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    logger = get_logger("eval_two_stage")

    stage1 = _resolve_artifacts(args.stage1_run)
    stage2 = _resolve_artifacts(args.stage2_run)
    logger.info("Stage-1 run: %s", stage1.run_dir)
    logger.info("Stage-2 run: %s", stage2.run_dir)

    data_file = Path(args.data_file).resolve()
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    payload = np.load(data_file, allow_pickle=True)
    X_test = payload["X_test"].astype(np.float32)
    y_test = payload["y_test"].astype(np.int64)
    feature_dim = int(X_test.shape[1])
    logger.info(
        "Loaded multiclass NPZ: X_test=%s y_test=%s classes=%d",
        X_test.shape,
        y_test.shape,
        int(np.unique(y_test).size),
    )

    stage1_model, stage1_cfg, stage1_is_classical = _load_stage_model(stage1, feature_dim, logger)
    stage2_model, stage2_cfg, stage2_is_classical = _load_stage_model(stage2, feature_dim, logger)

    if int(stage1_cfg.model.num_classes) != 2:
        raise ValueError(
            f"Stage-1 must be a binary model; got num_classes={stage1_cfg.model.num_classes}"
        )
    if int(stage2_cfg.model.num_classes) < 3:
        logger.warning(
            "Stage-2 num_classes=%d looks like a binary head; cascade will degenerate.",
            stage2_cfg.model.num_classes,
        )

    stage1_probs, _ = _stage_predict(
        stage1_model, stage1_cfg, X_test, stage1_is_classical, args.batch_size
    )
    stage2_probs, stage2_preds = _stage_predict(
        stage2_model, stage2_cfg, X_test, stage2_is_classical, args.batch_size
    )

    attack_scores = stage1_probs[:, 1]
    cascade_result = run_two_stage(
        stage1_scores=attack_scores,
        stage1_binary=None,
        stage2_multiclass=stage2_preds,
        y_true=y_test,
        threshold=float(args.threshold),
        benign_class=int(args.benign_class),
    )

    baseline_metrics = compute_nids_metrics(
        y_test, stage2_preds, benign_class=int(args.benign_class)
    )
    two_stage_metrics = compute_nids_metrics(
        y_test,
        cascade_result.final_predictions,
        benign_class=int(args.benign_class),
    )

    delta = {
        "accuracy": float(two_stage_metrics["accuracy"] - baseline_metrics["accuracy"]),
        "macro_f1": float(two_stage_metrics["macro_f1"] - baseline_metrics["macro_f1"]),
        "weighted_f1": float(two_stage_metrics["weighted_f1"] - baseline_metrics["weighted_f1"]),
        "mcc": float(two_stage_metrics["mcc"] - baseline_metrics["mcc"]),
        "benign_false_alarm_rate": float(
            two_stage_metrics["benign_false_alarm_rate"]
            - baseline_metrics["benign_false_alarm_rate"]
        ),
    }

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(two_stage_metrics.get("confusion_matrix", []), dtype=np.int64)
    if cm.size > 0:
        labels = [f"Class_{i}" for i in range(cm.shape[0])]
        labels[int(args.benign_class)] = "Benign"
        plot_confusion_matrix(
            cm,
            labels=labels,
            output_path=figures_dir / "two_stage_confusion_matrix.png",
        )

    np.savez_compressed(
        output_dir / "two_stage_predictions.npz",
        y_true=y_test,
        y_baseline=stage2_preds,
        y_two_stage=cascade_result.final_predictions,
        stage1_binary=cascade_result.stage1_predictions,
        stage1_scores=attack_scores,
    )

    report = {
        "stage1_run": str(stage1.run_dir),
        "stage1_model_path": str(stage1.model_path),
        "stage2_run": str(stage2.run_dir),
        "stage2_model_path": str(stage2.model_path),
        "data_file": str(data_file),
        "benign_class": int(args.benign_class),
        "threshold": float(args.threshold),
        "baseline_multiclass_metrics": baseline_metrics,
        "two_stage_metrics": two_stage_metrics,
        "stage1_stats": cascade_result.stats,
        "delta": delta,
    }
    save_json(report, output_dir / "two_stage_report.json")
    logger.info(
        "Two-stage evaluation done. baseline_macro_f1=%.4f two_stage_macro_f1=%.4f delta=%.4f",
        baseline_metrics["macro_f1"],
        two_stage_metrics["macro_f1"],
        delta["macro_f1"],
    )
    logger.info("Report saved to %s", output_dir / "two_stage_report.json")


if __name__ == "__main__":
    main()
