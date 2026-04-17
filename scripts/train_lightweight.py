from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import load_config
from nids.utils.logging import get_logger
from nids.utils.process import run_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain a lightweight model on SHAP-selected Top-K features. "
            "Preprocessing and feature-selection run automatically when the required "
            "artifacts are missing."
        )
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", choices=["random_forest", "xgboost"], default="random_forest")
    parser.add_argument("--train-dataset", default=None, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--test-dataset", default=None, choices=["cicids2017", "unsw_nb15"])
    parser.add_argument("--force", action="store_true", help="Retrain even if output already exists")
    parser.add_argument("--output-dir", default=None, help="Override the canonical output directory")

    # Advanced / debugging controls.
    parser.add_argument("--base-data-file", default=None, help="Advanced: explicit full-feature npz")
    parser.add_argument("--reduced-data", default=None, help="Advanced: explicit Top-K reduced npz")
    parser.add_argument(
        "--imbalance-strategy",
        default="auto",
        choices=["auto", "smote", "oversampling", "weighted_sampler", "none"],
        help="Advanced: imbalance strategy forwarded to scripts/train.py",
    )
    parser.add_argument("--run-tag", default="lightweight", help="Advanced: run tag for output directory")
    parser.add_argument("--resume", action="store_true", help="Advanced: forwarded to scripts/train.py")
    parser.add_argument("--resume-run-dir", default=None, help="Advanced: forwarded to scripts/train.py")
    parser.add_argument("--max-rows", type=int, default=None, help="Advanced: cap preprocessed rows")

    args = parser.parse_args()
    # Preprocess / feature-selection are always on — users should not need to opt
    # into resolving missing artifacts on the lightweight pipeline.
    args.auto_preprocess = True
    args.auto_feature_selection = True
    args.shap_dir = "artifacts/shap"
    args.feature_selection_dir = "artifacts/feature_selection"
    args.top_k = None
    return args



def _resolve_base_data_file(cfg, args: argparse.Namespace) -> Path:
    if args.base_data_file:
        return Path(args.base_data_file)
    train_ds = args.train_dataset or cfg.data.train_dataset
    test_ds = args.test_dataset or cfg.data.test_dataset
    return Path(cfg.data.processed_dir) / f"cross_{train_ds}_to_{test_ds}.npz"


def _resolve_shared_shap_dir(cfg, args: argparse.Namespace) -> Path:
    if args.shap_dir == "artifacts/shap":
        return (
            Path(args.shap_dir)
            / "shared"
            / f"{cfg.shap.reference_train_dataset}_to_{cfg.shap.reference_test_dataset}"
            / cfg.shap.reference_model_name
        )
    return Path(args.shap_dir)


def _has_existing_report(model_root: Path) -> bool:
    runs_dir = model_root / "runs"
    if runs_dir.exists():
        for path in runs_dir.glob("*/report.json"):
            if path.is_file():
                return True
    return (model_root / "report.json").exists()


def main() -> None:
    args = parse_args()
    logger = get_logger("train_lightweight")
    cfg = load_config(args.config)

    train_ds = args.train_dataset or cfg.data.train_dataset
    test_ds = args.test_dataset or cfg.data.test_dataset
    base_data_file = _resolve_base_data_file(cfg, args)
    top_k = cfg.shap.top_k
    shared_shap_dir = _resolve_shared_shap_dir(cfg, args)
    if args.reduced_data is None:
        reduced_data = (
            Path(args.feature_selection_dir)
            / f"{train_ds}_to_{test_ds}"
            / f"lightweight_top{top_k}"
            / "reduced_data.npz"
        )
    else:
        reduced_data = Path(args.reduced_data)
    feature_selection_dir = reduced_data.parent

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(cfg.runtime.output_dir) / "lightweight" / f"{train_ds}_to_{test_ds}" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if _has_existing_report(output_dir) and not args.force:
        logger.info(
            "Skip: lightweight model already trained under %s (use --force to retrain)",
            output_dir,
        )
        return

    # Step 1: Ensure base data artifact exists (auto-preprocessed).
    if not base_data_file.exists():
        preprocess_cmd = [
            sys.executable,
            "scripts/preprocess_cross_dataset.py",
            "--config",
            args.config,
            "--train-dataset",
            train_ds,
            "--test-dataset",
            test_ds,
        ]
        if args.max_rows is not None:
            preprocess_cmd += ["--max-rows", str(args.max_rows)]
        run_command(preprocess_cmd, logger)

    # Step 2: Ensure reduced_data exists (auto feature-selection).
    if not reduced_data.exists():
        selected_idx = shared_shap_dir / f"top{top_k}_idx.npy"
        feature_names = shared_shap_dir / "feature_names.npy"
        if not selected_idx.exists() or not feature_names.exists():
            raise FileNotFoundError(
                f"Shared SHAP ranking missing in {shared_shap_dir}. "
                "Please generate the shared reference SHAP outputs first."
            )

        feature_selection_cmd = [
            sys.executable,
            "scripts/feature_selection.py",
            "--selected-idx",
            str(selected_idx),
            "--feature-names",
            str(feature_names),
            "--data-file",
            str(base_data_file),
            "--output-dir",
            str(feature_selection_dir),
        ]
        run_command(feature_selection_cmd, logger)

    if not reduced_data.exists():
        raise FileNotFoundError(f"Reduced data still missing after feature selection: {reduced_data}")

    # Step 3: Train lightweight model on reduced features.
    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        args.config,
        "--data-file",
        str(reduced_data),
        "--models",
        args.model,
        "--output-dir",
        str(output_dir),
        "--imbalance-strategy",
        args.imbalance_strategy,
        "--run-tag",
        args.run_tag,
    ]
    if args.force:
        train_cmd.append("--force")
    if args.resume:
        train_cmd.append("--resume")
    if args.resume_run_dir:
        train_cmd += ["--resume-run-dir", args.resume_run_dir]
    run_command(train_cmd, logger)
    logger.info("Lightweight model training finished. output=%s", output_dir)


if __name__ == "__main__":
    main()
