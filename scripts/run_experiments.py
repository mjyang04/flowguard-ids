from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.utils.io import save_json
from nids.utils.logging import get_logger
from nids.utils.process import run_command

DEFAULT_EXPERIMENTS = [
    ("same_cicids", "cicids2017", "cicids2017"),
    ("cross_cic_to_unsw", "cicids2017", "unsw_nb15"),
    ("cross_unsw_to_cic", "unsw_nb15", "cicids2017"),
]
LAPTOP_3060_MODELS = ["cnn_bilstm_se", "random_forest", "xgboost"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all NIDS experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments",
        help="Directory used for experiment summary files such as experiment_status.json",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--models",
        default=None,
        help="Model selection: single name, comma-separated list, or 'all' (e.g. --models cnn_bilstm_se,xgboost)",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--auto-preprocess", action="store_true")
    parser.add_argument(
        "--cross-only",
        action="store_true",
        help="Run only cross-dataset experiments (skip same-dataset reference runs)",
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=["default", "laptop_3060"],
        help="Optional preset experiment profile",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full retraining of all selected models (disable skip logic)",
    )
    parser.add_argument(
        "--one-click",
        action="store_true",
        help="One-click experiment: auto preprocessing + all models + train only missing results",
    )
    return parser.parse_args()


def _apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    resolved = argparse.Namespace(**vars(args))
    if resolved.profile == "laptop_3060":
        resolved.cross_only = True
        if not resolved.models:
            resolved.models = ",".join(LAPTOP_3060_MODELS)
    return resolved


def _resolve_experiments(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.cross_only:
        return DEFAULT_EXPERIMENTS[1:]
    return DEFAULT_EXPERIMENTS


def _build_train_command(
    args: argparse.Namespace,
    train_ds: str,
    test_ds: str,
) -> list[str]:
    one_click = args.one_click
    skip_existing = (args.skip_existing or one_click) and (not args.force)
    auto_preprocess = args.auto_preprocess or one_click

    train_cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        args.config,
        "--train-dataset",
        train_ds,
        "--test-dataset",
        test_ds,
    ]

    if auto_preprocess:
        train_cmd.append("--auto-preprocess")
    if one_click:
        train_cmd.append("--auto-feature-selection")
    if args.max_rows is not None:
        train_cmd += ["--max-rows", str(args.max_rows)]
    models = args.models or ("all" if one_click else None)
    if models:
        train_cmd += ["--models", models]
    if skip_existing:
        train_cmd.append("--skip-existing")
    if one_click and not args.force:
        train_cmd.append("--resume")
    if args.force:
        train_cmd.append("--force")
    return train_cmd


def main() -> None:
    args = _apply_profile_defaults(parse_args())
    logger = get_logger("run_experiments")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = [sys.executable]
    results = {}
    experiments = _resolve_experiments(args)

    for exp_name, train_ds, test_ds in experiments:
        one_click = args.one_click
        skip_existing = (args.skip_existing or one_click) and (not args.force)
        auto_preprocess = args.auto_preprocess or one_click

        exp_output_dir = output_dir / exp_name
        exp_output_dir.mkdir(parents=True, exist_ok=True)

        preprocess_ok = True
        if not auto_preprocess:
            preprocess_cmd = base + [
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
            try:
                run_command(preprocess_cmd, logger)
            except RuntimeError as exc:
                logger.error("Preprocess failed for %s: %s", exp_name, exc)
                preprocess_ok = False

        train_cmd = _build_train_command(args, train_ds=train_ds, test_ds=test_ds)

        train_ok = False
        if preprocess_ok:
            try:
                run_command(train_cmd, logger)
                train_ok = True
            except RuntimeError as exc:
                logger.error("Train failed for %s: %s", exp_name, exc)

        results[exp_name] = {
            "preprocess_ok": preprocess_ok,
            "train_ok": train_ok,
            "skip_existing": skip_existing,
            "auto_preprocess": auto_preprocess,
            "profile": args.profile,
            "models": args.models or ("all" if one_click else None),
        }

    save_json(results, output_dir / "experiment_status.json")
    logger.info("Experiment run finished. Summary: %s", output_dir / "experiment_status.json")


if __name__ == "__main__":
    main()
