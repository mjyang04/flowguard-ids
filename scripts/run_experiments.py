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
    ("same_unsw", "unsw_nb15", "unsw_nb15"),
    ("cross_cic_to_unsw", "cicids2017", "unsw_nb15"),
    ("cross_unsw_to_cic", "unsw_nb15", "cicids2017"),
]
LAPTOP_3060_EXPERIMENTS = ["same_cicids", "same_unsw", "cross_cic_to_unsw"]
LAPTOP_3060_MODELS = ["cnn_bilstm_se", "random_forest", "xgboost"]
DEFAULT_SEEDS = [42, 43, 44]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all NIDS experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments",
        help="Directory for the experiment_status.json summary",
    )
    parser.add_argument(
        "--profile",
        default="default",
        choices=["default", "laptop_3060"],
        help="Experiment profile: 'default' runs the full matrix; 'laptop_3060' is the RTX 3060 preset",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Model selection: single name, comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--one-click",
        action="store_true",
        help="One-click: all models, auto-preprocess + auto feature-selection, skip finished runs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining (overrides the default skip-existing behavior)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Multi-seed: each seed lands in artifacts/seed<S>/<train>_to_<test>/ for downstream aggregation",
    )
    parser.add_argument(
        "--no-cross-dataset-enhancements",
        dest="cross_dataset_enhancements",
        action="store_false",
        default=True,
        help="Disable the AUC+Platt+LS bundle that is auto-enabled on cross-dataset directions",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Advanced: cap preprocessed rows")
    return parser.parse_args()


def _apply_profile_defaults(args: argparse.Namespace) -> argparse.Namespace:
    resolved = argparse.Namespace(**vars(args))
    if resolved.profile == "laptop_3060" and not resolved.models:
        resolved.models = ",".join(LAPTOP_3060_MODELS)
    return resolved


def _resolve_experiments(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    if args.profile == "laptop_3060":
        allowed = set(LAPTOP_3060_EXPERIMENTS)
        return [e for e in DEFAULT_EXPERIMENTS if e[0] in allowed]
    return DEFAULT_EXPERIMENTS


def _resolve_seeds(args: argparse.Namespace) -> list[int | None]:
    if args.seeds is None:
        return [None]
    if not args.seeds:
        raise ValueError("--seeds requires at least one integer")
    return [int(s) for s in args.seeds]


def _build_train_command(
    args: argparse.Namespace,
    train_ds: str,
    test_ds: str,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> list[str]:
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
    if args.one_click:
        train_cmd += ["--one-click", "--resume"]
    if args.max_rows is not None:
        train_cmd += ["--max-rows", str(args.max_rows)]
    models = args.models or ("all" if args.one_click else None)
    if models:
        train_cmd += ["--models", models]
    if args.force:
        train_cmd.append("--force")
    if seed is not None:
        train_cmd += ["--seed", str(seed)]
    if output_dir is not None:
        train_cmd += ["--output-dir", str(output_dir)]

    # Auto-enable the cross-dataset enhancement bundle on transfer directions
    # unless the caller explicitly opted out with --no-cross-dataset-enhancements.
    if train_ds != test_ds and args.cross_dataset_enhancements:
        train_cmd.append("--cross-dataset-enhancements")
    elif train_ds != test_ds and not args.cross_dataset_enhancements:
        train_cmd.append("--no-cross-dataset-enhancements")
    return train_cmd


def main() -> None:
    args = _apply_profile_defaults(parse_args())
    logger = get_logger("run_experiments")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    experiments = _resolve_experiments(args)
    seeds = _resolve_seeds(args)
    multi_seed = len(seeds) > 1 or (len(seeds) == 1 and seeds[0] is not None)

    for seed in seeds:
        for exp_name, train_ds, test_ds in experiments:
            run_key = f"{exp_name}__seed{seed}" if seed is not None else exp_name
            exp_output_dir = output_dir / exp_name
            exp_output_dir.mkdir(parents=True, exist_ok=True)

            # Multi-seed runs land in isolated artifact roots so skip-existing
            # and Top-K lookups do not collide across seeds.
            train_output_dir: Path | None = None
            if multi_seed and seed is not None:
                train_output_dir = (
                    Path("artifacts") / f"seed{seed}" / f"{train_ds}_to_{test_ds}"
                )

            train_cmd = _build_train_command(
                args,
                train_ds=train_ds,
                test_ds=test_ds,
                seed=seed,
                output_dir=train_output_dir,
            )

            train_ok = False
            try:
                run_command(train_cmd, logger)
                train_ok = True
            except RuntimeError as exc:
                logger.error("Train failed for %s: %s", run_key, exc)

            results[run_key] = {
                "experiment": exp_name,
                "seed": seed,
                "train_dataset": train_ds,
                "test_dataset": test_ds,
                "output_dir": str(train_output_dir) if train_output_dir else None,
                "train_ok": train_ok,
                "profile": args.profile,
                "models": args.models or ("all" if args.one_click else None),
                "cross_dataset_enhancements": bool(args.cross_dataset_enhancements),
            }

    save_json(results, output_dir / "experiment_status.json")
    logger.info("Experiment run finished. Summary: %s", output_dir / "experiment_status.json")


if __name__ == "__main__":
    main()
