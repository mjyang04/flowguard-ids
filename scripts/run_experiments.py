from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import load_config
from nids.utils.io import save_json
from nids.utils.logging import get_logger
from nids.utils.process import run_command

# Direction name map keeps legacy experiment keys stable across refactors.
# Any direction tuple not in the map falls back to ``<train>_to_<test>``.
DIRECTION_NAME_MAP = {
    ("cicids2017", "cicids2017"): "same_cicids",
    ("unsw_nb15", "unsw_nb15"): "same_unsw",
    ("cicids2017", "unsw_nb15"): "cross_cic_to_unsw",
    ("unsw_nb15", "cicids2017"): "cross_unsw_to_cic",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all NIDS experiments")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments",
        help="Directory for the experiment_status.json summary",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Override pipeline.models: single name, comma-separated list, or 'all'",
    )
    parser.add_argument(
        "--directions",
        default=None,
        help=(
            "Override pipeline.directions. Format: 'train:test,train:test' "
            "(e.g. 'cicids2017:cicids2017,unsw_nb15:unsw_nb15')."
        ),
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
    parser.add_argument(
        "--label-mode",
        choices=["binary", "multiclass"],
        default="binary",
        help=(
            "Target task. 'multiclass' restricts the matrix to same-dataset "
            "CIC + UNSW rows and disables cross-dataset enhancements."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None, help="Advanced: cap preprocessed rows")
    return parser.parse_args()


def _parse_directions_flag(raw: str | None) -> list[tuple[str, str]] | None:
    """Parse ``--directions`` CLI value into ``[(train, test), ...]`` or ``None``."""
    if not raw:
        return None
    out: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid direction spec {item!r}; expected 'train:test'")
        train, test = item.split(":", 1)
        out.append((train.strip(), test.strip()))
    return out


def _resolve_experiments(args: argparse.Namespace) -> list[tuple[str, str, str]]:
    """Build ``[(exp_name, train_ds, test_ds), ...]`` from resolved directions.

    Expects ``args.directions`` to already be a ``list[tuple[str, str]]`` after
    :func:`_apply_pipeline_config_defaults`.
    """
    directions: list[tuple[str, str]] = list(getattr(args, "directions", []) or [])
    if getattr(args, "label_mode", "binary") == "multiclass":
        # Multiclass has no cross-dataset mapping, so keep same-dataset rows only.
        directions = [(t, te) for t, te in directions if t == te]
    result: list[tuple[str, str, str]] = []
    for train_ds, test_ds in directions:
        name = DIRECTION_NAME_MAP.get((train_ds, test_ds), f"{train_ds}_to_{test_ds}")
        result.append((name, train_ds, test_ds))
    return result


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

    label_mode = getattr(args, "label_mode", "binary")
    # Multiclass runs are binary-only-incompatible with Platt/AUC-loss;
    # emit --label-mode explicitly and skip the cross-dataset bundle.
    if label_mode == "multiclass":
        train_cmd += ["--label-mode", "multiclass"]
        return train_cmd

    # Auto-enable the cross-dataset enhancement bundle on transfer directions
    # unless the caller explicitly opted out with --no-cross-dataset-enhancements.
    if train_ds != test_ds and args.cross_dataset_enhancements:
        train_cmd.append("--cross-dataset-enhancements")
    elif train_ds != test_ds and not args.cross_dataset_enhancements:
        train_cmd.append("--no-cross-dataset-enhancements")
    return train_cmd


def _apply_pipeline_config_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Fill in CLI defaults from ``cfg.pipeline`` when the user did not override.

    The CLI layer still wins; this only activates for fields left at their
    "unspecified" sentinel values (``None`` for lists, ``False`` for store_true
    flags that we consider opt-in via config).
    """
    try:
        cfg = load_config(args.config)
    except FileNotFoundError:
        return args

    pipeline = cfg.pipeline
    resolved = argparse.Namespace(**vars(args))
    if resolved.models is None and pipeline.models:
        resolved.models = ",".join(pipeline.models)
    if resolved.seeds is None and pipeline.seeds:
        resolved.seeds = list(pipeline.seeds)
    if not resolved.one_click and pipeline.one_click:
        resolved.one_click = True
    if not resolved.force and pipeline.force:
        resolved.force = True
    # BooleanOptionalAction semantics for cross_dataset_enhancements: the CLI
    # default is True, so we only override when the config explicitly says False.
    if resolved.cross_dataset_enhancements and not pipeline.cross_dataset_enhancements:
        resolved.cross_dataset_enhancements = False

    # Directions: CLI --directions wins; otherwise fall back to cfg.pipeline.directions.
    cli_directions = _parse_directions_flag(getattr(resolved, "directions", None))
    if cli_directions is not None:
        resolved.directions = cli_directions
    else:
        resolved.directions = [(d[0], d[1]) for d in pipeline.directions]
    return resolved


def main() -> None:
    args = _apply_pipeline_config_defaults(parse_args())
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
            # and Top-K lookups do not collide across seeds. Multiclass runs
            # also get an explicit suffix so they do not overwrite binary runs
            # that share the same seed and direction.
            label_mode = getattr(args, "label_mode", "binary")
            multiclass_suffix = (
                "_multiclass" if label_mode in ("multiclass", "multi") else ""
            )
            train_output_dir: Path | None = None
            if multi_seed and seed is not None:
                train_output_dir = (
                    Path("artifacts")
                    / f"seed{seed}"
                    / f"{train_ds}_to_{test_ds}{multiclass_suffix}"
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
                "models": args.models or ("all" if args.one_click else None),
                "cross_dataset_enhancements": bool(args.cross_dataset_enhancements),
                "label_mode": getattr(args, "label_mode", "binary"),
            }

    save_json(results, output_dir / "experiment_status.json")
    logger.info("Experiment run finished. Summary: %s", output_dir / "experiment_status.json")


if __name__ == "__main__":
    main()
