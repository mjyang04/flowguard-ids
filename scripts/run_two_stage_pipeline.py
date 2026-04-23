"""One-shot two-stage cascade pipeline.

Automates the end-to-end workflow for the same-dataset two-stage thesis
track:

    1. (optional) Train binary models on the requested directions.
    2. (optional) Train multiclass models on the same directions.
    3. For every (seed, direction, model) triple, run ``eval_two_stage.py``
       and collect the results into ``artifacts/two_stage_summary.json``.

Typical usage:

    # Headline: everything comes from configs/default.yaml (pipeline section)
    python scripts/run_two_stage_pipeline.py --config configs/default.yaml

    # Skip training, just rerun cascade eval on existing artifacts
    python scripts/run_two_stage_pipeline.py --config configs/default.yaml --no-do-train
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import ExperimentConfig, load_config
from nids.utils.io import save_json
from nids.utils.logging import get_logger
from nids.utils.process import run_command
from nids.utils.run_layout import find_latest_report


@dataclass(frozen=True)
class PipelineTarget:
    seed: int | None
    train_ds: str
    test_ds: str
    model: str


@dataclass(frozen=True)
class ResolvedPipeline:
    """Resolved pipeline settings = config.pipeline defaults + CLI overrides."""

    models: list[str]
    seeds: list[int]
    directions: list[tuple[str, str]]
    do_train: bool
    force: bool
    stage1_threshold: float
    benign_class: int
    summary_path: Path


def parse_args() -> argparse.Namespace:
    """CLI is intentionally minimal.

    Every pipeline knob lives in ``configs/default.yaml`` under ``pipeline:``.
    CLI flags override when explicitly passed. ``--do-train`` / ``--no-do-train``
    (and the same for ``--force``) use ``BooleanOptionalAction`` so either
    direction can flip the config default.
    """
    parser = argparse.ArgumentParser(description="End-to-end two-stage cascade pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models", default=None, help="Override pipeline.models (comma-separated)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Override pipeline.seeds")
    parser.add_argument(
        "--do-train",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override pipeline.do_train",
    )
    parser.add_argument(
        "--force",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override pipeline.force",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override pipeline.stage1_threshold")
    parser.add_argument("--benign-class", type=int, default=None, help="Override pipeline.benign_class")
    parser.add_argument("--summary-path", default=None, help="Override pipeline.summary_path")
    return parser.parse_args()


def _parse_models(raw: str | list[str]) -> list[str]:
    if isinstance(raw, list):
        return [str(m).strip() for m in raw if str(m).strip()]
    return [m.strip() for m in str(raw).split(",") if m.strip()]


def _resolve_pipeline(cfg: ExperimentConfig, args: argparse.Namespace) -> ResolvedPipeline:
    """Merge pipeline defaults from config with explicit CLI overrides."""
    p = cfg.pipeline
    models = _parse_models(args.models) if args.models else list(p.models)
    seeds = list(args.seeds) if args.seeds else list(p.seeds)
    directions = [(d[0], d[1]) for d in p.directions]
    do_train = p.do_train if args.do_train is None else bool(args.do_train)
    force = p.force if args.force is None else bool(args.force)
    return ResolvedPipeline(
        models=models,
        seeds=seeds,
        directions=directions,
        do_train=do_train,
        force=force,
        stage1_threshold=float(p.stage1_threshold if args.threshold is None else args.threshold),
        benign_class=int(p.benign_class if args.benign_class is None else args.benign_class),
        summary_path=Path(args.summary_path or p.summary_path),
    )


def _multiclass_data_file(train_ds: str, test_ds: str) -> Path:
    processed = Path("data/processed")
    if train_ds == test_ds:
        return processed / train_ds / "data_multiclass.npz"
    return processed / f"cross_{train_ds}_to_{test_ds}_multiclass.npz"


def _stage_model_root(
    seed: int | None, train_ds: str, test_ds: str, model: str, multiclass: bool
) -> Path:
    suffix = "_multiclass" if multiclass else ""
    if seed is None:
        return Path("artifacts") / f"{train_ds}_to_{test_ds}{suffix}" / model
    return Path("artifacts") / f"seed{seed}" / f"{train_ds}_to_{test_ds}{suffix}" / model


def _trigger_training(
    config_path: str,
    pipeline: ResolvedPipeline,
    label_mode: str,
    logger,
) -> None:
    cmd = [
        sys.executable,
        "scripts/run_experiments.py",
        "--config",
        config_path,
        "--models",
        ",".join(pipeline.models),
        "--label-mode",
        label_mode,
        "--one-click",
        "--seeds",
        *[str(s) for s in pipeline.seeds],
    ]
    if pipeline.force:
        cmd.append("--force")
    logger.info("Dispatching training (label_mode=%s): %s", label_mode, " ".join(cmd))
    run_command(cmd, logger)


def _resolve_run_dir(model_root: Path) -> Path | None:
    if not model_root.exists():
        return None
    report = find_latest_report(model_root)
    if report is None:
        return None
    return report.parent


def _cascade_one(
    target: PipelineTarget,
    pipeline: ResolvedPipeline,
    logger,
) -> dict | None:
    stage1_root = _stage_model_root(
        target.seed, target.train_ds, target.test_ds, target.model, multiclass=False
    )
    stage2_root = _stage_model_root(
        target.seed, target.train_ds, target.test_ds, target.model, multiclass=True
    )

    stage1_run = _resolve_run_dir(stage1_root)
    stage2_run = _resolve_run_dir(stage2_root)
    if stage1_run is None:
        logger.warning("Stage-1 artifact missing for %s; skipping", target)
        return None
    if stage2_run is None:
        logger.warning("Stage-2 artifact missing for %s; skipping", target)
        return None

    data_file = _multiclass_data_file(target.train_ds, target.test_ds)
    if not data_file.exists():
        logger.warning("Multiclass data file missing: %s; skipping", data_file)
        return None

    output_dir = stage2_root.parent / "two_stage" / target.model
    cmd = [
        sys.executable,
        "scripts/eval_two_stage.py",
        "--stage1-run",
        str(stage1_run),
        "--stage2-run",
        str(stage2_run),
        "--data-file",
        str(data_file),
        "--output-dir",
        str(output_dir),
        "--threshold",
        str(pipeline.stage1_threshold),
        "--benign-class",
        str(pipeline.benign_class),
    ]
    logger.info("Cascade eval: %s", " ".join(cmd))
    run_command(cmd, logger)

    report_path = output_dir / "two_stage_report.json"
    if not report_path.exists():
        logger.warning("Cascade report did not materialize at %s", report_path)
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    baseline = report.get("baseline_multiclass_metrics", {})
    cascade = report.get("two_stage_metrics", {})
    delta = report.get("delta", {})
    return {
        "seed": target.seed,
        "train_dataset": target.train_ds,
        "test_dataset": target.test_ds,
        "model": target.model,
        "stage1_run": report.get("stage1_run"),
        "stage2_run": report.get("stage2_run"),
        "report_path": str(report_path),
        "baseline_accuracy": float(baseline.get("accuracy", 0.0)),
        "baseline_macro_f1": float(baseline.get("macro_f1", 0.0)),
        "baseline_weighted_f1": float(baseline.get("weighted_f1", 0.0)),
        "two_stage_accuracy": float(cascade.get("accuracy", 0.0)),
        "two_stage_macro_f1": float(cascade.get("macro_f1", 0.0)),
        "two_stage_weighted_f1": float(cascade.get("weighted_f1", 0.0)),
        "delta_macro_f1": float(delta.get("macro_f1", 0.0)),
        "delta_accuracy": float(delta.get("accuracy", 0.0)),
        "stage1_recall": float(report.get("stage1_stats", {}).get("stage1_recall", 0.0)),
        "stage1_far": float(report.get("stage1_stats", {}).get("stage1_far", 0.0)),
    }


def main() -> None:
    args = parse_args()
    logger = get_logger("two_stage_pipeline")

    cfg = load_config(args.config)
    pipeline = _resolve_pipeline(cfg, args)
    logger.info(
        "Resolved pipeline | models=%s seeds=%s directions=%s do_train=%s force=%s threshold=%.3f",
        pipeline.models,
        pipeline.seeds,
        pipeline.directions,
        pipeline.do_train,
        pipeline.force,
        pipeline.stage1_threshold,
    )

    if pipeline.do_train:
        _trigger_training(args.config, pipeline, label_mode="binary", logger=logger)
        _trigger_training(args.config, pipeline, label_mode="multiclass", logger=logger)

    targets: list[PipelineTarget] = []
    for seed in pipeline.seeds:
        for train_ds, test_ds in pipeline.directions:
            for model in pipeline.models:
                targets.append(
                    PipelineTarget(seed=seed, train_ds=train_ds, test_ds=test_ds, model=model)
                )

    summary: dict = {}
    for target in targets:
        row = _cascade_one(target, pipeline, logger)
        if row is None:
            continue
        key = f"seed{target.seed}__{target.train_ds}_to_{target.test_ds}__{target.model}"
        summary[key] = row

    save_json(summary, pipeline.summary_path)
    logger.info("Two-stage summary written to %s (%d rows)", pipeline.summary_path, len(summary))


if __name__ == "__main__":
    main()
