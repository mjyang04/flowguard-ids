from __future__ import annotations

from pathlib import Path
from typing import Iterable


def find_latest_report(model_root: str | Path) -> Path | None:
    root = Path(model_root)
    runs_dir = root / "runs"
    if runs_dir.exists():
        candidates = sorted(
            [p for p in runs_dir.glob("*/report.json") if p.is_file()],
            key=lambda x: x.parent.name,
        )
        if candidates:
            return candidates[-1]
    legacy_report = root / "report.json"
    if legacy_report.exists():
        return legacy_report
    return None


def find_latest_best_model(model_root: str | Path, suffix: str = ".pt") -> Path | None:
    root = Path(model_root)
    runs_dir = root / "runs"
    if runs_dir.exists():
        candidates = sorted(
            [p for p in runs_dir.glob(f"*/best_model{suffix}") if p.is_file()],
            key=lambda x: x.parent.name,
        )
        if candidates:
            return candidates[-1]
    legacy_model = root / f"best_model{suffix}"
    if legacy_model.exists():
        return legacy_model
    return None


def find_latest_checkpoint_run(model_root: str | Path) -> Path | None:
    root = Path(model_root)
    runs_dir = root / "runs"
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


def experiment_group_name(train_ds: str, test_ds: str) -> str | None:
    mapping = {
        ("cicids2017", "cicids2017"): "same_cicids",
        ("cicids2017", "unsw_nb15"): "cross_cic_to_unsw",
        ("unsw_nb15", "cicids2017"): "cross_unsw_to_cic",
    }
    return mapping.get((train_ds, test_ds))


def canonical_model_root(
    artifacts_root: str | Path,
    train_ds: str,
    test_ds: str,
    model_name: str,
) -> Path:
    return Path(artifacts_root) / f"{train_ds}_to_{test_ds}" / model_name


def resolve_model_root_for_write(
    base_output_dir: str | Path,
    explicit_output_dir: str | Path | None,
    models_count: int,
    model_name: str,
) -> Path:
    base_dir = Path(base_output_dir)
    if explicit_output_dir and models_count == 1:
        return base_dir
    return base_dir / model_name


def candidate_model_roots(
    *,
    artifacts_root: str | Path,
    train_ds: str,
    test_ds: str,
    model_name: str,
    base_output_dir: str | Path,
    explicit_output_dir: str | Path | None,
    models_count: int,
) -> list[Path]:
    candidates: list[Path] = []
    write_root = resolve_model_root_for_write(
        base_output_dir=base_output_dir,
        explicit_output_dir=explicit_output_dir,
        models_count=models_count,
        model_name=model_name,
    )
    candidates.append(write_root)
    if explicit_output_dir and models_count == 1 and write_root.name != model_name:
        candidates.append(write_root / model_name)
    candidates.append(canonical_model_root(artifacts_root, train_ds, test_ds, model_name))

    exp_group = experiment_group_name(train_ds, test_ds)
    if exp_group is not None:
        candidates.append(Path(artifacts_root) / "experiments" / exp_group / model_name)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def find_latest_report_in_roots(model_roots: Iterable[str | Path]) -> Path | None:
    for model_root in model_roots:
        report_path = find_latest_report(model_root)
        if report_path is not None:
            return report_path
    return None


def find_latest_best_model_in_roots(
    model_roots: Iterable[str | Path],
    suffix: str = ".pt",
) -> Path | None:
    for model_root in model_roots:
        model_path = find_latest_best_model(model_root, suffix=suffix)
        if model_path is not None:
            return model_path
    return None


def find_latest_checkpoint_run_in_roots(model_roots: Iterable[str | Path]) -> Path | None:
    for model_root in model_roots:
        run_dir = find_latest_checkpoint_run(model_root)
        if run_dir is not None:
            return run_dir
    return None
