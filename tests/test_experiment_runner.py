from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from nids.config import ExperimentConfig
from nids.utils import run_layout
from scripts import run_experiments as run_experiments_script


def _write_report(model_root: Path, run_name: str = "20260407-120000_CNN-BiLSTM-SE_ClassWeights") -> Path:
    run_dir = model_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = run_dir / "report.json"
    report_path.write_text("{}", encoding="utf-8")
    return report_path


def _write_checkpoint(model_root: Path, run_name: str = "20260407-120000_CNN-BiLSTM-SE_ClassWeights") -> Path:
    run_dir = model_root / "runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "checkpoint_last.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    return checkpoint_path.parent


def test_legacy_experiment_roots_are_checked_for_existing_reports(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.runtime.output_dir = str(tmp_path / "artifacts")
    base_output_dir = Path(cfg.runtime.output_dir) / "cicids2017_to_unsw_nb15"

    legacy_root = Path(cfg.runtime.output_dir) / "experiments" / "cross_cic_to_unsw" / "cnn_bilstm_se"
    legacy_report = _write_report(legacy_root)

    candidate_roots = run_layout.candidate_model_roots(
        artifacts_root=cfg.runtime.output_dir,
        train_ds="cicids2017",
        test_ds="unsw_nb15",
        model_name="cnn_bilstm_se",
        base_output_dir=base_output_dir,
        explicit_output_dir=None,
        models_count=1,
    )

    assert run_layout.find_latest_report_in_roots(candidate_roots) == legacy_report


def test_legacy_experiment_roots_are_checked_for_resume_checkpoints(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.runtime.output_dir = str(tmp_path / "artifacts")
    base_output_dir = Path(cfg.runtime.output_dir) / "cicids2017_to_unsw_nb15"

    legacy_root = Path(cfg.runtime.output_dir) / "experiments" / "cross_cic_to_unsw" / "cnn_bilstm_se"
    checkpoint_run = _write_checkpoint(legacy_root)

    candidate_roots = run_layout.candidate_model_roots(
        artifacts_root=cfg.runtime.output_dir,
        train_ds="cicids2017",
        test_ds="unsw_nb15",
        model_name="cnn_bilstm_se",
        base_output_dir=base_output_dir,
        explicit_output_dir=None,
        models_count=1,
    )

    assert run_layout.find_latest_checkpoint_run_in_roots(candidate_roots) == checkpoint_run


def test_run_experiments_uses_canonical_train_output_by_default() -> None:
    args = Namespace(
        config="configs/default.yaml",
        max_rows=None,
        models=None,
        skip_existing=True,
        auto_preprocess=True,
        force=False,
        one_click=False,
        cross_only=False,
        profile="default",
    )

    cmd = run_experiments_script._build_train_command(
        args=args,
        train_ds="cicids2017",
        test_ds="unsw_nb15",
    )

    assert "--output-dir" not in cmd


def test_laptop_3060_profile_runs_cross_dataset_subset() -> None:
    args = Namespace(
        config="configs/default.yaml",
        max_rows=None,
        models=None,
        skip_existing=False,
        auto_preprocess=False,
        force=False,
        one_click=True,
        cross_only=False,
        profile="laptop_3060",
    )

    resolved = run_experiments_script._apply_profile_defaults(args)
    experiments = run_experiments_script._resolve_experiments(resolved)
    cmd = run_experiments_script._build_train_command(
        args=resolved,
        train_ds="cicids2017",
        test_ds="unsw_nb15",
    )

    assert experiments == [
        ("cross_cic_to_unsw", "cicids2017", "unsw_nb15"),
    ]
    assert "--models" in cmd
    assert cmd[cmd.index("--models") + 1] == "cnn_bilstm_se,random_forest,xgboost"
