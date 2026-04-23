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


def _minimal_args(**overrides) -> Namespace:
    """Build a Namespace mirroring the current run_experiments.py CLI surface."""

    defaults = dict(
        config="configs/default.yaml",
        max_rows=None,
        models=None,
        force=False,
        one_click=False,
        seeds=None,
        cross_dataset_enhancements=True,
        output_dir="artifacts/experiments",
        label_mode="binary",
        # ``directions`` is populated by _apply_pipeline_config_defaults at
        # runtime; tests provide an explicit list so _resolve_experiments can
        # run without loading any config file.
        directions=[
            ("cicids2017", "cicids2017"),
            ("unsw_nb15", "unsw_nb15"),
            ("cicids2017", "unsw_nb15"),
        ],
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_run_experiments_uses_canonical_train_output_by_default() -> None:
    args = _minimal_args(one_click=False)
    cmd = run_experiments_script._build_train_command(
        args=args,
        train_ds="cicids2017",
        test_ds="unsw_nb15",
    )
    assert "--output-dir" not in cmd


def test_resolve_experiments_reflects_args_directions() -> None:
    args = _minimal_args(one_click=True)
    experiments = run_experiments_script._resolve_experiments(args)
    cmd = run_experiments_script._build_train_command(
        args=args,
        train_ds="cicids2017",
        test_ds="unsw_nb15",
    )
    assert experiments == [
        ("same_cicids", "cicids2017", "cicids2017"),
        ("same_unsw", "unsw_nb15", "unsw_nb15"),
        ("cross_cic_to_unsw", "cicids2017", "unsw_nb15"),
    ]
    # one_click means --models defaults to "all" inside the built train command.
    assert "--models" in cmd
    assert cmd[cmd.index("--models") + 1] == "all"


def test_parse_directions_flag_parses_train_test_pairs() -> None:
    parsed = run_experiments_script._parse_directions_flag(
        "cicids2017:cicids2017, unsw_nb15:unsw_nb15"
    )
    assert parsed == [
        ("cicids2017", "cicids2017"),
        ("unsw_nb15", "unsw_nb15"),
    ]
    assert run_experiments_script._parse_directions_flag(None) is None
    assert run_experiments_script._parse_directions_flag("") is None


def test_resolve_seeds_default_and_multi() -> None:
    no_seeds = Namespace(seeds=None)
    assert run_experiments_script._resolve_seeds(no_seeds) == [None]

    multi = Namespace(seeds=[42, 43, 44])
    assert run_experiments_script._resolve_seeds(multi) == [42, 43, 44]


def test_build_train_command_injects_seed_and_output_dir() -> None:
    args = _minimal_args()
    cmd = run_experiments_script._build_train_command(
        args=args,
        train_ds="cicids2017",
        test_ds="unsw_nb15",
        seed=43,
        output_dir=Path("artifacts/seed43/cicids2017_to_unsw_nb15"),
    )
    assert "--seed" in cmd
    assert cmd[cmd.index("--seed") + 1] == "43"
    assert "--output-dir" in cmd
    assert cmd[cmd.index("--output-dir") + 1] == "artifacts/seed43/cicids2017_to_unsw_nb15"


def test_cross_dataset_direction_auto_enables_enhancements() -> None:
    args = _minimal_args(cross_dataset_enhancements=True)
    cross_cmd = run_experiments_script._build_train_command(
        args=args, train_ds="cicids2017", test_ds="unsw_nb15"
    )
    same_cmd = run_experiments_script._build_train_command(
        args=args, train_ds="cicids2017", test_ds="cicids2017"
    )
    assert "--cross-dataset-enhancements" in cross_cmd
    assert "--cross-dataset-enhancements" not in same_cmd
    assert "--no-cross-dataset-enhancements" not in same_cmd


def test_no_cross_dataset_enhancements_flag_propagates() -> None:
    args = _minimal_args(cross_dataset_enhancements=False)
    cross_cmd = run_experiments_script._build_train_command(
        args=args, train_ds="cicids2017", test_ds="unsw_nb15"
    )
    assert "--no-cross-dataset-enhancements" in cross_cmd
    assert "--cross-dataset-enhancements" not in cross_cmd


def test_one_click_passes_resume_and_models() -> None:
    args = _minimal_args(one_click=True)
    cmd = run_experiments_script._build_train_command(
        args=args, train_ds="cicids2017", test_ds="cicids2017"
    )
    assert "--one-click" in cmd
    assert "--resume" in cmd
    assert "--models" in cmd
    assert cmd[cmd.index("--models") + 1] == "all"


def test_multiclass_label_mode_restricts_to_same_dataset_rows() -> None:
    args = _minimal_args(one_click=True, label_mode="multiclass")
    experiments = run_experiments_script._resolve_experiments(args)
    names = {e[0] for e in experiments}
    # Cross directions are dropped because multiclass has no cross-dataset mapping.
    assert "same_cicids" in names
    assert "same_unsw" in names
    assert "cross_cic_to_unsw" not in names


def test_build_train_command_injects_label_mode() -> None:
    args = _minimal_args(label_mode="multiclass")
    cmd = run_experiments_script._build_train_command(
        args=args, train_ds="cicids2017", test_ds="cicids2017"
    )
    assert "--label-mode" in cmd
    assert cmd[cmd.index("--label-mode") + 1] == "multiclass"
    # multiclass must not pair with cross-dataset enhancements (binary-only).
    assert "--cross-dataset-enhancements" not in cmd


def test_build_train_command_binary_keeps_no_label_mode_flag() -> None:
    args = _minimal_args()
    cmd = run_experiments_script._build_train_command(
        args=args, train_ds="cicids2017", test_ds="cicids2017"
    )
    # Binary is the default; no need to emit --label-mode.
    assert "--label-mode" not in cmd
