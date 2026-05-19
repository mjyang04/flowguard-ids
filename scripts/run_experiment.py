"""Cross-platform CLAN experiment driver.

Examples:
    # One command, sequential and safe: both datasets, seeds 42/43/44.
    python scripts/run_experiment.py --datasets both

    # Same run, but the datasets live outside the repository.
    python scripts/run_experiment.py --datasets both \
        --lycos-csv "E:\\datasets\\lycos.csv" \
        --cicids-source "E:\\datasets\\cicids2017"

    # Resume an interrupted full run without redoing completed outputs.
    python scripts/run_experiment.py --datasets both --skip-existing

    # Preview commands without running training.
    python scripts/run_experiment.py --datasets both --dry-run

Each dataset/seed job runs stages in order:
    train -> eval -> finetune_sweep

The default is intentionally sequential: it finishes seed 42 before seed 43,
and so on. Keep --parallel-seeds at 1 on a single 6 GB GPU.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = (42, 43, 44)
DEFAULT_STAGES = ("train", "eval", "finetune")
DATASET_ALIASES = {
    "lycos": ("lycos",),
    "lycos2017": ("lycos",),
    "cicids": ("cicids",),
    "cicids2017": ("cicids",),
    "both": ("lycos", "cicids"),
    "all": ("lycos", "cicids"),
}


class StageFailure(RuntimeError):
    """Raised when one child training/evaluation command fails."""

    def __init__(self, cmd: list[str], log_path: Path, returncode: int) -> None:
        """Capture the failed command, its log path, and process exit code."""
        self.cmd = cmd
        self.log_path = log_path
        self.returncode = returncode
        printable = " ".join(cmd)
        super().__init__(
            f"Command failed with exit code {returncode}: {printable}\n"
            f"See log: {log_path}"
        )


def parse_args() -> argparse.Namespace:
    """Parse cross-dataset experiment driver CLI arguments."""
    parser = argparse.ArgumentParser("Run CLAN experiments across datasets/seeds")
    parser.add_argument(
        "--datasets",
        default="both",
        choices=sorted(DATASET_ALIASES),
        help="Dataset selector. Use 'both' for Lycos2017 plus CICIDS2017.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Pretraining seeds to run.",
    )
    parser.add_argument(
        "--parallel-seeds",
        type=int,
        default=1,
        help="How many seed pipelines to run at the same time.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=DEFAULT_STAGES,
        default=list(DEFAULT_STAGES),
        help="Stages to run, in the given order.",
    )
    parser.add_argument(
        "--config-dir",
        default="configs",
        help="Directory containing lycos.yaml and cicids.yaml.",
    )
    parser.add_argument(
        "--lycos-csv",
        default=None,
        help="Override Lycos2017 CSV path without editing configs/lycos.yaml.",
    )
    parser.add_argument(
        "--cicids-source",
        default=None,
        help="Override CICIDS2017 source path without editing configs/cicids.yaml.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages whose expected output file already exists.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/run_experiment",
        help="Directory for per-dataset/seed logs.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch child scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def config_path(config_dir: str, dataset: str) -> Path:
    """Return the dataset config path, raising if it does not exist."""
    path = ROOT / config_dir / f"{dataset}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return path


def maybe_relative(path: Path) -> str:
    """Render a path relative to the repository root when possible."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dataset_override(args: argparse.Namespace, dataset: str) -> str | None:
    """Return the CLI-provided dataset source override for one dataset."""
    if dataset == "lycos":
        return args.lycos_csv
    if dataset == "cicids":
        return args.cicids_source
    return None


def effective_config_path(args: argparse.Namespace, dataset: str, log_dir: Path) -> Path:
    """Return the base config or a temporary config with source-path overrides."""
    base_config = config_path(args.config_dir, dataset)
    override = dataset_override(args, dataset)
    if not override:
        return base_config

    raw = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    raw.setdefault("data", {})["csv_path"] = override

    out_dir = log_dir / "resolved_configs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset}.yaml"
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return out_path


def artifact_dir(config: Path, seed: int) -> Path:
    """Resolve the artifact directory implied by a config file and seed."""
    sys.path.insert(0, str(ROOT))
    from nids.config import load_config

    cfg = load_config(config)
    return ROOT / cfg.runtime.output_dir / cfg.data.dataset / "clan" / f"seed{seed}"


def output_for_stage(config: Path, seed: int, stage: str) -> Path:
    """Return the file that indicates completion for a pipeline stage."""
    run_dir = artifact_dir(config, seed)
    if stage == "train":
        return run_dir / "clan.pt.tar"
    if stage == "eval":
        return run_dir / "eval_report.json"
    if stage == "finetune":
        return run_dir / "finetune_summary.csv"
    raise ValueError(f"Unknown stage: {stage}")


def command_for_stage(python: str, config: Path, seed: int, stage: str) -> list[str]:
    """Build the child Python command for one dataset, seed, and stage."""
    config_arg = maybe_relative(config)
    if stage == "train":
        return [python, "scripts/train.py", "--config", config_arg, "--seed", str(seed)]
    if stage == "eval":
        return [python, "scripts/eval.py", "--config", config_arg, "--seed", str(seed)]
    if stage == "finetune":
        return [
            python,
            "scripts/finetune_sweep.py",
            "--config",
            config_arg,
            "--pretrain-seed",
            str(seed),
        ]
    raise ValueError(f"Unknown stage: {stage}")


def tail_text(path: Path, lines: int = 60) -> str:
    """Return the last ``lines`` of a text log, or an empty string if absent."""
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def run_command(cmd: list[str], log_path: Path, *, dry_run: bool) -> None:
    """Run one child command while streaming stdout and stderr to a log file."""
    printable = " ".join(cmd)
    if dry_run:
        print(f"[dry-run] {printable}")
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} ====\n")
        log.write(f"$ {printable}\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if proc.returncode != 0:
        raise StageFailure(cmd, log_path, proc.returncode)


def run_seed_pipeline(
    *,
    dataset: str,
    config: Path,
    seed: int,
    stages: list[str],
    python: str,
    log_dir: Path,
    skip_existing: bool,
    dry_run: bool,
) -> None:
    """Run the requested stages sequentially for a single dataset/seed pair."""
    log_path = log_dir / f"{dataset}_seed{seed}.log"
    print(f"[start] {dataset} seed={seed} stages={','.join(stages)} log={log_path}")
    for stage in stages:
        expected = output_for_stage(config, seed, stage)
        if skip_existing and expected.exists():
            print(f"[skip] {dataset} seed={seed} {stage}: {expected}")
            continue

        cmd = command_for_stage(python, config, seed, stage)
        print(f"[run] {dataset} seed={seed} {stage}")
        run_command(cmd, log_path, dry_run=dry_run)
        print(f"[done] {dataset} seed={seed} {stage}")
    print(f"[complete] {dataset} seed={seed}")


def run_dataset(args: argparse.Namespace, dataset: str) -> None:
    """Run all requested seed pipelines for one resolved dataset selector."""
    log_dir = ROOT / args.log_dir
    config = effective_config_path(args, dataset, log_dir)
    max_workers = max(1, min(args.parallel_seeds, len(args.seeds)))

    print(f"\n==== dataset={dataset} config={maybe_relative(config)} ====")
    print(f"seeds={args.seeds} parallel_seeds={max_workers} stages={args.stages}")
    if max_workers > 1:
        print("[warn] Parallel seeds launch independent Python processes.")
        print("[warn] On one 6 GB GPU, this may hit CUDA out-of-memory.")

    if max_workers == 1:
        for seed in args.seeds:
            run_seed_pipeline(
                dataset=dataset,
                config=config,
                seed=seed,
                stages=args.stages,
                python=args.python,
                log_dir=log_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
        return

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                run_seed_pipeline,
                dataset=dataset,
                config=config,
                seed=seed,
                stages=args.stages,
                python=args.python,
                log_dir=log_dir,
                skip_existing=args.skip_existing,
                dry_run=args.dry_run,
            )
            for seed in args.seeds
        ]
        for future in as_completed(futures):
            future.result()


def main() -> None:
    """Coordinate requested dataset runs and print helpful failure log tails."""
    args = parse_args()
    datasets = DATASET_ALIASES[args.datasets]
    try:
        for dataset in datasets:
            run_dataset(args, dataset)
    except StageFailure as exc:
        print(f"\n[failed] {exc}", file=sys.stderr)
        tail = tail_text(exc.log_path)
        if tail:
            print("\n==== log tail ====", file=sys.stderr)
            print(tail, file=sys.stderr)
            print("==== end log tail ====", file=sys.stderr)
        sys.exit(exc.returncode)
    print("\nAll requested runs complete.")


if __name__ == "__main__":
    main()
