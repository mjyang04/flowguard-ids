"""Paper-faithful fine-tune sweep: 8 shot counts × N sample seeds.

Usage:
    python scripts/finetune_sweep.py --config configs/lycos.yaml --pretrain-seed 42

The CLAN paper (Wilkie et al., IEEE CSR 2025, §V-A) reports:
    "reported results were averaged over 10 runs, with a different seed
     being used to sample the subset of training data each time"

This script implements that protocol and writes a summary CSV with the
mean ± std of macro-F1 per shot count, which is the number that goes into
thesis Table 4.6.

On RTX 3060 6 GB this takes roughly:
    8 shots × 10 sample seeds × ~2 min/run ≈ 160 minutes per pretrain seed.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nids.config import ExperimentConfig, load_config  # noqa: E402
from nids.utils import ensure_dir, get_logger, save_json, seed_everything  # noqa: E402
from scripts.finetune import resolve_device, run_finetune  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Run the configured shot-count and sample-seed fine-tuning sweep."""
    parser = argparse.ArgumentParser("CLAN fine-tune sweep (paper protocol)")
    parser.add_argument("--config", type=str, default="configs/lycos.yaml")
    parser.add_argument("--pretrain-seed", type=int, default=42)
    args = parser.parse_args()

    cfg: ExperimentConfig = load_config(args.config)
    cfg = replace(cfg, runtime=replace(cfg.runtime, seed=args.pretrain_seed))

    device = resolve_device(cfg.runtime.device)
    shot_list = cfg.finetune.samples_per_class

    run_dir = Path(cfg.runtime.output_dir) / cfg.data.dataset / "clan" / f"seed{cfg.runtime.seed}"
    checkpoint_path = run_dir / "clan.pt.tar"
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Pretrain checkpoint not found: {checkpoint_path}. "
            f"Run scripts/train.py first."
        )

    ensure_dir(run_dir)
    summary_path = run_dir / "finetune_summary.csv"
    per_run_log: list[dict] = []

    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["shots", "mean_macro_f1", "std_macro_f1",
                         "mean_accuracy", "std_accuracy", "n_runs"])

        for shots in shot_list:
            f1s: list[float] = []
            accs: list[float] = []
            for sample_seed in range(cfg.finetune.n_sample_seeds):
                seed_everything(sample_seed)
                logger.info("shots=%d sample_seed=%d", shots, sample_seed)
                metrics = run_finetune(
                    cfg, shots=shots, sample_seed=sample_seed,
                    device=device, checkpoint_path=checkpoint_path,
                )
                f1 = float(metrics.get("macro_f1", 0.0))
                acc = float(metrics.get("accuracy", 0.0))
                f1s.append(f1)
                accs.append(acc)
                per_run_log.append({
                    "shots": shots,
                    "sample_seed": sample_seed,
                    "macro_f1": f1,
                    "accuracy": acc,
                })
                logger.info("shots=%d seed=%d macro_f1=%.4f accuracy=%.4f",
                            shots, sample_seed, f1, acc)

            mean_f1 = statistics.mean(f1s)
            std_f1 = statistics.stdev(f1s) if len(f1s) > 1 else 0.0
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            writer.writerow([shots, f"{mean_f1:.6f}", f"{std_f1:.6f}",
                             f"{mean_acc:.6f}", f"{std_acc:.6f}", len(f1s)])
            logger.info("shots=%d mean_macro_f1=%.4f ± %.4f  (n=%d)",
                        shots, mean_f1, std_f1, len(f1s))

    save_json({"per_run": per_run_log,
               "dataset": cfg.data.dataset,
               "pretrain_seed": args.pretrain_seed,
               "n_sample_seeds": cfg.finetune.n_sample_seeds},
              run_dir / "finetune_per_run.json")
    logger.info("summary written to %s", summary_path)


if __name__ == "__main__":
    main()
