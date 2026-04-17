"""Print parameter counts for every registered deep model.

Runs on CPU only — safe for local Mac verification. Use this to check that
the ``~0.72 M`` claim in the thesis matches the actual ``cnn_bilstm_se``
parameter count, and to report the Najar baseline size in Appendix D.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from nids.config import load_config
from nids.models.registry import create_model

DEEP_MODELS = [
    "cnn_bilstm",
    "cnn_bilstm_se",
    "cnn_bilstm_attention",
    "cnn_bilstm_at",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report parameter counts for deep models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Override input_dim (defaults to cfg.model.input_dim, typically 55)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="num_classes passed to the model constructor (default: binary)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEEP_MODELS,
        help="Subset of deep models to benchmark",
    )
    return parser.parse_args()


def _count_params(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.input_dim is not None:
        cfg.model.input_dim = int(args.input_dim)
    cfg.model.num_classes = int(args.num_classes)

    rows: list[tuple[str, int, int]] = []
    for name in args.models:
        cfg.model.name = name
        try:
            model = create_model(cfg.model)
        except Exception as exc:  # pragma: no cover - diagnostic path
            rows.append((name, -1, -1))
            print(f"[ERROR] {name}: {exc}", file=sys.stderr)
            continue
        total, trainable = _count_params(model)
        rows.append((name, total, trainable))

    col_name = max(len("model"), *(len(r[0]) for r in rows))
    header = f"{'model'.ljust(col_name)}  {'total':>12}  {'trainable':>12}  {'approx_M':>10}"
    print(header)
    print("-" * len(header))
    for name, total, trainable in rows:
        if total < 0:
            print(f"{name.ljust(col_name)}  {'ERR':>12}  {'ERR':>12}  {'-':>10}")
            continue
        print(
            f"{name.ljust(col_name)}  {total:>12,d}  {trainable:>12,d}  "
            f"{total / 1e6:>10.3f}"
        )


if __name__ == "__main__":
    main()
