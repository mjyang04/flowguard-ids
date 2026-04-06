from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nids.config import load_config
from nids.models.registry import create_model
from nids.utils.io import save_json
from nids.utils.logging import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained model to TorchScript/ONNX")
    parser.add_argument("--model", required=True, help="Path to model weights")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-file", default=None, help="Optional data npz for input shape")
    parser.add_argument("--output-dir", default="artifacts/export")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("export_model")
    cfg = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_file:
        payload = np.load(args.data_file, allow_pickle=True)
        input_dim = int(payload["X_train"].shape[1])
    else:
        input_dim = cfg.model.input_dim
    cfg.model.input_dim = input_dim

    model = create_model(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    example_inputs = torch.randn(1, input_dim, device=device)

    torchscript_path = output_dir / "model.pt"
    scripted = torch.jit.trace(model, example_inputs)
    scripted.save(str(torchscript_path))
    logger.info("TorchScript exported: %s", torchscript_path)

    onnx_path = output_dir / "model.onnx"
    try:
        torch.onnx.export(
            model,
            example_inputs,
            str(onnx_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        onnx_ok = True
        logger.info("ONNX exported: %s", onnx_path)
    except Exception as exc:  # noqa: BLE001
        onnx_ok = False
        logger.warning("ONNX export skipped: %s", exc)

    save_json(
        {
            "weights": args.model,
            "torchscript": str(torchscript_path),
            "onnx": str(onnx_path) if onnx_ok else None,
            "input_dim": input_dim,
            "num_classes": cfg.model.num_classes,
        },
        output_dir / "export_summary.json",
    )


if __name__ == "__main__":
    main()
