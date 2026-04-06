from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader


MODULE_PATH = Path(__file__).resolve().parents[1] / "autoresearch" / "prepare.py"
SPEC = spec_from_file_location("autoresearch_prepare", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DummyBinaryModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0]


def test_autoresearch_prepare_exposes_v2_binary_metrics():
    features = np.array(
        [
            [-4.0, 0.0],
            [-2.0, 0.0],
            [-0.2, 0.0],
            [0.1, 0.0],
            [0.4, 0.0],
            [1.2, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)

    loader = DataLoader(MODULE.NIDSDataset(features, labels), batch_size=4, shuffle=False)
    metrics = MODULE.evaluate_nids(
        model=DummyBinaryModel(),
        loader=loader,
        device=torch.device("cpu"),
        num_classes=2,
        use_amp=False,
    )

    for key in [
        "avg_attack_recall",
        "pr_auc",
        "roc_auc",
        "recall_at_far_1pct",
        "recall_at_far_5pct",
        "benign_false_alarm_rate",
    ]:
        assert key in metrics

    assert 0.0 <= float(metrics["pr_auc"]) <= 1.0
    assert 0.0 <= float(metrics["roc_auc"]) <= 1.0
