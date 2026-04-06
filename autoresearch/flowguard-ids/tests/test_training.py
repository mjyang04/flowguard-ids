from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
from nids.config import TrainingConfig
from nids.data.dataset import create_dataloaders
from nids.models.cnn_bilstm_se import CNNBiLSTMSE
from nids.training.trainer import Trainer


def test_trainer_fit_runs(tmp_path: Path):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 32)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int64)

    X_train, X_val, X_test = X[:80], X[80:100], X[100:]
    y_train, y_val, y_test = y[:80], y[80:100], y[100:]

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=16, num_workers=0
    )
    model = CNNBiLSTMSE(
        input_dim=32,
        num_classes=2,
        conv_channels=[8, 16],
        conv_kernel_sizes=[3, 3],
        conv_pool_sizes=[2, 2],
        lstm_hidden_size=16,
        lstm_num_layers=1,
        dropout=0.1,
        bidirectional=True,
        use_attention=False,
        use_se=True,
    )
    trainer = Trainer(
        TrainingConfig(
            num_epochs=2,
            learning_rate=1e-3,
            weight_decay=1e-4,
            optimizer="adam",
            scheduler="plateau",
            scheduler_factor=0.5,
            scheduler_patience=1,
            min_learning_rate=1e-6,
            early_stopping_patience=2,
            early_stopping_delta=1e-5,
            gradient_clip=1.0,
            amp=False,
            selection_metric="avg_attack_recall",
        ),
        output_dir=tmp_path,
    )
    summary = trainer.fit(model, train_loader, val_loader, num_classes=2)
    assert Path(summary.best_model_path).exists()
    result = trainer.evaluate(model, test_loader, criterion=None, num_classes=2)
    assert "accuracy" in result.metrics
