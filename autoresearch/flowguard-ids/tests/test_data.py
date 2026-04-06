import numpy as np
import pytest

pytest.importorskip("torch")
from nids.data.dataset import NIDSDataset, create_dataloaders


def test_dataset_and_dataloader_shapes():
    X_train = np.random.randn(64, 16).astype(np.float32)
    X_val = np.random.randn(16, 16).astype(np.float32)
    X_test = np.random.randn(16, 16).astype(np.float32)
    y_train = np.random.randint(0, 2, size=64, dtype=np.int64)
    y_val = np.random.randint(0, 2, size=16, dtype=np.int64)
    y_test = np.random.randint(0, 2, size=16, dtype=np.int64)

    ds = NIDSDataset(X_train, y_train)
    assert len(ds) == 64
    x0, y0 = ds[0]
    assert x0.shape[0] == 16
    assert int(y0) in (0, 1)

    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test, batch_size=8, num_workers=0
    )
    xb, yb = next(iter(train_loader))
    assert xb.shape == (8, 16)
    assert yb.shape == (8,)
    assert len(list(val_loader)) > 0
    assert len(list(test_loader)) > 0
