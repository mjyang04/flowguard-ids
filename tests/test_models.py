import pytest

torch = pytest.importorskip("torch")
from nids.models.cnn_bilstm_se import CNNBiLSTMSE


def test_cnn_bilstm_se_binary_shape():
    model = CNNBiLSTMSE(
        input_dim=55,
        num_classes=2,
        conv_channels=[16, 32],
        conv_kernel_sizes=[3, 3],
        conv_pool_sizes=[2, 2],
        lstm_hidden_size=32,
        lstm_num_layers=1,
        dropout=0.1,
        bidirectional=True,
        use_attention=False,
        use_se=True,
    )
    x = torch.randn(4, 55)
    y = model(x)
    assert y.shape == (4,)


def test_cnn_bilstm_se_multiclass_shape():
    model = CNNBiLSTMSE(
        input_dim=55,
        num_classes=4,
        conv_channels=[16, 32],
        conv_kernel_sizes=[3, 3],
        conv_pool_sizes=[2, 2],
        lstm_hidden_size=32,
        lstm_num_layers=1,
        dropout=0.1,
        bidirectional=True,
        use_attention=True,
        use_se=True,
    )
    x = torch.randn(5, 55)
    y = model(x)
    assert y.shape == (5, 4)
