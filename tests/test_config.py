from nids.config import load_config


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.name == "cnn_bilstm_se"
    assert cfg.data.train_dataset in {"cicids2017", "unsw_nb15"}
