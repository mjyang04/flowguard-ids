from nids.config import ExperimentConfig, load_config


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.name == "cnn_bilstm_se"
    assert cfg.data.train_dataset in {"cicids2017", "unsw_nb15"}


def test_new_config_fields_have_defaults():
    cfg = ExperimentConfig()
    assert cfg.model.se_reduction == 16
    assert cfg.training.loss_type == "bce"
    assert cfg.training.focal_alpha == 0.25
    assert cfg.training.focal_gamma == 2.0
    assert cfg.training.label_smoothing == 0.0


def test_cross_dataset_config_loads():
    cfg = load_config("configs/cross_dataset.yaml")
    assert cfg.training.use_auc_loss is True
    assert cfg.training.use_platt_calibration is True
    assert cfg.training.label_smoothing == 0.05


def test_same_dataset_config_loads():
    cfg = load_config("configs/same_dataset.yaml")
    assert cfg.training.use_auc_loss is False
    assert cfg.training.use_platt_calibration is False
    assert cfg.training.label_smoothing == 0.0
