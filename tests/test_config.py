from nids.config import ExperimentConfig, load_config


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.name == "cnn_bilstm_se"
    assert cfg.data.train_dataset in {"cicids2017", "unsw_nb15"}


def test_default_config_disables_cross_dataset_enhancements():
    """Cross-dataset enhancements are off by default — they are opt-in via CLI."""

    cfg = load_config("configs/default.yaml")
    assert cfg.training.use_auc_loss is False
    assert cfg.training.use_platt_calibration is False
    assert cfg.training.label_smoothing == 0.0


def test_new_config_fields_have_defaults():
    cfg = ExperimentConfig()
    assert cfg.model.se_reduction == 16
    assert cfg.training.loss_type == "bce"
    assert cfg.training.focal_alpha == 0.25
    assert cfg.training.focal_gamma == 2.0
    assert cfg.training.label_smoothing == 0.0
