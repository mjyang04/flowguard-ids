# FlowGuard IDS

Graduation-design research project: **reproduce CLAN on Lycos2017 and compare against 7 self-supervised NIDS baselines**.

- **Paper**: Wilkie et al., *"CLAN: Contrastive Self-Supervised NIDS Using Augmented Negative Pairs"*, IEEE CSR 2025.
- **Upstream code**: https://github.com/jackwilkie/CLAN (Apache-2.0).
- **Dataset**: [Lycos2017](https://lycos-ids.univ-lemans.fr) — a relabelled CICIDS2017 that fixes the labelling issues documented by Engelen et al. (WTMC 2021) and Rosay et al. (2023).

> **Status (2026-04-23)**: repository reset from the previous CNN-BiLSTM-SE-Transformer / SHAP / XI2S-cascade direction. Only reusable infrastructure (config, metrics, logging, I/O, reproducibility) remains. CLAN porting happens in the next session. See `CLAUDE.md` for the detailed porting plan.

## Thesis Story

1. **Main method**: faithful CLAN reproduction (CLDNN encoder + `CLANLoss` + L2-normalized benign embeddings).
2. **Comparison table**: CLAN vs SimCLR / Barlow Twins / BYOL / VICReg / SimSiam / ConFlow / SSCL-IDS under the same encoder, dataset, and augmentation pipeline.
3. **Ablation**: CLAN's margin `m`, augmentation type, augmentation strength, encoder depth, few-shot sample count, L2-normalization on/off.
4. **Downstream few-shot multiclass**: sweep shots-per-class ∈ {8, 16, 32, 64, 128, 256, 512, 1024}.

## Repository Layout

```text
flowguard-ids/
├── configs/default.yaml           # Single config: data + model + loss + aug + training + finetune + runtime
├── nids/                          # Package (installed via setup.py)
│   ├── config.py                  # Frozen-dataclass config + YAML loader
│   ├── data/                      # [stub] Lycos2017Dataset + DataLoader (to be filled)
│   ├── models/                    # [stub] CLDNN encoder + ablation variants (to be filled)
│   ├── training/                  # [stub] CLANLoss + 7 SSL baseline losses + Trainer (to be filled)
│   ├── evaluation/
│   │   ├── metrics.py             # compute_nids_metrics (AUC/PR-AUC/F1/MCC/ECE/FAR-recall)
│   │   └── latency.py             # Inference latency benchmarking
│   └── utils/                     # logging, I/O, reproducibility
├── tests/
│   ├── conftest.py
│   └── test_metrics.py            # 9 tests, pure numpy — passes without torch
├── requirements.txt
├── setup.py
├── CLAUDE.md                      # Internal project instructions (loaded into Claude sessions)
└── README.md                      # This file
```

Model / loss / data / training modules exist as import-stable empty packages; they are filled in the follow-up session by porting code from the upstream CLAN repo.

## Quick Start (after CLAN port)

```bash
# 1. Install
pip install -e .

# 2. Place the Lycos2017 dataset (one CSV or the preprocessed Drive bundle)
#    at:  data/raw/lycos.csv

# 3. Tests (pure numpy, no torch needed)
pytest -q

# 4. Train CLAN on Lycos2017 (command becomes available after scripts/ is ported)
# python scripts/train.py --config configs/default.yaml

# 5. Run a head-to-head SSL baseline comparison
# python scripts/train.py --config configs/default.yaml loss.name=simclr
# python scripts/train.py --config configs/default.yaml loss.name=barlow_twins
# ... (7 baselines)

# 6. Few-shot multiclass fine-tune
# python scripts/finetune.py --config configs/default.yaml
```

## Configuration

Everything is driven by `configs/default.yaml`. Seven sections:

| Section | Purpose |
|---|---|
| `data` | Dataset path, batch size, splits, scaler, benign-label string |
| `model` | CLDNN encoder hyperparameters (embedding dim, conv/LSTM shapes, L2-norm toggle) |
| `loss` | Which SSL loss to use — `clan` by default; switch to one of 7 baselines for comparison |
| `augmentation` | Augmentation strategy for negative views (Gaussian noise, feature dropout) |
| `training` | Epochs, optimizer, scheduler, AMP, early stopping, selection metric |
| `finetune` | Few-shot sweep schedule for downstream multiclass evaluation |
| `runtime` | Output dir, seed, device |

## Evaluation Metrics

`nids.evaluation.metrics.compute_nids_metrics` (9 unit tests, pure numpy) computes:

### Binary (requires `y_score`)
- `pr_auc`, `roc_auc`, `ece`
- `best_f1`, `best_f1_threshold`
- `recall_at_far_1pct`, `recall_at_far_5pct`

### Multiclass
- `accuracy`, `macro_f1`, `weighted_f1`, `mcc`
- `per_class_f1`, `per_class_recall`, `per_class_precision`
- `confusion_matrix`

### Shared
- `avg_attack_recall`, `attack_macro_precision`, `benign_false_alarm_rate`, `attack_miss_rate`

## Requirements

- Python 3.10+
- PyTorch 2.5+ (once CLAN is ported; current infrastructure does not require torch)
- CUDA-capable GPU recommended (6 GB VRAM is enough — CLAN uses a lightweight CLDNN)

Install via `pip install -r requirements.txt` or `pip install -e .`.

## Testing

```bash
pytest -q
# Current baseline: 9 passed (all metrics tests; no torch required).
# After CLAN port: add tests for Lycos2017 loader, CLAN loss, encoder output shapes, trainer step.
```

## Citations

- J. Wilkie, H. Hindy, C. Tachtatzis, R. Atkinson. *CLAN: Contrastive Self-Supervised NIDS Using Augmented Negative Pairs.* IEEE CSR 2025. DOI:10.1109/CSR64739.2025.11129979
- L. Lanvin, P.-F. Gimenez, Y. Han, F. Majorczyk, L. Me, E. Totel. *Errors in the CICIDS2017 Dataset and the Significant Differences in Detection Performances It Makes.* Springer 2023.
- G. Engelen, V. Rimmer, W. Joosen. *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* WTMC 2021.
