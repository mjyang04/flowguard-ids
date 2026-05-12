# FlowGuard IDS

Graduation-design research project: **reproduce CLAN on Lycos2017 and run a
controlled Lycos2017-vs-CICIDS2017 dataset-integrity audit**.

- **Paper**: Wilkie et al., *"CLAN: Contrastive Self-Supervised NIDS Using
  Augmented Negative Pairs"*, IEEE CSR 2025.
- **Upstream code**: https://github.com/jackwilkie/CLAN (Apache-2.0).
- **Main dataset**: [Lycos2017](https://lycos-ids.univ-lemans.fr), a relabelled
  CICIDS2017 that fixes the labelling issues documented by Engelen et al. and
  Rosay et al.
- **Control dataset**: original CICIDS2017, intentionally consumed without
  relabelling so the noisy-release effect can be measured.

## Thesis Scope

The final thesis scope is intentionally narrow enough to finish:

1. **Main method**: faithful CLAN reproduction with the upstream
   `ContrastiveMLP` encoder, `CLANLoss`, uniform-resample augmentation, and
   centroid-based anomaly score.
2. **Dataset audit**: run the same CLAN pipeline on Lycos2017 and original
   CICIDS2017 with matched seeds and hyperparameters.
3. **Few-shot curve**: sweep shots per class over
   `{8, 16, 32, 64, 128, 256, 512, 1024}`.
4. **Future work**: the seven SSL baselines from the CLAN paper
   (SimCLR, Barlow Twins, BYOL, VICReg, SimSiam, ConFlow, SSCL-IDS) are discussed
   in the literature review but are **not implemented** in this thesis.

The scripts intentionally expose CLAN only; baseline losses are not selectable
from config or CLI.

## Repository Layout

```text
flowguard-ids/
├── configs/
│   ├── default.yaml       # Base CLAN config
│   ├── lycos.yaml         # Lycos2017 dataset profile
│   └── cicids.yaml        # CICIDS2017 noisy-control profile
├── nids/
│   ├── config.py          # Frozen-dataclass config + YAML loader
│   ├── data/              # Lycos/CICIDS loaders, split logic, DataLoader
│   ├── models/            # ContrastiveMLP encoder
│   ├── training/          # CLAN loss, augmentations, schedules, checkpoints
│   ├── evaluation/        # NIDS metrics + CLAN centroid/AUROC helpers
│   └── utils/             # logging, I/O, reproducibility
├── scripts/
│   ├── train.py           # CLAN pretraining
│   ├── eval.py            # Centroid-based AUROC evaluation
│   ├── finetune.py        # Single few-shot fine-tune run
│   ├── finetune_sweep.py  # Paper-style few-shot sweep
│   └── run_experiment.sh  # Lycos/CICIDS driver
├── tests/                 # Synthetic-fixture unit/contract tests
└── docs/thesis/           # FYP thesis chapters and template
```

## Quick Start

```bash
# 1. Install
pip install -e .

# 2. Place datasets
# Lycos2017 CSV:
#   data/raw/lycos.csv
# CICIDS2017 raw CSV/zip directory:
#   data/raw/lycos-ids2017/cicids2017/csv_files

# 3. Local tests
pytest -q

# 4. Train and evaluate CLAN on Lycos2017
python scripts/train.py --config configs/lycos.yaml --seed 42
python scripts/eval.py --config configs/lycos.yaml --seed 42

# 5. Run few-shot sweep
python scripts/finetune_sweep.py --config configs/lycos.yaml --pretrain-seed 42

# 6. Full dual-dataset audit
bash scripts/run_experiment.sh both
```

## Configuration

Everything is YAML-driven. The main sections are:

| Section | Purpose |
|---|---|
| `data` | Dataset selector/path, metadata drops, split seed, batch size |
| `model` | ContrastiveMLP hidden widths, embedding dim, residual flag |
| `loss` | CLAN loss `margin` and `loss_alpha` |
| `augmentation` | Uniform-resample strength (`max_val`, `p_feature`) |
| `training` | Pretraining epochs, learning rates, scheduler, AMP |
| `finetune` | Few-shot schedule, fine-tune epochs/lr, sample-seed count |
| `runtime` | Output directory, seed, device |

Run artifacts are written to:

```text
artifacts/<dataset>/clan/seed<S>/
```

Raw data and artifacts are gitignored.

## Evaluation Metrics

`nids.evaluation.metrics.compute_nids_metrics` computes:

- Binary score metrics: `pr_auc`, `roc_auc`, `ece`, best-F1 threshold, and
  recall under FAR constraints.
- Multiclass metrics: accuracy, macro-F1, weighted-F1, MCC, per-class precision,
  recall, F1, and confusion matrix.

CLAN-specific helpers in `nids.evaluation.clan_metrics` compute centroid-based
scores, mean AUROC, per-class AUROC, and supervised fine-tune metrics.

## Requirements

- Python 3.10+
- PyTorch 2.5+
- CUDA-capable GPU recommended; the configs are tuned for an RTX 3060 Laptop GPU
  with 6 GB VRAM using AMP and `batch_size=2048`.

Install via:

```bash
pip install -r requirements.txt
pip install -e .
```

## Testing

```bash
pytest -q
```

Tests use synthetic fixtures and do not require the real Lycos2017 or CICIDS2017
datasets. Torch-dependent tests skip cleanly when PyTorch is not installed.

## Citations

- J. Wilkie, H. Hindy, C. Tachtatzis, R. Atkinson. *CLAN: Contrastive
  Self-Supervised NIDS Using Augmented Negative Pairs.* IEEE CSR 2025.
  DOI:10.1109/CSR64739.2025.11129979.
- L. Lanvin, P.-F. Gimenez, Y. Han, F. Majorczyk, L. Me, E. Totel. *Errors in
  the CICIDS2017 Dataset and the Significant Differences in Detection
  Performances It Makes.* Springer 2023.
- G. Engelen, V. Rimmer, W. Joosen. *Troubleshooting an Intrusion Detection
  Dataset: the CICIDS2017 Case Study.* WTMC 2021.
