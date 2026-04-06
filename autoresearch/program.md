# autoresearch — FlowGuard IDS

Autonomous hyperparameter search for CNN-BiLSTM-SE intrusion detection model.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `prepare.py` — shared constants, data loading, and the v2 evaluation harness.
   - `train.py` — model architecture, optimizer, hyperparameters, and selection logic.
4. **Verify data exists**: Check that both data files exist:
   - `/workspace/data/processed/cross_cicids2017_to_unsw_nb15.npz` (cross-dataset)
   - `/workspace/data/processed/cicids2017/data.npz` (single-dataset)
   If not, run preprocessing:
   ```
   python scripts/preprocess.py --dataset cicids2017 --config configs/default.yaml
   python scripts/preprocess_cross_dataset.py --config configs/default.yaml
   ```
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**.

## Experimentation

Run the training script from inside the `autoresearch/` directory:

```bash
cd /workspace/autoresearch
python train.py
```

**What you CAN do:**
- Modify `train.py` for model architecture, optimizer, learning rate, scheduler, batch size, class weighting, dropout, and selection logic.
- Modify `prepare.py` only when intentionally evolving the shared evaluation harness.

**What you CANNOT do:**
- Change `MAX_EPOCHS`, `INPUT_DIM`, `NUM_CLASSES`, or data file paths in `prepare.py`.

**The v2 goal:**
- Primary: maximize `recall_at_far_1pct`
- Secondary: maximize `pr_auc`
- Tertiary: maximize `avg_attack_recall`
- Constraint-aware tie-break: prefer lower `false_alarm_rate`

Track both transfer settings:
- `test_single_recall_at_far_1pct` / `test_single_pr_auc`
- `test_cross_recall_at_far_1pct` / `test_cross_pr_auc`

Training runs up to MAX_EPOCHS=50 with early stopping (patience=5, delta=1e-4). The best model weights are selected by the v2 validation objective: `recall_at_far_1pct`, then `pr_auc`, then `avg_attack_recall`.

**VRAM** is a soft constraint. Keep `peak_vram_mb` reasonable; don't blow it up for marginal gains.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing complexity with equal or better results is a win.

**First run**: Always run the baseline as-is first.

## Output format

The script prints a summary like:

```
---
selection_primary_metric:        recall_at_far_1pct
selection_secondary_metric:      pr_auc
val_avg_attack_recall:           0.982345
val_pr_auc:                      0.996100
val_recall_at_far_1pct:          0.980000
test_single_avg_attack_recall:   0.991234
test_single_pr_auc:              0.998120
test_single_recall_at_far_1pct:  0.989100
test_single_false_alarm_rate:    0.000456
test_cross_avg_attack_recall:    0.971234
test_cross_pr_auc:               0.987650
test_cross_recall_at_far_1pct:   0.962300
test_cross_false_alarm_rate:     0.001234
training_seconds:                245.1
total_seconds:                   248.3
peak_vram_mb:                    1234.5
best_epoch:                      38
num_epochs:                      43
num_params_M:                    2.345
```

Extract the key metrics:
```bash
grep "^test_single_recall_at_far_1pct:\|^test_cross_recall_at_far_1pct:\|^test_single_pr_auc:\|^test_cross_pr_auc:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	test_single_r1	test_cross_r1	test_single_pr_auc	test_cross_pr_auc	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. test_single_recall_at_far_1pct (e.g. 0.989100) — use 0.000000 for crashes
3. test_cross_recall_at_far_1pct (e.g. 0.962300) — use 0.000000 for crashes
4. test_single_pr_auc (e.g. 0.998120) — use 0.000000 for crashes
5. test_cross_pr_auc (e.g. 0.987650) — use 0.000000 for crashes
6. peak memory in GB (peak_vram_mb / 1024, rounded to .1f) — use 0.0 for crashes
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:
```
commit	test_single_r1	test_cross_r1	test_single_pr_auc	test_cross_pr_auc	memory_gb	status	description
a1b2c3d	0.989100	0.962300	0.998120	0.987650	1.2	keep	baseline
b2c3d4e	0.991400	0.968500	0.998500	0.989200	1.3	keep	increase LSTM_HIDDEN to 256
c3d4e5f	0.988000	0.960100	0.997400	0.986000	1.2	discard	remove SE blocks
d4e5f6g	0.000000	0.000000	0.000000	0.000000	0.0	crash	CONV_CHANNELS=[256,512,512] OOM
```

## Experiment loop

LOOP FOREVER:

1. Check current branch/commit.
2. Edit `train.py` with one experimental idea.
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^test_single_recall_at_far_1pct:\|^test_cross_recall_at_far_1pct:\|^test_single_pr_auc:\|^test_cross_pr_auc:\|^peak_vram_mb:" run.log`
6. If empty → crash. Run `tail -50 run.log` for traceback. Fix if trivial, else discard.
7. Record in `results.tsv`.
8. If both constrained recalls improve, keep the commit.
9. If constrained recalls tie, keep only when both PR-AUC values improve.
10. If constrained recall regresses on either split, or PR-AUC regresses on a tie, discard the commit.

**Timeout**: Each experiment should take under 10 minutes. If it exceeds 10 minutes, kill it and discard.

**Crashes**: Fix trivial issues (typo, shape mismatch). If the idea is fundamentally broken, skip it.

**NEVER STOP**: Once the loop has begun, do NOT pause to ask if you should continue. Run indefinitely until manually interrupted.

## Promising directions to explore

- Larger/smaller LSTM_HIDDEN (64, 256, 512)
- More/fewer conv blocks and channels
- Attention pooling vs mean pooling
- AdamW with higher weight decay
- Focal loss for imbalanced classes (implement in train.py)
- Cosine LR schedule vs plateau
- Larger batch size (512, 1024)
- Residual connections between conv blocks
- LayerNorm instead of BatchNorm
- Feature-level dropout before the CNN
