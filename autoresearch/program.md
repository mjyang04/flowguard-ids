# autoresearch — FlowGuard IDS

Autonomous hyperparameter search for CNN-BiLSTM-SE intrusion detection model.

## Setup

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr6`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**:
   - `prepare.py` — fixed constants, data loading, evaluation metric. **Do not modify.**
   - `train.py` — the file you modify. Model architecture, optimizer, hyperparameters.
4. **Verify data exists**: Check that `/workspace/data/processed/cross_cicids2017_to_unsw_nb15.npz` exists.
   If not, run from inside the container:
   ```
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
- Modify `train.py` only. Everything is fair game: model architecture (conv channels, LSTM size, SE, attention), optimizer, learning rate, scheduler, batch size, class weighting, dropout, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed data loading and evaluation harness.
- Change `EPOCH_BUDGET`, `INPUT_DIM`, `NUM_CLASSES`, or `DATA_FILE` in `prepare.py`.

**The goal: maximize `val_avg_attack_recall`** (higher = better).
This is the primary metric: average recall across all attack classes on the validation set.
Since EPOCH_BUDGET is fixed at 50 epochs, you don't need to worry about training time.

**VRAM** is a soft constraint. Keep `peak_vram_mb` reasonable; don't blow it up for marginal gains.

**Simplicity criterion**: All else being equal, simpler is better. A tiny improvement that adds ugly complexity is not worth it. Removing complexity with equal or better results is a win.

**First run**: Always run the baseline as-is first.

## Output format

The script prints a summary like:

```
---
val_avg_attack_recall:    0.982345
test_avg_attack_recall:   0.971234
test_false_alarm_rate:    0.001234
test_attack_precision:    0.984567
test_accuracy:            0.993456
training_seconds:         245.1
total_seconds:            248.3
peak_vram_mb:             1234.5
best_epoch:               38
num_epochs:               50
num_params_M:             2.345
```

Extract the key metric:
```bash
grep "^val_avg_attack_recall:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated, NOT comma-separated):

```
commit	val_avg_attack_recall	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. val_avg_attack_recall (e.g. 0.982345) — use 0.000000 for crashes
3. peak memory in GB (peak_vram_mb / 1024, rounded to .1f) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:
```
commit	val_avg_attack_recall	memory_gb	status	description
a1b2c3d	0.000000	0.0	keep	baseline
b2c3d4e	0.984200	1.3	keep	increase LSTM_HIDDEN to 256
c3d4e5f	0.976000	1.2	discard	remove SE blocks
d4e5f6g	0.000000	0.0	crash	CONV_CHANNELS=[256,512,512] OOM
```

## Experiment loop

LOOP FOREVER:

1. Check current branch/commit.
2. Edit `train.py` with one experimental idea.
3. `git commit`
4. Run: `python train.py > run.log 2>&1`
5. Read results: `grep "^val_avg_attack_recall:\|^peak_vram_mb:" run.log`
6. If empty → crash. Run `tail -50 run.log` for traceback. Fix if trivial, else discard.
7. Record in `results.tsv`.
8. If `val_avg_attack_recall` **improved** (higher) → keep commit, advance branch.
9. If equal or worse → `git reset --hard HEAD~1` to revert.

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
