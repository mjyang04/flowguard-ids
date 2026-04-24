#!/usr/bin/env bash
# run_experiment.sh — full CLAN reproduction + CICIDS2017 control driver.
#
# Usage:
#   bash scripts/run_experiment.sh                  # both datasets, 3 seeds
#   bash scripts/run_experiment.sh lycos            # lycos only, 3 seeds
#   bash scripts/run_experiment.sh cicids           # cicids only, 3 seeds
#   SEEDS="42" SAMPLE_SEEDS=3 bash scripts/run_experiment.sh lycos
#                                                    # quick smoke: 1 pretrain
#                                                    # seed + 3 sample seeds
#
# On RTX 3060 6 GB expected wall-clock:
#   pretrain 200 epochs × bs=2048 × AMP  ≈  30–60 min per run
#   eval                                  ≈  1 min per run
#   finetune sweep 8 shots × 10 seeds    ≈  80–160 min per run
#
# Total for both datasets × 3 seeds:
#   6 × 45 min pretrain  +  6 × 1 min eval  +  6 × 120 min ft
#   ≈ 4.5 h pretrain + 12 h ft  ≈  ~17 h total
#
# Recommended: run overnight. Artifacts land in artifacts/<dataset>/clan/seed<S>/.

set -euo pipefail

DATASETS=${1:-both}
SEEDS=${SEEDS:-"42 43 44"}
SAMPLE_SEEDS=${SAMPLE_SEEDS:-10}
DEVICE=${DEVICE:-cuda}
CONFIG_DIR=${CONFIG_DIR:-configs}

run_one_dataset() {
    local name=$1
    local config="${CONFIG_DIR}/${name}.yaml"
    if [[ ! -f "$config" ]]; then
        echo "Config not found: $config" >&2
        exit 1
    fi

    for seed in $SEEDS; do
        echo "==== $name | pretrain seed=$seed ===="
        python scripts/train.py --config "$config" --device "$DEVICE" --seed "$seed"

        echo "==== $name | eval seed=$seed ===="
        python scripts/eval.py --config "$config" --device "$DEVICE" --seed "$seed"

        echo "==== $name | finetune sweep (pretrain seed=$seed × $SAMPLE_SEEDS ft seeds) ===="
        python scripts/finetune_sweep.py --config "$config" \
            --device "$DEVICE" \
            --pretrain-seed "$seed" \
            --n-sample-seeds "$SAMPLE_SEEDS"
    done
}

case "$DATASETS" in
    lycos|lycos2017)
        run_one_dataset lycos
        ;;
    cicids|cicids2017)
        run_one_dataset cicids
        ;;
    both|all|"")
        run_one_dataset lycos
        run_one_dataset cicids
        ;;
    *)
        echo "Unknown dataset selector: $DATASETS (use: lycos | cicids | both)" >&2
        exit 1
        ;;
esac

echo "All runs complete. Artifacts under artifacts/<dataset>/clan/seed<S>/."
