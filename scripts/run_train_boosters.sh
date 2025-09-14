#!/bin/bash
# Run booster model training (XGBoost/LightGBM)

set -euo pipefail

echo "=== Running Booster Model Training ==="
echo "Start time: $(date)"

# Ensure output directory exists
mkdir -p artifacts/reports/boosters

# Run booster training with redundancy pruning
python -m src.models.train_boosters \
    --data-dir data/processed \
    --out-dir artifacts/reports/boosters \
    --redundancy-r 0.97 \
    --folds 5 \
    --seed 42

echo "=== Booster Training Complete ==="
echo "End time: $(date)"
echo "Results saved to: artifacts/reports/boosters/"
