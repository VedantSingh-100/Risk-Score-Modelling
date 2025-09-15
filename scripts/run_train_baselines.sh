#!/bin/bash
# Run baseline model training (Logistic + GBDT)

set -euo pipefail

echo "=== Running Baseline Model Training ==="
echo "Start time: $(date)"

# Ensure output directory exists
mkdir -p artifacts/reports/baselines

# Run baseline training
python -m src.models.train_baselines \
    --data-dir data/processed \
    --config-dir configs \
    --out-dir artifacts/reports/baselines \
    --folds 5

echo "=== Baseline Training Complete ==="
echo "End time: $(date)"
echo "Results saved to: artifacts/reports/baselines/"

