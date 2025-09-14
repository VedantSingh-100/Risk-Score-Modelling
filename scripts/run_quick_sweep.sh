#!/bin/bash
# Quick Risk Model Sweep (for testing)
# Runs a smaller sweep for validation

set -euo pipefail

echo "🚀 Quick Risk Model Sweep (Testing)"
echo "=================================="

# Weights & Biases Configuration
export WANDB_PROJECT="Risk Score"
export WANDB_API_KEY="3ff6a13421fb5921502235dde3f9a4700f33b5b8"
export WANDB_MODE="online"

# Quick test configuration
ALGO="${1:-xgb}"
TRIALS=10  # Reduced for quick testing
N_SPLITS=3  # Reduced for speed
SEED=42

echo "Algorithm: ${ALGO^^}"
echo "Trials: $TRIALS (quick test)"
echo "CV Folds: $N_SPLITS"
echo "Start: $(date)"

# Run the main pipeline with reduced parameters
bash scripts/run_risk_model_sweep.sh "$ALGO" "$TRIALS" "$N_SPLITS" "$SEED"

echo "✅ Quick sweep completed!"
