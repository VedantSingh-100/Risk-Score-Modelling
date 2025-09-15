#!/bin/bash
# Simple GBDT Sweep Script for your revised train_gbdt_sweep.py
# This runs the random sampling hyperparameter sweep with W&B logging

set -euo pipefail

echo "🚀 Simple GBDT Hyperparameter Sweep"
echo "=================================="

# Configuration
ALGO="${1:-xgb}"        # xgb or lgb
TRIALS="${2:-60}"       # number of random trials
N_SPLITS="${3:-5}"      # CV folds
SEED="${4:-42}"         # random seed

# W&B Configuration
export WANDB_API_KEY="3ff6a13421fb5921502235dde3f9a4700f33b5b8"
export WANDB_MODE="online"
WANDB_PROJECT="Risk Score"

# Paths
DATA_ROOT="data/processed"
OUTPUT_DIR="model_outputs/gbdt_sweep_${ALGO}"

echo "Algorithm: ${ALGO^^}"
echo "Trials: $TRIALS"
echo "CV Folds: $N_SPLITS"
echo "Seed: $SEED"
echo "W&B Project: $WANDB_PROJECT"
echo "Output: $OUTPUT_DIR"
echo "Start: $(date)"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check data files
echo "🔍 Checking data files..."
if [[ -f "$DATA_ROOT/X_features.parquet" ]]; then
    echo "✓ Found: $DATA_ROOT/X_features.parquet"
else
    echo "❌ Missing: $DATA_ROOT/X_features.parquet"
    exit 1
fi

if [[ -f "$DATA_ROOT/y_label.csv" ]]; then
    echo "✓ Found: $DATA_ROOT/y_label.csv"
else
    echo "❌ Missing: $DATA_ROOT/y_label.csv"
    exit 1
fi

echo ""
echo "🎯 Starting ${ALGO^^} hyperparameter sweep..."

# Run the sweep
python -m src.train_gbdt_sweep \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUTPUT_DIR" \
    --algo "$ALGO" \
    --trials "$TRIALS" \
    --n-splits "$N_SPLITS" \
    --seed "$SEED" \
    --wandb \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-entity "ved100-carnegie-mellon-university"

exit_code=$?

echo ""
echo "=========================================="
if [[ $exit_code -eq 0 ]]; then
    echo "✅ GBDT sweep completed successfully!"
    
    # Display results
    if [[ -f "$OUTPUT_DIR/best_params.json" ]]; then
        echo ""
        echo "🏆 Best Results:"
        python -c "
import json
with open('$OUTPUT_DIR/best_params.json', 'r') as f:
    results = json.load(f)
print(f'Best Trial: #{results[\"trial\"]}')
print(f'Best AUC: {results[\"auc\"]:.6f}')
print(f'Best AP: {results[\"ap\"]:.6f}')
"
    fi
    
    echo ""
    echo "📁 Generated Files:"
    echo "  ✓ $OUTPUT_DIR/best_params.json - Best hyperparameters"
    echo "  ✓ $OUTPUT_DIR/trials_summary.csv - All trials ranked by AUC"
    echo "  ✓ $OUTPUT_DIR/oof_running_best.csv - Out-of-fold predictions"
    echo "  ✓ $OUTPUT_DIR/feature_importance_running_best.csv - Feature importance"
    echo "  ✓ $OUTPUT_DIR/deciles_running_best.csv - Decile analysis"
    
    echo ""
    echo "🔗 Weights & Biases:"
    echo "  Project: $WANDB_PROJECT"
    echo "  URL: https://wandb.ai/[your-username]/Risk%20Score"
    
else
    echo "❌ GBDT sweep failed with exit code: $exit_code"
fi

echo "End: $(date)"
echo "=========================================="

exit $exit_code
