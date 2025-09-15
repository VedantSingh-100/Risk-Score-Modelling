#!/bin/bash
# Risk Model Hyperparameter Sweep & MLP Stacking Pipeline
# Runs GBDT optimization followed by MLP stacking with W&B logging

set -euo pipefail

# =============================================================================
# Configuration
# =============================================================================

# Weights & Biases Configuration
export WANDB_API_KEY="3ff6a13421fb5921502235dde3f9a4700f33b5b8"
export WANDB_MODE="online"
WANDB_PROJECT="Risk Score"

# Pipeline Configuration
ALGO="${1:-xgb}"  # xgb or lgb
TRIALS="${2:-60}"
N_SPLITS="${3:-5}"
SEED="${4:-42}"

# Paths
DATA_ROOT="data/processed"
OUTPUT_ROOT="model_outputs"
GBDT_OUTPUT="${OUTPUT_ROOT}/gbdt_sweep_${ALGO}"
STACK_OUTPUT="${OUTPUT_ROOT}/stack_${ALGO}"

# =============================================================================
# Setup and Validation
# =============================================================================

echo "=========================================="
echo "🚀 Risk Model Hyperparameter Sweep Pipeline"
echo "=========================================="
echo "Algorithm: ${ALGO^^}"
echo "Trials: $TRIALS"
echo "CV Folds: $N_SPLITS"
echo "Random Seed: $SEED"
echo "W&B Project: $WANDB_PROJECT"
echo "Start Time: $(date)"
echo "=========================================="

# Validate inputs
if [[ "$ALGO" != "xgb" && "$ALGO" != "lgb" ]]; then
    echo "❌ Error: Algorithm must be 'xgb' or 'lgb'"
    exit 1
fi

# Check required files
echo "🔍 Validating input files..."
required_files=(
    "$DATA_ROOT/X_features.parquet"
    "$DATA_ROOT/y_label.csv"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ Found: $file"
    else
        echo "❌ Missing: $file"
        exit 1
    fi
done

# Create output directories
mkdir -p "$GBDT_OUTPUT" "$STACK_OUTPUT"

# Check Python environment
echo "🐍 Python Environment:"
echo "- Python: $(python --version)"
echo "- Working Directory: $(pwd)"

# Check required packages
echo "📦 Checking required packages..."
python -c "
import sys
required = ['pandas', 'numpy', 'scikit-learn', 'tqdm']
if '$ALGO' == 'xgb':
    required.append('xgboost')
else:
    required.append('lightgbm')

try:
    import wandb
    required.append('wandb')
    print('✓ W&B available')
except ImportError:
    print('⚠️ W&B not available - logging disabled')

missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'❌ {pkg}')

if missing:
    print(f'Missing packages: {missing}')
    sys.exit(1)
else:
    print('✅ All required packages available')
"

# =============================================================================
# Part A: GBDT Hyperparameter Sweep
# =============================================================================

echo ""
echo "=========================================="
echo "📊 Part A: GBDT Hyperparameter Sweep"
echo "=========================================="
echo "Algorithm: ${ALGO^^}"
echo "Trials: $TRIALS"
echo "Output: $GBDT_OUTPUT"
echo ""

# Set W&B run name for GBDT sweep
export WANDB_RUN_NAME="${ALGO^^}_Sweep_${TRIALS}trials_$(date +%Y%m%d_%H%M%S)"

echo "🎯 Starting ${ALGO^^} hyperparameter optimization..."
echo "W&B Run: $WANDB_RUN_NAME"

start_time=$(date +%s)

python -m src.train_gbdt_sweep \
    --data-root "$DATA_ROOT" \
    --out-dir "$GBDT_OUTPUT" \
    --algo "$ALGO" \
    --trials "$TRIALS" \
    --n-splits "$N_SPLITS" \
    --seed "$SEED" \
    --wandb \
    --wandb-project "$WANDB_PROJECT"

gbdt_exit_code=$?
end_time=$(date +%s)
gbdt_runtime=$((end_time - start_time))

if [[ $gbdt_exit_code -eq 0 ]]; then
    echo "✅ GBDT sweep completed successfully!"
    echo "⏱️ Runtime: ${gbdt_runtime}s"
    
    # Display best results
    if [[ -f "$GBDT_OUTPUT/best_params.json" ]]; then
        echo ""
        echo "🏆 Best Results:"
        python -c "
import json
with open('$GBDT_OUTPUT/best_params.json', 'r') as f:
    results = json.load(f)
print(f'Best Trial: #{results[\"trial\"]}')
print(f'Best AUC: {results[\"auc\"]:.6f}')
print(f'Best AP: {results[\"ap\"]:.6f}')
"
    fi
else
    echo "❌ GBDT sweep failed with exit code: $gbdt_exit_code"
    echo "Check logs for details"
    exit $gbdt_exit_code
fi

# =============================================================================
# Part B: MLP + Calibrated Stacking
# =============================================================================

echo ""
echo "=========================================="
echo "🧠 Part B: MLP + Calibrated Stacking"
echo "=========================================="
echo "Using best GBDT params from: $GBDT_OUTPUT/best_params.json"
echo "Output: $STACK_OUTPUT"
echo ""

# Set W&B run name for stacking
export WANDB_RUN_NAME="Stack_${ALGO^^}+MLP_$(date +%Y%m%d_%H%M%S)"

echo "🎯 Starting MLP + Stacking pipeline..."
echo "W&B Run: $WANDB_RUN_NAME"

start_time=$(date +%s)

# Initialize W&B for stacking run
python -c "
try:
    import wandb
    wandb.init(
        project='$WANDB_PROJECT',
        name='$WANDB_RUN_NAME',
        config={
            'stage': 'stacking',
            'base_algorithm': '$ALGO',
            'n_splits': $N_SPLITS,
            'seed': $SEED,
            'gbdt_trials': $TRIALS
        }
    )
    print('✓ W&B initialized for stacking')
    wandb.finish()
except ImportError:
    print('⚠️ W&B not available for stacking')
"

python -m src.train_mlp_stack \
    --data-root "$DATA_ROOT" \
    --out-dir "$STACK_OUTPUT" \
    --best-xgb-params "$GBDT_OUTPUT/best_params.json" \
    --n-splits "$N_SPLITS" \
    --seed "$SEED"

stack_exit_code=$?
end_time=$(date +%s)
stack_runtime=$((end_time - start_time))

if [[ $stack_exit_code -eq 0 ]]; then
    echo "✅ MLP stacking completed successfully!"
    echo "⏱️ Runtime: ${stack_runtime}s"
    
    # Display stacking results
    if [[ -f "$STACK_OUTPUT/summary.csv" ]]; then
        echo ""
        echo "🏆 Stacking Results:"
        python -c "
import pandas as pd
df = pd.read_csv('$STACK_OUTPUT/summary.csv')
print(df.to_string(index=False))
"
    fi
else
    echo "❌ MLP stacking failed with exit code: $stack_exit_code"
    echo "Check logs for details"
    exit $stack_exit_code
fi

# =============================================================================
# Final Summary and Cleanup
# =============================================================================

total_runtime=$((gbdt_runtime + stack_runtime))
total_runtime_formatted=$(printf '%02d:%02d:%02d' $((total_runtime/3600)) $((total_runtime%3600/60)) $((total_runtime%60)))

echo ""
echo "=========================================="
echo "🎉 Pipeline Completion Summary"
echo "=========================================="
echo "Total Runtime: $total_runtime_formatted (HH:MM:SS)"
echo "End Time: $(date)"
echo ""

echo "📁 Generated Artifacts:"
echo ""
echo "GBDT Sweep Results ($GBDT_OUTPUT):"
echo "  ✓ best_params.json - Optimal hyperparameters"
echo "  ✓ final_results.json - Complete results summary"
echo "  ✓ all_trials.csv - All optimization trials"
echo "  ✓ oof_predictions.csv - Out-of-fold predictions"
echo "  ✓ decile_analysis.csv - Performance by decile"
echo ""
echo "Stacking Results ($STACK_OUTPUT):"
echo "  ✓ summary.csv - Model comparison (XGB vs MLP vs Stack)"
echo "  ✓ oof_predictions.csv - All model predictions"
echo "  ✓ stacker_logit.json - Stacking coefficients"
echo "  ✓ deciles_*.csv - Decile analysis for each model"

# Check file sizes
echo ""
echo "📊 Output File Sizes:"
find "$OUTPUT_ROOT" -name "*.csv" -o -name "*.json" | while read -r file; do
    if [[ -f "$file" ]]; then
        size=$(du -h "$file" | cut -f1)
        echo "  $file ($size)"
    fi
done

echo ""
echo "🔗 Weights & Biases:"
echo "  Project: $WANDB_PROJECT"
echo "  GBDT Run: ${ALGO^^}_Sweep_${TRIALS}trials_*"
echo "  Stack Run: Stack_${ALGO^^}+MLP_*"
echo "  URL: https://wandb.ai/[your-username]/$WANDB_PROJECT"

echo ""
echo "🚀 Next Steps:"
echo "  1. Review W&B dashboard for detailed metrics and comparisons"
echo "  2. Check $GBDT_OUTPUT/best_params.json for optimal hyperparameters"
echo "  3. Examine $STACK_OUTPUT/summary.csv for model performance comparison"
echo "  4. Use the stacked model for production deployment"
echo "  5. Consider ensemble approaches using top-performing configurations"

echo ""
echo "✅ Risk Model Pipeline completed successfully!"
echo "=========================================="
