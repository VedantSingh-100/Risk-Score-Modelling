#!/bin/bash
#SBATCH --job-name=risk_overnight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/vhsingh/Parshvi_project/artifacts/logs/overnight_hpc_%j.out
#SBATCH --error=/home/vhsingh/Parshvi_project/artifacts/logs/overnight_hpc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu

set -euo pipefail

echo "=== Risk Model Overnight Training Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "Max runtime: 16 hours"

# Create logs directory
mkdir -p /home/vhsingh/Parshvi_project/artifacts/logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Research

echo "✓ Using Python: $(which python)"
echo "✓ Python version: $(python --version)"

# Set threading for performance
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Change to project directory
cd /home/vhsingh/Parshvi_project

# Configuration from command line arguments or defaults
TRIALS="${1:-120}"
USE_WANDB="${2:-1}"
USE_MONOTONE="${3:-1}"
WANDB_PROJECT="${4:-Risk_Score}"

echo "=== Configuration ==="
echo "Trials: $TRIALS"
echo "W&B Enabled: $USE_WANDB"
echo "Monotone Constraints: $USE_MONOTONE"
echo "W&B Project: $WANDB_PROJECT"
echo "Expected runtime: 12-16 hours"

# Install/update wandb if needed
echo "=== Checking dependencies ==="
pip install --user -q wandb

echo ""
echo "=== Starting Overnight Risk Model Training ==="
echo "This will run:"
echo "  1. GBDT sweep (XGBoost optimization)"
echo "  2. Baseline MLP (TabMLP with PyTorch)"
echo "  3. MLP stacking (sklearn MLP + ensemble)"
echo "Progress will be logged to W&B dashboard"
echo ""

# Run the overnight script with environment variables
TRIALS=$TRIALS USE_WANDB=$USE_WANDB USE_MONOTONE=$USE_MONOTONE WANDB_PROJECT=$WANDB_PROJECT scripts/run_overnight.sh

overnight_exit_code=$?

echo ""
echo "=== Job Completion Summary ==="
echo "Exit Code: $overnight_exit_code"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

if [[ $overnight_exit_code -eq 0 ]]; then
    echo "✅ Overnight training completed successfully!"
    echo ""
    echo "📊 Results Summary:"
    
    # Find the most recent output directories
    LATEST_GBDT=$(find model_outputs -name "gbdt_sweep_*" -type d | sort | tail -1)
    LATEST_TABMLP=$(find model_outputs -name "tabmlp_baseline_*" -type d | sort | tail -1)
    LATEST_STACK=$(find model_outputs -name "stack_*" -type d | sort | tail -1)
    
    if [[ -n "$LATEST_GBDT" && -f "$LATEST_GBDT/best_params.json" ]]; then
        echo "GBDT Optimization Results:"
        python -c "
import json
try:
    with open('$LATEST_GBDT/best_params.json', 'r') as f:
        results = json.load(f)
    print(f'  Best Trial: #{results[\"trial\"]}')
    print(f'  Best AUC: {results[\"auc\"]:.6f}')
    print(f'  Best AP: {results[\"ap\"]:.6f}')
    print(f'  Output: $LATEST_GBDT')
except Exception as e:
    print(f'  Could not parse GBDT results: {e}')
"
    fi
    
    if [[ -n "$LATEST_TABMLP" && -f "$LATEST_TABMLP/summary.json" ]]; then
        echo ""
        echo "Baseline MLP (TabMLP) Results:"
        python -c "
import json
try:
    with open('$LATEST_TABMLP/summary.json', 'r') as f:
        results = json.load(f)
    print(f'  AUC: {results[\"auc\"]:.6f}')
    print(f'  AP: {results[\"ap\"]:.6f}')
    print(f'  Output: $LATEST_TABMLP')
except Exception as e:
    print(f'  Could not parse TabMLP results: {e}')
"
    fi
    
    if [[ -n "$LATEST_STACK" && -f "$LATEST_STACK/summary.csv" ]]; then
        echo ""
        echo "Stacking Results:"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('$LATEST_STACK/summary.csv')
    for _, row in df.iterrows():
        print(f'  {row[\"model\"].upper()}: AUC={row[\"auc\"]:.6f}, AP={row[\"ap\"]:.6f}')
    print(f'  Output: $LATEST_STACK')
except Exception as e:
    print(f'  Could not parse stacking results: {e}')
"
    fi
    
    echo ""
    echo "📁 Key Output Files:"
    if [[ -n "$LATEST_GBDT" ]]; then
        echo "  GBDT Results: $LATEST_GBDT/"
        echo "    ✓ best_params.json - Optimal hyperparameters"
        echo "    ✓ trials_summary.csv - All trials ranked by performance"
        echo "    ✓ oof_running_best.csv - Best out-of-fold predictions"
        echo "    ✓ feature_importance_running_best.csv - Feature importance"
    fi
    
    if [[ -n "$LATEST_TABMLP" ]]; then
        echo "  Baseline MLP Results: $LATEST_TABMLP/"
        echo "    ✓ summary.json - Overall performance metrics"
        echo "    ✓ oof_deep.csv - Out-of-fold predictions"
        echo "    ✓ tabmlp_state.pt - PyTorch model weights"
        echo "    ✓ folds_summary.csv - Per-fold performance"
        echo "    ✓ deciles_deep.csv - Performance by decile"
    fi
    
    if [[ -n "$LATEST_STACK" ]]; then
        echo "  Stacking Results: $LATEST_STACK/"
        echo "    ✓ summary.csv - Model comparison (XGB vs MLP vs Stack)"
        echo "    ✓ oof_predictions.csv - All model predictions"
        echo "    ✓ deciles_*.csv - Performance by decile"
    fi
    
    echo ""
    echo "🔗 Weights & Biases:"
    echo "  Project: $WANDB_PROJECT"
    echo "  URL: https://wandb.ai/[your-username]/$WANDB_PROJECT"
    
    echo ""
    echo "🎉 Ready for Production:"
    echo "  1. Review W&B dashboard for training dynamics and hyperparameters"
    echo "  2. Compare all three models: GBDT vs TabMLP vs Stacked ensemble"
    echo "  3. Use best performing model for final deployment"
    echo "  4. TabMLP offers deep learning baseline with end-to-end training"
    
else
    echo "❌ Overnight training failed with exit code: $overnight_exit_code"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "  1. Check the error log: artifacts/logs/overnight_hpc_${SLURM_JOB_ID}.err"
    echo "  2. Check the detailed overnight log in artifacts/logs/"
    echo "  3. Verify data files exist in data/processed/"
    echo "  4. Check Python environment and package versions"
    echo "  5. Review W&B dashboard for partial results"
fi

echo ""
echo "📋 Detailed logs:"
echo "  SLURM output: artifacts/logs/overnight_hpc_${SLURM_JOB_ID}.out"
echo "  SLURM error: artifacts/logs/overnight_hpc_${SLURM_JOB_ID}.err"
echo "  Overnight log: artifacts/logs/overnight_*.log"
echo ""
echo "=== HPC Job Complete ==="

exit $overnight_exit_code

