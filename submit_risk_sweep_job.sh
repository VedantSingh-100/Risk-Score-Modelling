#!/bin/bash
#SBATCH --job-name=risk_sweep
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/vhsingh/Parshvi_project/artifacts/logs/risk_sweep_%j.out
#SBATCH --error=/home/vhsingh/Parshvi_project/artifacts/logs/risk_sweep_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu

set -euo pipefail

echo "=== Risk Model Hyperparameter Sweep Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"

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

# Algorithm selection (default to XGBoost, can be overridden)
ALGO="${1:-xgb}"
TRIALS="${2:-60}"

echo "=== Configuration ==="
echo "Algorithm: ${ALGO^^}"
echo "Trials: $TRIALS"
echo "Expected runtime: 8-12 hours"

# Install additional packages if needed
echo "=== Checking dependencies ==="
pip install --user -q optuna wandb

echo "=== Starting Risk Model Sweep ==="
bash scripts/run_risk_model_sweep.sh "$ALGO" "$TRIALS"

sweep_exit_code=$?

echo ""
echo "=== Job Completion Summary ==="
echo "Exit Code: $sweep_exit_code"
echo "End Time: $(date)"
echo "Job ID: $SLURM_JOB_ID"

if [[ $sweep_exit_code -eq 0 ]]; then
    echo "✅ Risk model sweep completed successfully!"
    echo ""
    echo "📊 Results Summary:"
    
    # Display key results if available
    if [[ -f "model_outputs/gbdt_sweep_${ALGO}/final_results.json" ]]; then
        echo "GBDT Optimization Results:"
        python -c "
import json
try:
    with open('model_outputs/gbdt_sweep_${ALGO}/final_results.json', 'r') as f:
        results = json.load(f)
    print(f'  Best CV AUC: {results[\"best_score\"]:.6f}')
    print(f'  Final OOF AUC: {results.get(\"final_oof_auc\", \"N/A\")}')
    print(f'  Final OOF AP: {results.get(\"final_oof_ap\", \"N/A\")}')
except Exception as e:
    print(f'  Could not parse results: {e}')
"
    fi
    
    if [[ -f "model_outputs/stack_${ALGO}/summary.csv" ]]; then
        echo ""
        echo "Stacking Results:"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('model_outputs/stack_${ALGO}/summary.csv')
    for _, row in df.iterrows():
        print(f'  {row[\"model\"].upper()}: AUC={row[\"auc\"]:.6f}, AP={row[\"ap\"]:.6f}')
except Exception as e:
    print(f'  Could not parse stacking results: {e}')
"
    fi
    
    echo ""
    echo "📁 Output Locations:"
    echo "  GBDT Results: model_outputs/gbdt_sweep_${ALGO}/"
    echo "  Stacking Results: model_outputs/stack_${ALGO}/"
    echo "  W&B Project: Risk Score"
    
else
    echo "❌ Risk model sweep failed with exit code: $sweep_exit_code"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "  1. Check the error log: artifacts/logs/risk_sweep_${SLURM_JOB_ID}.err"
    echo "  2. Verify input data files exist in data/processed/"
    echo "  3. Check Python environment and package versions"
    echo "  4. Ensure sufficient memory and time allocation"
fi

echo "=== Resource Usage ==="
if command -v free &> /dev/null; then
    echo "Memory Usage:"
    free -h
fi

echo ""
echo "📋 Detailed log: artifacts/logs/risk_sweep_${SLURM_JOB_ID}.out"
echo "=== Job Complete ==="

exit $sweep_exit_code
