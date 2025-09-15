#!/bin/bash
#SBATCH --job-name=ft_transformer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/vhsingh/Parshvi_project/artifacts/logs/fttransformer_hpc_%j.out
#SBATCH --error=/home/vhsingh/Parshvi_project/artifacts/logs/fttransformer_hpc_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu

set -euo pipefail

echo "=== FT-Transformer Training Job ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"
echo "Max runtime: 12 hours"

# Create logs directory
mkdir -p /home/vhsingh/Parshvi_project/artifacts/logs

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Research

echo "✓ Using Python: $(which python)"
echo "✓ Python version: $(python --version)"
echo "✓ PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "✓ CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; torch.cuda.is_available()' 2>/dev/null; then
    echo "✓ GPU device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
fi

# Set threading for performance
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

# Change to project directory
cd /home/vhsingh/Parshvi_project

# ============================================================
# Configuration - following the same pattern as run_overnight.sh
# ============================================================
export WANDB_API_KEY="3ff6a13421fb5921502235dde3f9a4700f33b5b8"
export WANDB_MODE="online"

# Configuration knobs (can be set via env vars)
FOLDS="${FOLDS:-5}"              # CV folds
SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-1}"      # 1 to enable W&B logging  
WANDB_PROJECT="${WANDB_PROJECT:-Risk_Score_Transformer}"

# Paths - following the same structure
REPO_ROOT="$(pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/processed}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/model_outputs}"
LOG_DIR="${REPO_ROOT}/artifacts/logs"

echo "=== Configuration ==="
echo "WandB Project: $WANDB_PROJECT"
echo "Data Root: $DATA_ROOT"
echo "Output Root: $OUT_ROOT"
echo "CV Folds: $FOLDS"
echo "Seed: $SEED"
echo "Use WandB: $USE_WANDB"
echo "================================"

# Create timestamp for output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR="${OUT_ROOT}/fttr_${TIMESTAMP}"

echo "Output directory: $OUT_DIR"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Setup WandB following the same pattern as run_overnight.sh
WANDB_FLAG=""
DISABLE_WANDB=""
if [[ "${USE_WANDB}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] USE_WANDB=1 but WANDB_API_KEY is not set. Disabling W&B."
    DISABLE_WANDB="--disable-wandb"
  else
    export WANDB_API_KEY
    export WANDB_PROJECT
    export WANDB_ENTITY="ved100-carnegie-mellon-university"
    WANDB_FLAG="--wandb-project ${WANDB_PROJECT}"
    echo "✓ WandB setup complete: ${WANDB_PROJECT}"
  fi
else
  echo "WandB logging disabled"
  DISABLE_WANDB="--disable-wandb"
fi

echo "=== Starting FT-Transformer Training ==="
echo "Command: python -m src.models.train_fttransformer"

# Run FT-Transformer with optimal hyperparameters for 570 features
python -m src.models.train_fttransformer \
    --data-root "$DATA_ROOT" \
    --out-dir "$OUT_DIR" \
    --layers 4 \
    --d-model 96 \
    --heads 8 \
    --ff-mult 2.0 \
    --dropout 0.15 \
    --attn-dropout 0.10 \
    --feature-dropout 0.05 \
    --epochs 160 \
    --batch-size 256 \
    --lr 1e-3 \
    --weight-decay 1e-4 \
    --warmup-ratio 0.06 \
    --patience 20 \
    --n-splits "$FOLDS" \
    --seed "$SEED" \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-run-name "fttr_${TIMESTAMP}" \
    $DISABLE_WANDB

TRAINING_EXIT_CODE=$?

echo "=== Training Complete ==="
echo "Exit code: $TRAINING_EXIT_CODE"
echo "End time: $(date)"
echo "Output directory: $OUT_DIR"

# Display summary of generated files
if [ -d "$OUT_DIR" ]; then
    echo "=== Generated Files ==="
    ls -la "$OUT_DIR"
    
    # Display key metrics if available
    if [ -f "$OUT_DIR/summary.json" ]; then
        echo "=== Training Summary ==="
        python -c "import json; print(json.dumps(json.load(open('$OUT_DIR/summary.json')), indent=2))"
    fi
    
    if [ -f "$OUT_DIR/folds_summary.csv" ]; then
        echo "=== Fold Summary ==="
        head -10 "$OUT_DIR/folds_summary.csv"
    fi
fi

# Cleanup - remove any temporary files if needed
echo "=== Cleanup ==="
# Add any cleanup commands here if needed

echo "=== Job Complete ==="
echo "Final exit code: $TRAINING_EXIT_CODE"

exit $TRAINING_EXIT_CODE
