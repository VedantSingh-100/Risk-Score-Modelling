#!/bin/bash
#SBATCH --job-name=advanced_train
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/vhsingh/Parshvi_project/logs/advanced_train_%j.out
#SBATCH --error=/home/vhsingh/Parshvi_project/logs/advanced_train_%j.err

set -euo pipefail
echo "=== Advanced Model Training Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

mkdir -p /home/vhsingh/Parshvi_project/logs

# Activate conda environment instead of using modules
source ~/miniconda3/etc/profile.d/conda.sh
conda activate Research

echo "✓ Using Python: $(which python)"
echo "✓ Python version: $(python --version)"

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "=== Validating Input Files ==="
feature_files=(/data/X_features.parquet /home/vhsingh/Parshvi_project/data/X_features.parquet /home/vhsingh/Parshvi_project/engineered/X_features.parquet /home/vhsingh/deterministic_fe_outputs/X_features.parquet)
label_files=(/data/y_label.csv /home/vhsingh/Parshvi_project/data/y_label.csv /home/vhsingh/Parshvi_project/engineered/y_label.csv /home/vhsingh/deterministic_fe_outputs/y_label.csv)
guard_files=(/data/guard_Set.txt /data/guard_set.txt /home/vhsingh/Parshvi_project/data/guard_Set.txt /home/vhsingh/Parshvi_project/engineered/guard_set.txt)

feature_found=false; for f in "${feature_files[@]}"; do [[ -f "$f" ]] && echo "✓ Found feature file: $f" && feature_found=true && FEATURE="$f" && break; done
[[ $feature_found == false ]] && { printf "❌ No feature file found in:\n%s\n" "${feature_files[@]}"; exit 1; }

label_found=false; for f in "${label_files[@]}"; do [[ -f "$f" ]] && echo "✓ Found label file: $f" && label_found=true && LABEL="$f" && break; done
[[ $label_found == false ]] && { printf "❌ No label file found in:\n%s\n" "${label_files[@]}"; exit 1; }

guard_found=false; for f in "${guard_files[@]}"; do [[ -f "$f" ]] && echo "✓ Found guard file: $f" && guard_found=true && GUARD="$f" && break; done
[[ $guard_found == false ]] && echo "⚠️  No guard file found (optional)"

echo "✅ Input validation completed"
echo "=== Checking Python dependencies ==="
python -c "import pandas, numpy, sklearn, tqdm, xgboost; print('✓ All required packages available')" || echo "⚠️ Some packages missing"

cd /home/vhsingh/Parshvi_project

echo "=== Starting Advanced Model Training ==="
srun python train_advanced.py \
  --features "${FEATURE}" \
  --labels "${LABEL}" \
  ${GUARD:+--guards "$GUARD"}

output_dir="/home/vhsingh/Parshvi_project/model_outputs"  # <<< was '/model_outputs'; use a writable path
echo "=== Post-Execution Analysis ==="
if [[ -d "$output_dir" ]]; then
  echo "📁 Output directory: $output_dir"
  ls -la "$output_dir"
  if [[ -f "$output_dir/model_cv_report.json" ]]; then
    echo "🏆 Model Performance Summary:"
    python - <<'PY'
import json, sys, pathlib
p = pathlib.Path("/home/vhsingh/Parshvi_project/model_outputs/model_cv_report.json")
r = json.load(open(p))
print(f"Model Type: {r.get('model_kind','?').upper()}")
print(f"Dataset Size: {r.get('n_rows',0):,} samples, {r.get('n_features',0)} features")
print(f"Cross-Validation AUC: {r.get('auc_cv_mean',0):.4f} ± {r.get('auc_cv_std',0):.4f}")
print(f"Cross-Validation AP: {r.get('ap_cv_mean',0):.4f} ± {r.get('ap_cv_std',0):.4f}")
print(f"OOF AUC (Raw): {r.get('auc_oof_raw',0):.4f}")
print(f"OOF AUC (Calibrated): {r.get('auc_oof_cal',0):.4f}")
PY
  fi
  [[ -f "$output_dir/feature_importance_cv.csv" ]] && { echo "🔍 Top 10 Features:"; head -11 "$output_dir/feature_importance_cv.csv" | column -t -s,; }
  [[ -f "$output_dir/cv_fold_metrics.csv" ]] && { echo "📈 Fold-wise Performance:"; column -t -s, "$output_dir/cv_fold_metrics.csv"; }
else
  echo "❌ Output directory not found: $output_dir"
fi

echo ""
echo "=== Resource Usage Summary ==="
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs allocated: ${SLURM_CPUS_PER_TASK:-1}"
echo "Memory allocated: 32G"
echo "End time: $(date)"
echo "=== Job Completed ==="
