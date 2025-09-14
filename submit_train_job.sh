#!/bin/bash

#SBATCH --job-name=model_training
#SBATCH --partition=cpu                    # Use CPU partition
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks-per-node=1              # One task per node
#SBATCH --cpus-per-task=20               # 20 CPU cores (good for ML training)
#SBATCH --mem=64G                        # 64GB memory (sufficient for model training)
#SBATCH --time=04:00:00                  # 4 hour time limit
#SBATCH --output=logs/train_%j.out       # Output file (%j = job ID)
#SBATCH --error=logs/train_%j.err        # Error file (%j = job ID)

# Load necessary modules (adjust based on your HPC environment)
# module load python/3.9
# module load scikit-learn
# module load pandas
# module load numpy

# Ensure logs directory exists
mkdir -p logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo ""

# Print environment information
echo "Python version:"
python --version
echo ""

echo "Available memory:"
free -h
echo ""

echo "CPU information:"
lscpu | grep "Model name\|CPU(s)\|Thread(s)"
echo ""

echo "Disk space:"
df -h .
echo ""

# Check if required files exist
echo "Checking required files for model training..."

# Primary data files required
required_files=(
    "data/X_features.parquet"
    "data/y_label.csv"
)

# QC files required for redundancy analysis
qc_files=(
    "data/qc_redundancy_pairs.csv"
    "data/qc_single_feature_metrics.csv"
)

# Optional but important files
optional_files=(
    "data/dropped_features.json"
    "data/guard_Set.txt"
    "data/guard_set.txt"
    "data/do_not_use_features.txt"
    "data/Selected_label_sources.csv"
)

# Check primary required files
missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    else
        echo "✓ Found: $file ($(du -h "$file" | cut -f1))"
    fi
done

# Check QC files (at least one variation must exist)
qc_found=false
for file in "${qc_files[@]}"; do
    # Check multiple naming variations
    if [[ -f "$file" ]]; then
        echo "✓ Found QC file: $file"
        qc_found=true
    elif [[ -f "${file/qc_single_feature_metrics/qc_single_feature_matrix}" ]]; then
        echo "✓ Found QC file: ${file/qc_single_feature_metrics/qc_single_feature_matrix}"
        qc_found=true
    elif [[ -f "${file/qc_redundancy_pairs/qc_redundancyt_pairs}" ]]; then
        echo "✓ Found QC file: ${file/qc_redundancy_pairs/qc_redundancyt_pairs}"
        qc_found=true
    fi
done

if [[ "$qc_found" == false ]]; then
    missing_files+=("QC files (qc_redundancy_pairs.csv and qc_single_feature_metrics.csv)")
fi

echo ""
echo "Optional files:"
for file in "${optional_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ Found: $file"
    else
        echo "○ Missing (optional): $file"
    fi
done

if [[ ${#missing_files[@]} -ne 0 ]]; then
    echo ""
    echo "ERROR: Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "  ✗ $file"
    done
    echo ""
    echo "Please ensure all required files are present before running the job."
    echo ""
    echo "Expected workflow:"
    echo "1. Run deterministic feature engineering to generate X_features.parquet and y_label.csv"
    echo "2. Run QC suite to generate redundancy analysis files"
    echo "3. Copy/symlink files to data/ directory"
    echo "4. Run this model training script"
    exit 1
fi

echo ""
echo "All required files found. Starting model training..."
echo "=================================================="
echo ""

# Set Python to unbuffered output for real-time progress
export PYTHONUNBUFFERED=1

# Set number of parallel jobs for scikit-learn
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Install required packages if not available (uncomment if needed)
# pip install --user tqdm pandas numpy scikit-learn pyarrow

# Create data/outputs directory
mkdir -p data/outputs

# Print training configuration
echo "Training configuration:"
echo "- Parallel threads: $SLURM_CPUS_PER_TASK"
echo "- Models: Logistic Regression (ElasticNet) + Gradient Boosting"
echo "- Cross-validation: 5-fold × 5 seeds = 25 total fits per model"
echo "- Features: Redundancy pruning + guard filtering"
echo "- Stability analysis: Top-20 feature Jaccard similarity"
echo ""

# Run the Python script
python train.py

# Capture exit code
exit_code=$?

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"

if [[ $exit_code -eq 0 ]]; then
    echo "Model training completed successfully!"
    echo ""
    echo "Output files generated:"
    
    # Check output files
    output_files=(
        "data/outputs/X_features_model.parquet"
        "data/outputs/model_eval_summary.csv"
        "data/outputs/pcs_stability_summary.json"
        "data/outputs/feature_importance_lr.csv"
        "data/outputs/feature_importance_gb.csv"
        "data/outputs/kept_features.csv"
        "data/outputs/redundancy_drops.csv"
    )
    
    for file in "${output_files[@]}"; do
        if [[ -f "$file" ]]; then
            echo "✓ $file ($(du -h "$file" | cut -f1))"
        else
            echo "✗ Missing: $file"
        fi
    done
    
    echo ""
    echo "Model performance summary:"
    if [[ -f "data/outputs/pcs_stability_summary.json" ]]; then
        python -c "
import json
try:
    with open('data/outputs/pcs_stability_summary.json') as f:
        summary = json.load(f)
    
    print(f'Features: {summary.get(\"n_features_after_prune\", \"N/A\")} (after pruning from {summary.get(\"n_features_before_prune\", \"N/A\")})')
    print(f'Label: {summary.get(\"label\", \"N/A\")}')
    
    if 'logit_en' in summary:
        lr = summary['logit_en']
        print(f'Logistic Regression: AUC = {lr.get(\"mean_auc\", 0):.4f} ± {lr.get(\"std_auc\", 0):.4f}')
        print(f'                     AP  = {lr.get(\"mean_ap\", 0):.4f} ± {lr.get(\"std_ap\", 0):.4f}')
        print(f'                     Top-20 Stability = {lr.get(\"mean_jaccard_top20\", 0):.4f}')
    
    if 'gbdt' in summary:
        gb = summary['gbdt']
        print(f'Gradient Boosting:   AUC = {gb.get(\"mean_auc\", 0):.4f} ± {gb.get(\"std_auc\", 0):.4f}')
        print(f'                     AP  = {gb.get(\"mean_ap\", 0):.4f} ± {gb.get(\"std_ap\", 0):.4f}')
        print(f'                     Top-20 Stability = {gb.get(\"mean_jaccard_top20\", 0):.4f}')
        
except Exception as e:
    print(f'Could not read summary: {e}')
"
    fi
    
    echo ""
    echo "Feature matrix details:"
    if [[ -f "data/outputs/X_features_model.parquet" ]]; then
        python -c "
import pandas as pd
try:
    df = pd.read_parquet('data/outputs/X_features_model.parquet')
    print(f'Final feature matrix: {df.shape[0]:,} samples × {df.shape[1]} features')
    print(f'Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB')
except Exception as e:
    print(f'Could not read feature matrix: {e}')
" 2>/dev/null || echo "Could not determine feature matrix details"
    fi
    
else
    echo "Model training failed with exit code: $exit_code"
    echo "Check the error log for details."
    echo ""
    echo "Common issues:"
    echo "- Missing input files (check file paths and QC outputs)"
    echo "- Memory issues (increase --mem if needed)"
    echo "- Data format problems (check parquet/CSV structure)"
    echo "- Feature engineering pipeline failures"
fi

echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

echo ""
echo "=== Model Training Job Complete ==="
