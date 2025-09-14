#!/bin/bash

#SBATCH --job-name=deterministic_fe
#SBATCH --partition=cpu                    # Use CPU partition
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks-per-node=1              # One task per node
#SBATCH --cpus-per-task=12               # 12 CPU cores (good for feature engineering)
#SBATCH --mem=48G                        # 48GB memory (moderate for deterministic FE)
#SBATCH --time=02:30:00                  # 2.5 hour time limit
#SBATCH --output=logs/deterministic_%j.out    # Output file (%j = job ID)
#SBATCH --error=logs/deterministic_%j.err     # Error file (%j = job ID)

# Load necessary modules (adjust based on your HPC environment)
# module load python/3.9
# module load pandas
# module load numpy
# module load scikit-learn

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
echo "Checking required files for deterministic feature engineering..."

# Primary required files
required_files=(
    "50k_users_merged_data_userfile_updated_shopping.csv"
)

# Feature selection files (one of these must exist)
feature_files=(
    "selected_features_finalcsv"
    "deterministic_build/selected_features_final.csv"
    "engineered/feature_list_final.csv"
    "feature_list_final.csv"
)

# Label files (at least one must exist OR label sources)
label_files=(
    "engineered/y_label.csv"
    "deterministic_build/label_union.csv"
    "deterministic_build/selected_label_sources.csv"
    "y_label.csv"
    "label_union.csv"
    "Selected_label_sources.csv"
)

# Optional configuration files
optional_files=(
    "build_summary.json"
    "best_config_used.json"
    "guard_Set.txt"
    "guard_set.txt"
    "do_not_use_features.txt"
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

# Check for at least one feature file
feature_found=false
for file in "${feature_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ Found feature file: $file"
        feature_found=true
        break
    fi
done

if [[ "$feature_found" == false ]]; then
    missing_files+=("feature selection file (one of: ${feature_files[*]})")
fi

# Check for at least one label file
label_found=false
for file in "${label_files[@]}"; do
    if [[ -f "$file" ]]; then
        echo "✓ Found label file: $file"
        label_found=true
        break
    fi
done

if [[ "$label_found" == false ]]; then
    missing_files+=("label file (one of: ${label_files[*]})")
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
    echo "1. Run build_labels_and_features.py to generate feature selection and labels"
    echo "2. Run this deterministic feature engineering script"
    exit 1
fi

echo ""
echo "All required files found. Starting deterministic feature engineering..."
echo "=================================================="
echo ""

# Set Python to unbuffered output for real-time progress
export PYTHONUNBUFFERED=1

# Install required packages if not available (uncomment if needed)
# pip install --user tqdm pandas numpy scikit-learn pyarrow

# Print script info
echo "Running deterministic feature engineering with intelligent type routing..."
echo "Features:"
echo "- Automatic feature type detection (scores, vintages, ratios, counts, amounts)"
echo "- Intelligent transformation routing (log1p, asinh, robust scaling)"
echo "- Comprehensive leakage detection (AUC, Spearman, Jaccard)"
echo "- Complete audit trail and reproducibility"
echo ""

# Run the Python script
python determinstic_fe_build.py

# Capture exit code
exit_code=$?

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"

if [[ $exit_code -eq 0 ]]; then
    echo "Deterministic feature engineering completed successfully!"
    echo ""
    echo "Output files generated:"
    
    # List output files with sizes
    output_files=(
        "/home/vhsingh/deterministic_fe_outputs/X_features.parquet"
        "/home/vhsingh/deterministic_fe_outputs/y_label.csv"
        "/home/vhsingh/deterministic_fe_outputs/feature_engineering_report_pre.csv"
        "/home/vhsingh/deterministic_fe_outputs/feature_engineering_report_post.csv"
        "/home/vhsingh/deterministic_fe_outputs/leakage_check.csv"
        "/home/vhsingh/deterministic_fe_outputs/transforms_config.json"
        "/home/vhsingh/deterministic_fe_outputs/dropped_features.json"
    )
    
    for file in "${output_files[@]}"; do
        if [[ -f "$file" ]]; then
            echo "✓ $file ($(du -h "$file" | cut -f1))"
        else
            echo "✗ Missing: $file"
        fi
    done
    
    echo ""
    echo "Feature matrix details:"
    if [[ -f "/home/vhsingh/deterministic_fe_outputs/X_features.parquet" ]]; then
        echo "Feature matrix: /home/vhsingh/deterministic_fe_outputs/X_features.parquet"
        echo "Size: $(du -h /home/vhsingh/deterministic_fe_outputs/X_features.parquet)"
        # Could add python one-liner to get shape info
        python -c "import pandas as pd; df=pd.read_parquet('/home/vhsingh/deterministic_fe_outputs/X_features.parquet'); print(f'Shape: {df.shape[0]:,} rows × {df.shape[1]} features')" 2>/dev/null || echo "Shape: Could not determine"
    fi
    
    echo ""
    echo "Transformation summary:"
    if [[ -f "/home/vhsingh/deterministic_fe_outputs/transforms_config.json" ]]; then
        python -c "
import json
try:
    with open('/home/vhsingh/deterministic_fe_outputs/transforms_config.json') as f:
        cfg = json.load(f)
    if 'transforms' in cfg:
        types = {}
        for v, t in cfg['transforms'].items():
            ttype = t.get('type', 'unknown')
            types[ttype] = types.get(ttype, 0) + 1
        print('Transform types applied:')
        for ttype, count in types.items():
            print(f'  {ttype}: {count} features')
except Exception as e:
    print(f'Could not read transforms: {e}')
"
    fi
    
else
    echo "Deterministic feature engineering failed with exit code: $exit_code"
    echo "Check the error log for details."
    echo ""
    echo "Common issues:"
    echo "- Missing input files (check file paths and existence)"
    echo "- Memory issues (increase --mem if needed)"
    echo "- Data format problems (check CSV structure)"
    echo "- Feature type detection issues (review error log)"
fi

echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

echo ""
echo "=== Deterministic Feature Engineering Job Complete ==="
