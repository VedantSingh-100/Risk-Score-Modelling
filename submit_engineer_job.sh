#!/bin/bash

#SBATCH --job-name=engineer_features
#SBATCH --partition=cpu                    # Use CPU partition
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks-per-node=1              # One task per node
#SBATCH --cpus-per-task=16               # 16 CPU cores (more for feature engineering)
#SBATCH --mem=64G                        # 64GB memory (more for feature engineering)
#SBATCH --time=03:00:00                  # 3 hour time limit
#SBATCH --output=logs/engineer_%j.out    # Output file (%j = job ID)
#SBATCH --error=logs/engineer_%j.err     # Error file (%j = job ID)

# Load necessary modules (adjust based on your HPC environment)
# module load python/3.9
# module load pandas
# module load numpy
# module load pyarrow  # for parquet support

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
echo "Checking required files..."
required_files=(
    "50k_users_merged_data_userfile_updated_shopping.csv"
    "deterministic_build/selected_features_final.csv"
    "deterministic_build/selected_label_sources.csv"
    "deterministic_build/guard_set.txt"
)

# Optional files (will use if present)
optional_files=(
    "deterministic_build/build_summary.json"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    else
        echo "✓ Found: $file ($(du -h "$file" | cut -f1))"
    fi
done

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
    echo "Note: You may need to run build_labels_and_features.py first to generate the required files."
    exit 1
fi

echo ""
echo "All required files found. Starting feature engineering..."
echo "=================================================="
echo ""

# Set Python to unbuffered output for real-time progress
export PYTHONUNBUFFERED=1

# Install required packages if not available (uncomment if needed)
# pip install --user tqdm pandas numpy pyarrow

# Set up output directory
mkdir -p engineered

# Run the Python script with appropriate arguments
echo "Running feature engineering pipeline..."
python engineer_features.py \
    --data "50k_users_merged_data_userfile_updated_shopping.csv" \
    --selected-features "deterministic_build/selected_features_final.csv" \
    --label-sources "deterministic_build/selected_label_sources.csv" \
    --guard "deterministic_build/guard_set.txt" \
    --build-summary "deterministic_build/build_summary.json" \
    --outdir "engineered" \
    --fill-threshold 0.85

# Capture exit code
exit_code=$?

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"

if [[ $exit_code -eq 0 ]]; then
    echo "Feature engineering completed successfully!"
    echo ""
    echo "Output files:"
    if [[ -d "engineered" ]]; then
        echo "Directory: engineered/"
        ls -la engineered/
        echo ""
        echo "File sizes:"
        du -h engineered/*
    fi
else
    echo "Feature engineering failed with exit code: $exit_code"
    echo "Check the error log for details."
fi

echo ""
echo "Job resource usage:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

# Optional: Compress large output files for storage efficiency
if [[ $exit_code -eq 0 && -f "engineered/X_features.parquet" ]]; then
    echo ""
    echo "Compressing large output files..."
    # Note: Parquet is already compressed, but you could add other compression here
    echo "Feature matrix size: $(du -h engineered/X_features.parquet)"
fi

echo ""
echo "=== Feature Engineering Job Complete ==="

