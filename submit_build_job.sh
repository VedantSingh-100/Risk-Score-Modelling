#!/bin/bash

#SBATCH --job-name=build_labels_features
#SBATCH --partition=cpu                    # Use CPU partition
#SBATCH --nodes=1                         # Use 1 node
#SBATCH --ntasks-per-node=1              # One task per node
#SBATCH --cpus-per-task=8                # 8 CPU cores
#SBATCH --mem=32G                        # 32GB memory
#SBATCH --time=02:00:00                  # 2 hour time limit
#SBATCH --output=logs/build_%j.out       # Output file (%j = job ID)
#SBATCH --error=logs/build_%j.err        # Error file (%j = job ID)

# Load necessary modules (adjust based on your HPC environment)
# module load python/3.9
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

# Check if required files exist
echo "Checking required files..."
required_files=(
    "sweep/best_config.json"
    "smart_label_candidates.csv"
    "negative_pattern_variables.csv"
    "Internal_Algo360VariableDictionary_WithExplanation.xlsx"
    "50k_users_merged_data_userfile_updated_shopping.csv"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        missing_files+=("$file")
    else
        echo "✓ Found: $file"
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
    exit 1
fi

echo ""
echo "All required files found. Starting build process..."
echo "=================================================="
echo ""

# Set Python to unbuffered output for real-time progress
export PYTHONUNBUFFERED=1

# Install tqdm if not available (uncomment if needed)
# pip install --user tqdm

# Run the Python script
python build_labels_and_features.py

# Capture exit code
exit_code=$?

echo ""
echo "=================================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"

if [[ $exit_code -eq 0 ]]; then
    echo "Job completed successfully!"
    echo ""
    echo "Output files:"
    if [[ -d "deterministic_build" ]]; then
        ls -la deterministic_build/
    fi
else
    echo "Job failed with exit code: $exit_code"
    echo "Check the error log for details."
fi

echo ""
echo "Job statistics:"
sacct -j $SLURM_JOB_ID --format=JobID,JobName,Partition,Account,AllocCPUS,State,ExitCode,Elapsed,MaxRSS,MaxVMSize

