#!/bin/bash
#SBATCH --job-name=risk_feature_analysis
#SBATCH --output=logs/risk_analysis_%j.out
#SBATCH --error=logs/risk_analysis_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu

# Risk Feature Selection Analysis Submission Script
# Author: Generated for risk modeling pipeline
# Date: $(date +"%Y-%m-%d")

echo "=========================================="
echo "Risk Feature Selection Analysis Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Print system information
echo "System Information:"
echo "- Hostname: $(hostname)"
echo "- CPUs allocated: $SLURM_CPUS_PER_TASK"
echo "- Memory allocated: $SLURM_MEM_PER_NODE MB"
echo "- Working directory: $(pwd)"
echo ""

# Load necessary modules (adjust based on your cluster)
echo "Loading modules..."
# module load python/3.9
# module load anaconda3
# Uncomment and modify the above lines based on your cluster's module system

# Activate conda environment if needed
# echo "Activating conda environment..."
# source ~/.bashrc
# conda activate risk_analysis  # Replace with your environment name

# Alternative: Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

# Install required packages
echo "Installing/updating required packages..."
pip install --upgrade pip
pip install pandas numpy scikit-learn tqdm openpyxl xlrd

# Print Python environment info
echo ""
echo "Python Environment:"
echo "- Python version: $(python --version)"
echo "- Python path: $(which python)"
echo "- Pip packages:"
pip list | grep -E "(pandas|numpy|scikit-learn|tqdm|openpyxl)"
echo ""

# Check if required files exist
echo "Checking required input files..."
required_files=(
    "50k_users_merged_data_userfile_updated_shopping.csv"
    "Internal_Algo360VariableDictionary_WithExplanation.xlsx"
    "risk_feature_selection.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file ($(du -h "$file" | cut -f1))"
    else
        echo "✗ Missing: $file"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "Error: Missing required files. Exiting."
    exit 1
fi

# Optional file check
if [ -f "Variable_Classification_Table_v2.xlsx" ]; then
    echo "✓ Found optional: Variable_Classification_Table_v2.xlsx"
else
    echo "⚠ Optional file not found: Variable_Classification_Table_v2.xlsx"
fi

echo ""

# Set up timing and memory monitoring
echo "Starting risk feature selection analysis..."
start_time=$(date +%s)

# Run the main script with error handling
echo "Executing risk_feature_selection.py..."
python risk_feature_selection.py 2>&1 | tee logs/analysis_output_${SLURM_JOB_ID}.log

# Capture exit code
exit_code=${PIPESTATUS[0]}

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_formatted=$(printf '%02d:%02d:%02d' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)))

echo ""
echo "=========================================="
echo "Job Completion Summary"
echo "=========================================="
echo "Exit Code: $exit_code"
echo "Runtime: $runtime_formatted (HH:MM:SS)"
echo "End Time: $(date)"

# Check output files
echo ""
echo "Output Files Generated:"
output_files=(
    "variable_catalog.csv"
    "candidate_targets_ranked.csv"
    "feature_importance_consensus.csv"
    "model_card.md"
)

for file in "${output_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        lines=$(wc -l < "$file" 2>/dev/null || echo "N/A")
        echo "✓ $file ($size, $lines lines)"
    else
        echo "✗ $file (not generated)"
    fi
done

# Print memory usage if available
if command -v free &> /dev/null; then
    echo ""
    echo "Final Memory Usage:"
    free -h
fi

# Cleanup (optional)
# echo "Cleaning up temporary files..."
# rm -f *.tmp

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Analysis completed successfully!"
    echo "Check the generated files and model_card.md for results."
else
    echo "❌ Analysis failed with exit code $exit_code"
    echo "Check the error logs for details."
fi

echo "Log files saved to logs/ directory"
echo "=========================================="

# Exit with the same code as the main script
exit $exit_code
