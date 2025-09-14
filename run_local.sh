#!/bin/bash

# Local execution script for Risk Feature Selection Analysis
# Use this script to run the analysis locally without SLURM

echo "=========================================="
echo "Risk Feature Selection Analysis (Local)"
echo "Start Time: $(date)"
echo "=========================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# Print system information
echo "System Information:"
echo "- Hostname: $(hostname)"
echo "- CPUs available: $(nproc)"
echo "- Memory available: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "- Working directory: $(pwd)"
echo ""

# Check if virtual environment exists, create if not
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
echo "- Key packages:"
pip list | grep -E "(pandas|numpy|scikit-learn|tqdm|openpyxl)" || echo "Some packages may need to be installed"
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

# Set up timing
echo "Starting risk feature selection analysis..."
start_time=$(date +%s)

# Create timestamped log file
log_file="logs/analysis_output_$(date +%Y%m%d_%H%M%S).log"

# Run the main script with error handling
echo "Executing risk_feature_selection.py..."
echo "Output will be saved to: $log_file"
python risk_feature_selection.py 2>&1 | tee "$log_file"

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

echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Analysis completed successfully!"
    echo "Check the generated files and model_card.md for results."
    echo "Detailed log saved to: $log_file"
else
    echo "❌ Analysis failed with exit code $exit_code"
    echo "Check the error log for details: $log_file"
fi

echo "=========================================="

# Exit with the same code as the main script
exit $exit_code
