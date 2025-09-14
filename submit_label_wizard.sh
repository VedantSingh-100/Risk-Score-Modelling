#!/bin/bash
#SBATCH --job-name=label_policy_wizard
#SBATCH --output=logs/label_wizard_%j.out
#SBATCH --error=logs/label_wizard_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu

# Label Policy Wizard Submission Script
# Author: Generated for label policy analysis
# Date: $(date +"%Y-%m-%d")

echo "=========================================="
echo "Label Policy Wizard Job"
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
pip install pandas numpy scikit-learn tqdm

# Print Python environment info
echo ""
echo "Python Environment:"
echo "- Python version: $(python --version)"
echo "- Python path: $(which python)"
echo "- Key packages:"
pip list | grep -E "(pandas|numpy|scikit-learn|tqdm)" || echo "Some packages may need to be installed"
echo ""

# Check if required files exist
echo "Checking required input files..."
required_files=(
    "variable_catalog.csv"
    "50k_users_merged_data_userfile_updated_shopping.csv"
    "label_policy_wizard.py"
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
    echo "Note: Run the main risk_feature_selection.py first to generate variable_catalog.csv"
    exit 1
fi

echo ""

# Set up timing
echo "Starting Label Policy Wizard..."
start_time=$(date +%s)

# Create timestamped log file
log_file="logs/label_wizard_output_$(date +%Y%m%d_%H%M%S).log"

# Run the label policy wizard with error handling
echo "Executing label_policy_wizard.py..."
echo "Output will be saved to: $log_file"
python label_policy_wizard.py 2>&1 | tee "$log_file"

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
    "outcome_candidates.csv"
    "outcome_agreement_jaccard.csv"
    "labels_preview.csv"
    "label_variant_eval.csv"
    "label_policy.json"
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

# Summary of results
if [ $exit_code -eq 0 ]; then
    echo ""
    echo "🎉 Label Policy Wizard completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Review outcome_candidates.csv - check which outcome variables were found"
    echo "2. Examine outcome_agreement_jaccard.csv - see how well different outcomes agree"
    echo "3. Compare label_variant_eval.csv - which label definition performs best?"
    echo "4. Check labels_preview.csv - sample of how different policies label the same data"
    echo "5. Review label_policy.json - complete documentation of the analysis"
    echo ""
    echo "💡 Recommendations:"
    echo "- Choose the label variant with highest AUC from label_variant_eval.csv"
    echo "- Consider label variants with good business interpretation"
    echo "- Update your main analysis pipeline with the chosen label definition"
else
    echo ""
    echo "❌ Label Policy Wizard failed with exit code $exit_code"
    echo "Check the error log for details: $log_file"
fi

echo ""
echo "Detailed log saved to: $log_file"
echo "=========================================="

# Exit with the same code as the main script
exit $exit_code

