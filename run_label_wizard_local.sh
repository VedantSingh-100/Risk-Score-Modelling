#!/bin/bash

# Local execution script for Label Policy Wizard
# Use this script to run the label analysis locally without SLURM

echo "=========================================="
echo "Label Policy Wizard (Local Execution)"
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
    echo ""
    echo "❌ Error: Missing required files."
    echo ""
    echo "💡 To fix this:"
    if [[ " ${missing_files[@]} " =~ " variable_catalog.csv " ]]; then
        echo "1. Run the main risk feature selection pipeline first:"
        echo "   ./run_local.sh"
        echo "   (This will generate variable_catalog.csv)"
    fi
    echo "2. Ensure all required data files are in the current directory"
    echo ""
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

total_size=0
for file in "${output_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        lines=$(wc -l < "$file" 2>/dev/null || echo "N/A")
        size_bytes=$(du -b "$file" | cut -f1)
        total_size=$((total_size + size_bytes))
        echo "✓ $file ($size, $lines lines)"
    else
        echo "✗ $file (not generated)"
    fi
done

if [ $total_size -gt 0 ]; then
    total_size_mb=$((total_size / 1024 / 1024))
    echo "Total output size: ${total_size_mb} MB"
fi

# Summary of results and next steps
echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Label Policy Wizard completed successfully!"
    echo ""
    echo "📋 Analysis Summary:"
    
    # Try to extract key stats from the generated files
    if [ -f "outcome_candidates.csv" ]; then
        candidate_count=$(tail -n +2 "outcome_candidates.csv" | wc -l)
        echo "- Found $candidate_count outcome candidate variables"
    fi
    
    if [ -f "label_variant_eval.csv" ]; then
        echo "- Generated 3 label variants for comparison"
        echo ""
        echo "🏆 Label Performance Summary:"
        if command -v python3 &> /dev/null; then
            python3 -c "
import pandas as pd
try:
    df = pd.read_csv('label_variant_eval.csv')
    for _, row in df.iterrows():
        auc = f'{row[\"auc\"]:.4f}' if pd.notna(row['auc']) else 'N/A'
        pos_rate = f'{row[\"pos_rate\"]:.4f}'
        print(f'   {row[\"label_variant\"]:20s} | AUC: {auc:>6s} | Pos Rate: {pos_rate}')
except: pass
" 2>/dev/null
        fi
    fi
    
    echo ""
    echo "📂 Next Steps:"
    echo "1. 📊 Review outcome_candidates.csv - check which outcome variables were found"
    echo "2. 🤝 Examine outcome_agreement_jaccard.csv - see how well different outcomes agree"
    echo "3. 🏅 Compare label_variant_eval.csv - which label definition performs best?"
    echo "4. 👀 Check labels_preview.csv - sample of how different policies label the same data"
    echo "5. 📝 Review label_policy.json - complete documentation of the analysis"
    echo ""
    echo "💡 Recommendations:"
    echo "- Choose the label variant with highest AUC from label_variant_eval.csv"
    echo "- Consider label variants with good business interpretation"
    echo "- Update LABEL_COL in risk_feature_selection.py with your chosen label"
    echo ""
    echo "🔄 To use your chosen label in the main pipeline:"
    echo "   1. Edit risk_feature_selection.py"
    echo "   2. Set LABEL_COL = 'your_chosen_label_column'"
    echo "   3. Re-run: ./run_local.sh"
    
else
    echo "❌ Label Policy Wizard failed with exit code $exit_code"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "- Check the error log for details: $log_file"
    echo "- Ensure all required input files are present and valid"
    echo "- Verify Python environment has all required packages"
fi

echo ""
echo "📋 Detailed log saved to: $log_file"
echo "=========================================="

# Exit with the same code as the main script
exit $exit_code

