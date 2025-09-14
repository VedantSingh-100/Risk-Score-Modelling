#!/bin/bash
#SBATCH --job-name=smart_framework_sweep
#SBATCH --output=logs/smart_framework_%j.out
#SBATCH --error=logs/smart_framework_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=vhsingh@andrew.cmu.edu

# Smart Variable Framework Hyperparameter Sweep Submission Script
# Runs comprehensive hyperparameter sweep for optimal label and feature configuration

echo "=========================================="
echo "🔄 Smart Variable Framework Hyperparameter Sweep"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Sweep Configuration: 160 configs, top 5 saved"
echo "=========================================="

# Send deployment notification
echo "🚀 HYPERPARAMETER SWEEP STARTED 🚀"
echo "✅ Job deployed successfully - safe to sleep!"
echo "📧 Email notifications enabled for BEGIN/END/FAIL"
echo "⏰ Expected runtime: 8-12 hours for 160 configurations"
echo "📊 Progress will be saved incrementally in sweep/sweep_results.csv"
echo ""

# Create logs and sweep directories if they don't exist
mkdir -p logs
mkdir -p sweep
mkdir -p sweep/best

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
pip install pandas numpy scikit-learn tqdm scipy openpyxl xlrd

# Print Python environment info
echo ""
echo "Python Environment:"
echo "- Python version: $(python --version)"
echo "- Python path: $(which python)"
echo "- Key packages:"
pip list | grep -E "(pandas|numpy|scikit-learn|tqdm|scipy)" || echo "Installing packages..."
echo ""

# Check if required files exist
echo "Checking required input files..."
required_files=(
    "50k_users_merged_data_userfile_updated_shopping.csv"
    "Internal_Algo360VariableDictionary_WithExplanation.xlsx"
    "smart_variable_framework.py"
)

optional_files=(
    "variable_catalog.csv"
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

for file in "${optional_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Optional: $file ($(du -h "$file" | cut -f1))"
    else
        echo "○ Optional (missing): $file"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "❌ Error: Missing required files. Cannot proceed."
    echo ""
    echo "Required files:"
    for file in "${missing_files[@]}"; do
        echo "  - $file"
    done
    echo ""
    exit 1
fi

echo ""

# Set up timing
echo "🔄 Starting Smart Variable Framework Hyperparameter Sweep..."
start_time=$(date +%s)

# Create timestamped log file
log_file="logs/smart_framework_sweep_$(date +%Y%m%d_%H%M%S).log"

# Run the smart framework sweep with specified arguments
echo "Executing hyperparameter sweep..."
echo "Command: python smart_variable_framework.py --sweep --sweep-max-configs 160 --sweep-save-top-k 5 --sweep-seed 42"
echo "Output will be saved to: $log_file"
echo ""
echo "🎯 Sweep will test 160 different configurations"
echo "💾 Top 5 configurations will be saved with full artifacts"
echo "📈 Progress can be monitored in sweep/sweep_results.csv"
echo "⚡ Each config tests: label strategies, thresholds, rescue policies"
echo ""

python smart_variable_framework.py --sweep --sweep-max-configs 160 --sweep-save-top-k 5 --sweep-seed 42 2>&1 | tee "$log_file"

# Capture exit code
exit_code=${PIPESTATUS[0]}

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_formatted=$(printf '%02d:%02d:%02d' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)))

echo ""
echo "=========================================="
echo "🏁 Hyperparameter Sweep Completion Summary"
echo "=========================================="
echo "Exit Code: $exit_code"
echo "Runtime: $runtime_formatted (HH:MM:SS)"
echo "End Time: $(date)"

# Check sweep-specific output files
echo ""
echo "Sweep Output Files Generated:"
sweep_output_files=(
    "sweep/sweep_results.csv"
    "sweep/best_config.json"
    "sweep/best"
)

# Check standard output files
echo ""
echo "Standard Output Files:"
output_files=(
    "negative_pattern_variables.csv"
    "smart_label_candidates.csv"
    "composite_labels.csv"
    "variable_quality_report.csv"
    "feature_importance_matrix.csv"
    "recommended_pipeline.json"
    "smart_framework_report.md"
)

# Check sweep files first
for file in "${sweep_output_files[@]}"; do
    if [[ "$file" == "sweep/best" ]]; then
        if [ -d "$file" ]; then
            best_count=$(find "$file" -name "rank_*" -type d | wc -l)
            echo "✓ $file/ (directory with $best_count ranked configurations)"
        else
            echo "✗ $file/ (directory not created)"
        fi
    elif [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        lines=$(wc -l < "$file" 2>/dev/null || echo "N/A")
        echo "✓ $file ($size, $lines lines)"
    else
        echo "✗ $file (not generated)"
    fi
done

echo ""
total_size=0
generated_count=0
for file in "${output_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        lines=$(wc -l < "$file" 2>/dev/null || echo "N/A")
        size_bytes=$(du -b "$file" | cut -f1)
        total_size=$((total_size + size_bytes))
        generated_count=$((generated_count + 1))
        echo "✓ $file ($size, $lines lines)"
    else
        echo "✗ $file (not generated)"
    fi
done

if [ $total_size -gt 0 ]; then
    total_size_mb=$((total_size / 1024 / 1024))
    echo ""
    echo "Total output size: ${total_size_mb} MB ($generated_count/$((${#output_files[@]})) files)"
fi

# Print memory usage if available
if command -v free &> /dev/null; then
    echo ""
    echo "Final Memory Usage:"
    free -h
fi

# Summary and recommendations
echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Hyperparameter Sweep completed successfully!"
    echo ""
    
    # Try to extract key insights from sweep results
    if [ -f "sweep/best_config.json" ]; then
        echo "🏆 Best Configuration Found:"
        
        # Extract best configuration details
        if command -v python3 &> /dev/null; then
            python3 -c "
import json
try:
    with open('sweep/best_config.json', 'r') as f:
        best = json.load(f)
    print(f'   🆔 Config ID: {best.get(\"config_id\", \"Unknown\")}')
    print(f'   📊 Score: {best.get(\"score\", \"N/A\")}')
    print(f'   🎯 AUC: {best.get(\"auc\", \"N/A\")}')
    print(f'   📈 Prevalence: {best.get(\"prevalence\", \"N/A\")}')
    print(f'   🔢 Label Sources: {best.get(\"num_sources\", \"N/A\")}')
    print(f'   ⚡ Runtime: {best.get(\"runtime_sec\", \"N/A\")}s')
except Exception as e:
    print(f'   Could not parse best config: {e}')
" 2>/dev/null
        fi
    fi
    
    # Count how many configs were tested
    if [ -f "sweep/sweep_results.csv" ]; then
        total_configs=$(tail -n +2 "sweep/sweep_results.csv" | wc -l)
        echo "   🔬 Total Configurations Tested: $total_configs"
    fi
    
    echo ""
    echo "📂 Generated Sweep Files:"
    echo "   1. 🏆 sweep/best_config.json - Optimal configuration parameters"
    echo "   2. 📊 sweep/sweep_results.csv - All tested configurations ranked by score"
    echo "   3. 📁 sweep/best/ - Top 5 configurations with full artifacts"
    echo "      • rank_01_<config_id>/ - Best configuration artifacts"
    echo "      • rank_02_<config_id>/ - Second best configuration artifacts"
    echo "      • ... (up to rank_05)"
    echo ""
    echo "📋 Each Best Configuration Directory Contains:"
    echo "   • composite_labels.csv - Optimized label variants"
    echo "   • event_contribution_summary.csv - How each event contributes"
    echo "   • jaccard_matrix.csv - Label source overlap analysis"
    echo "   • do_not_use_features.txt - Leakage guard list"
    echo "   • weighted_label_meta.json - Label threshold metadata"
    echo ""
    echo "🚀 Next Steps:"
    echo "   1. Review sweep/best_config.json for optimal hyperparameters"
    echo "   2. Examine sweep/best/rank_01_*/ for the best configuration artifacts"
    echo "   3. Use the best configuration's composite_labels.csv as your target"
    echo "   4. Apply the optimal hyperparameters to your production pipeline"
    echo "   5. Consider the top 2-3 configurations for ensemble approaches"
    echo ""
    echo "💡 Quick Start:"
    echo "   • Check sweep/best_config.json for the winning hyperparameters"
    echo "   • Use sweep/best/rank_01_*/composite_labels.csv as your labels"
    echo "   • Review sweep/sweep_results.csv to understand parameter effects"
    
else
    echo "❌ Hyperparameter Sweep failed with exit code $exit_code"
    echo ""
    echo "🔍 Troubleshooting Steps:"
    echo "   1. Check the error log: $log_file"
    echo "   2. Verify all input files are present and valid"
    echo "   3. Ensure sufficient memory (current: $SLURM_MEM_PER_NODE MB)"
    echo "   4. Check that Excel file contains 'Variables' and 'Explanation' columns"
    echo "   5. Verify CSV file is readable and contains user data"
    echo "   6. Check if partial results were saved in sweep/sweep_results.csv"
    echo ""
    echo "Common Issues:"
    echo "   • Excel file format not supported → Convert to CSV"
    echo "   • Column names don't match expected patterns → Check file structure"
    echo "   • Insufficient memory for large datasets → Request more memory"
    echo "   • Missing dependencies → Check Python environment"
    echo "   • Sweep timeout → Reduce --sweep-max-configs or increase time limit"
    echo ""
    echo "🔄 Partial Results:"
    if [ -f "sweep/sweep_results.csv" ]; then
        partial_configs=$(tail -n +2 "sweep/sweep_results.csv" | wc -l)
        echo "   📊 $partial_configs configurations were tested before failure"
        echo "   📂 Check sweep/sweep_results.csv for partial results"
    fi
fi

echo ""
echo "📋 Detailed execution log: $log_file"
echo "=========================================="

# Exit with the same code as the main script
exit $exit_code
