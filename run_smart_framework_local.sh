#!/bin/bash

# Local execution script for Smart Variable Framework
# Comprehensive variable analysis and label identification

echo "=========================================="
echo "Smart Variable Framework (Local Execution)"
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
echo "Required files:"
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Found: $file ($(du -h "$file" | cut -f1))"
    else
        echo "✗ Missing: $file"
        missing_files+=("$file")
    fi
done

echo ""
echo "Optional files:"
for file in "${optional_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ Available: $file ($(du -h "$file" | cut -f1))"
    else
        echo "○ Not found: $file (will be skipped)"
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo ""
    echo "❌ Error: Missing required files. Cannot proceed."
    echo ""
    echo "Missing files and how to obtain them:"
    for file in "${missing_files[@]}"; do
        case "$file" in
            "50k_users_merged_data_userfile_updated_shopping.csv")
                echo "  ❌ $file"
                echo "     → This should be your main user dataset"
                echo "     → Ensure the file name matches exactly"
                ;;
            "Internal_Algo360VariableDictionary_WithExplanation.xlsx")
                echo "  ❌ $file"
                echo "     → This should contain variable descriptions"
                echo "     → Must have 'Variables' and 'Explanation' columns"
                ;;
            "smart_variable_framework.py")
                echo "  ❌ $file"
                echo "     → This is the main analysis script"
                echo "     → Should have been created in current directory"
                ;;
        esac
        echo ""
    done
    
    echo "💡 Tips:"
    echo "   • Verify file names match exactly (case-sensitive)"
    echo "   • Ensure Excel file contains the expected column structure"
    echo "   • Check that CSV file is readable and not corrupted"
    echo ""
    exit 1
fi

echo ""

# Set up timing
echo "Starting Smart Variable Framework Analysis..."
start_time=$(date +%s)

# Create timestamped log file
log_file="logs/smart_framework_output_$(date +%Y%m%d_%H%M%S).log"

# Run the smart framework with error handling
echo "Executing smart_variable_framework.py..."
echo "Progress will be displayed below and saved to: $log_file"
echo ""

python smart_variable_framework.py 2>&1 | tee "$log_file"

# Capture exit code
exit_code=${PIPESTATUS[0]}

# Calculate runtime
end_time=$(date +%s)
runtime=$((end_time - start_time))
runtime_formatted=$(printf '%02d:%02d:%02d' $((runtime/3600)) $((runtime%3600/60)) $((runtime%60)))

echo ""
echo "=========================================="
echo "Analysis Completion Summary"
echo "=========================================="
echo "Exit Code: $exit_code"
echo "Runtime: $runtime_formatted (HH:MM:SS)"
echo "End Time: $(date)"

# Check output files and provide detailed analysis
echo ""
echo "📁 Generated Files Analysis:"
output_files=(
    "negative_pattern_variables.csv"
    "smart_label_candidates.csv"
    "composite_labels.csv"
    "variable_quality_report.csv"
    "feature_importance_matrix.csv"
    "recommended_pipeline.json"
    "smart_framework_report.md"
)

total_size=0
generated_count=0
for file in "${output_files[@]}"; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        size_bytes=$(du -b "$file" | cut -f1)
        total_size=$((total_size + size_bytes))
        generated_count=$((generated_count + 1))
        
        case "$file" in
            "negative_pattern_variables.csv")
                lines=$(tail -n +2 "$file" | wc -l)
                echo "✓ $file ($size) - $lines variables matching your negative patterns"
                ;;
            "smart_label_candidates.csv")
                lines=$(tail -n +2 "$file" | wc -l)
                echo "✓ $file ($size) - $lines label candidates analyzed"
                ;;
            "composite_labels.csv")
                cols=$(head -1 "$file" | tr ',' '\n' | wc -l)
                rows=$(tail -n +2 "$file" | wc -l)
                echo "✓ $file ($size) - $cols label variants, $rows data points"
                ;;
            "variable_quality_report.csv")
                lines=$(tail -n +2 "$file" | wc -l)
                echo "✓ $file ($size) - $lines features analyzed for quality"
                ;;
            "feature_importance_matrix.csv")
                cols=$(head -1 "$file" | tr ',' '\n' | wc -l)
                rows=$(tail -n +2 "$file" | wc -l)
                echo "✓ $file ($size) - $rows features × $cols importance scores"
                ;;
            "recommended_pipeline.json")
                echo "✓ $file ($size) - Machine-readable recommendations"
                ;;
            "smart_framework_report.md")
                lines=$(wc -l < "$file")
                echo "✓ $file ($size) - Comprehensive report ($lines lines)"
                ;;
        esac
    else
        echo "✗ $file (not generated)"
    fi
done

if [ $total_size -gt 0 ]; then
    total_size_mb=$((total_size / 1024 / 1024))
    echo ""
    echo "📊 Total Analysis Output: ${total_size_mb} MB across $generated_count files"
fi

# Extract and display key insights
echo ""
if [ $exit_code -eq 0 ]; then
    echo "🎉 Smart Variable Framework Analysis Completed Successfully!"
    echo ""
    
    # Extract key recommendations if available
    if [ -f "recommended_pipeline.json" ] && command -v python3 &> /dev/null; then
        echo "🎯 Key Findings:"
        python3 -c "
import json
import pandas as pd
try:
    # Load recommendations
    with open('recommended_pipeline.json', 'r') as f:
        rec = json.load(f)
    
    print(f'📌 Recommended Label: {rec.get(\"best_label\", \"None identified\")}')
    
    top_features = rec.get('top_features', [])
    print(f'🔧 Top Features Identified: {len(top_features)}')
    
    issues = rec.get('data_quality_issues', [])
    if issues:
        print(f'⚠️  Data Quality Issues: {len(issues)}')
        for issue in issues[:3]:  # Show first 3 issues
            print(f'   • {issue}')
    else:
        print(f'✅ Data Quality: No major issues detected')
    
    # Load label analysis if available
    if 'composite_labels.csv' in open('smart_framework_report.md').read():
        try:
            labels_df = pd.read_csv('composite_labels.csv')
            print(f'')
            print(f'🏷️  Label Variants Created:')
            for col in labels_df.columns:
                pos_rate = labels_df[col].mean()
                pos_count = labels_df[col].sum()
                print(f'   • {col}: {pos_rate:.4f} positive rate ({pos_count:,} cases)')
        except:
            pass
            
except Exception as e:
    print(f'Could not extract detailed insights: {e}')
" 2>/dev/null
    fi
    
    echo ""
    echo "📋 What to do next:"
    echo ""
    echo "1. 📖 READ THE REPORT:"
    echo "   → Open 'smart_framework_report.md' for comprehensive analysis"
    echo "   → This contains executive summary, recommendations, and implementation guidance"
    echo ""
    echo "2. 🎯 IMPLEMENT RECOMMENDATIONS:"
    echo "   → Use the recommended label as your target variable"
    echo "   → Start with the top-ranked features for initial modeling"
    echo "   → Address any data quality issues identified"
    echo ""
    echo "3. 🔧 CONFIGURE YOUR PIPELINE:"
    echo "   → Copy the recommended configuration from the report"
    echo "   → Update your modeling scripts with the suggested variables"
    echo "   → Validate the label definition with business stakeholders"
    echo ""
    echo "4. 📊 REVIEW DETAILED RESULTS:"
    echo "   → negative_pattern_variables.csv - Variables matching your exhaustive negative word research"
    echo "   → smart_label_candidates.csv - All potential labels ranked by suitability"
    echo "   → composite_labels.csv - Different ways to combine outcome variables"
    echo "   → variable_quality_report.csv - Quality assessment of all features"
    echo "   → feature_importance_matrix.csv - Feature importance for each label variant"
    echo ""
    echo "5. 🚀 START MODELING:"
    echo "   → Use the framework's recommendations as your starting point"
    echo "   → Consider time-based validation if temporal data is available"
    echo "   → Implement proper cross-validation with the identified features"
    
    echo ""
    echo "💡 Pro Tips:"
    echo "   • The 'composite_labels.csv' shows different label strategies - compare their performance"
    echo "   • High-importance features in the matrix are good candidates for feature engineering"
    echo "   • Variables marked as low quality might need special handling or removal"
    echo "   • The framework handles the complex task of identifying what to predict and what to use as inputs"
    
else
    echo "❌ Smart Variable Framework failed with exit code $exit_code"
    echo ""
    echo "🔍 Troubleshooting:"
    echo ""
    echo "Common Issues and Solutions:"
    echo "1. 📁 File Format Problems:"
    echo "   → Excel file cannot be read: Convert to CSV format"
    echo "   → Column names don't match: Ensure 'Variables' and 'Explanation' columns exist"
    echo "   → CSV encoding issues: Save with UTF-8 encoding"
    echo ""
    echo "2. 💾 Memory Issues:"
    echo "   → Large dataset: Framework automatically samples for analysis"
    echo "   → Insufficient RAM: Close other applications"
    echo "   → Many variables: Framework caps analysis at reasonable limits"
    echo ""
    echo "3. 🐍 Python Environment:"
    echo "   → Missing packages: Re-run to install dependencies"
    echo "   → Version conflicts: Use a fresh virtual environment"
    echo "   → Permission errors: Check file access permissions"
    echo ""
    echo "4. 📊 Data Structure Issues:"
    echo "   → No eligible labels found: Check that outcome variables exist in data"
    echo "   → All features filtered out: Review data quality in your source files"
    echo "   → Excel sheet selection: Framework automatically picks sheet with most data"
    echo ""
    echo "Check the detailed error log: $log_file"
fi

echo ""
echo "📋 Session Details:"
echo "   • Execution log: $log_file"
echo "   • Working directory: $(pwd)"
echo "   • Analysis timestamp: $(date)"
echo "   • Runtime: $runtime_formatted"
echo "=========================================="

# Exit with the same code as the main script
exit $exit_code
