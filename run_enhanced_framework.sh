#!/bin/bash

#=====================================================
# Enhanced PD Framework Execution Script
# Implements Other AI's Recommendations
#=====================================================

echo "🚀 ENHANCED PD FRAMEWORK EXECUTION"
echo "Implementing Other AI's Statistical Verification Recommendations"
echo "=================================================="

# Set working directory
cd /home/vhsingh/Parshvi_project

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install any additional required packages
echo "📦 Installing additional requirements..."
pip install --quiet scikit-learn pandas numpy tqdm

# Check required input files
echo "✅ Checking required files..."
required_files=(
    "50k_users_merged_data_userfile_updated_shopping.csv"
    "variable_catalog.csv"
    "event_contribution_summary.csv"
    "jaccard_matrix.csv" 
    "weighted_label_tuning.csv"
    "do_not_use_features.txt"
    "label_union_provisional.csv"
)

for file in "${required_files[@]}"; do
    if [[ -f "$file" ]]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✓ Found: $file ($size)"
    else
        echo "❌ Missing: $file"
        echo "⚠️  Run label_audit.py first to generate required files"
        exit 1
    fi
done

echo ""
echo "🎯 EXECUTING ENHANCED FRAMEWORK..."
echo "=================================================="

# Run the enhanced framework
python enhanced_pd_framework.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ENHANCED FRAMEWORK COMPLETED SUCCESSFULLY!"
    echo "=================================================="
    
    # Display generated files
    echo "📁 Generated Files:"
    echo "==================="
    
    output_files=(
        "enhanced_labels.csv"
        "clean_features_ranking.csv"
        "leakage_guard_list.csv"
        "validation_results.json"
        "enhanced_pd_framework_report.md"
    )
    
    total_size=0
    for file in "${output_files[@]}"; do
        if [[ -f "$file" ]]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            lines=$(wc -l < "$file" 2>/dev/null || echo "N/A")
            echo "✓ $file ($size, $lines lines)"
            
            # Add to total size calculation
            if [[ "$size" =~ ([0-9]+)K ]]; then
                total_size=$((total_size + ${BASH_REMATCH[1]}))
            elif [[ "$size" =~ ([0-9]+)M ]]; then
                total_size=$((total_size + ${BASH_REMATCH[1]} * 1024))
            fi
        else
            echo "❌ Missing: $file"
        fi
    done
    
    echo ""
    echo "📊 FRAMEWORK SUMMARY:"
    echo "===================="
    
    # Quick analysis of results
    if [[ -f "validation_results.json" ]]; then
        echo "🔍 Label Validation Results:"
        python -c "
import json
with open('validation_results.json', 'r') as f:
    results = json.load(f)

for label, data in results.items():
    status = '✅' if data['recommendation'].startswith('✅') else '⚠️' if 'ACCEPTABLE' in data['recommendation'] else '❌'
    print(f'  {status} {label}: {data[\"prevalence\"]:.1%} positive, AUC {data[\"baseline_auc\"]:.3f}')
"
    fi
    
    if [[ -f "enhanced_labels.csv" ]]; then
        echo ""
        echo "📈 Label Statistics:"
        python -c "
import pandas as pd
labels = pd.read_csv('enhanced_labels.csv')
print(f'  📊 Total samples: {len(labels):,}')
for col in labels.columns:
    pos_rate = labels[col].mean()
    pos_count = labels[col].sum()
    print(f'  📍 {col}: {pos_rate:.1%} ({pos_count:,} positives)')
"
    fi
    
    if [[ -f "leakage_guard_list.csv" ]]; then
        guard_count=$(wc -l < leakage_guard_list.csv)
        echo "🛡️  Leakage protection: $((guard_count - 1)) variables guarded"
    fi
    
    if [[ -f "clean_features_ranking.csv" ]]; then
        feature_count=$(wc -l < clean_features_ranking.csv)
        echo "🧹 Clean features: $((feature_count - 1)) features ranked"
    fi
    
    echo ""
    echo "🎯 NEXT STEPS:"
    echo "=============="
    echo "1. 📖 Review enhanced_pd_framework_report.md for detailed analysis"
    echo "2. 🎯 Choose recommended label from validation_results.json"
    echo "3. 🧹 Use clean_features_ranking.csv for feature selection"
    echo "4. 🛡️  Exclude variables in leakage_guard_list.csv"
    echo "5. 🚀 Implement in production modeling pipeline"
    
    echo ""
    echo "💡 CRITICAL REMINDERS:"
    echo "====================="
    echo "⚠️  Never use label components as features (data leakage)"
    echo "⚠️  Validate business logic with stakeholders"
    echo "⚠️  Implement temporal validation if dates available"
    echo "⚠️  Monitor for concept drift over time"
    
    echo ""
    echo "🎉 ENHANCED FRAMEWORK ANALYSIS COMPLETE!"
    echo "All other AI's recommendations have been implemented ✅"
    
else
    echo ""
    echo "❌ ENHANCED FRAMEWORK FAILED"
    echo "Check error messages above for details"
    exit 1
fi
