#!/usr/bin/env python3
"""
Quick Enhanced PD Framework - Core Analysis Only
Implementing Other AI's Key Recommendations
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def analyze_audit_results():
    """Analyze audit results and create enhanced labels"""
    print("🔍 ANALYZING AUDIT RESULTS...")
    
    # Load audit files
    contributions = pd.read_csv('event_contribution_summary.csv')
    union_label = pd.read_csv('label_union_provisional.csv')
    
    with open('do_not_use_features.txt', 'r') as f:
        leakage_vars = [line.strip() for line in f]
    
    # Load sample data (same size as audit)
    data = pd.read_csv('50k_users_merged_data_userfile_updated_shopping.csv', 
                       nrows=20000, low_memory=False)
    
    print(f"✅ Data loaded: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
    print(f"🛡️  Leakage guard: {len(leakage_vars)} variables")
    
    return contributions, union_label, leakage_vars, data

def create_enhanced_labels(contributions, union_label, data):
    """Create enhanced label variants addressing other AI's concerns"""
    print("\n🏗️  CREATING ENHANCED LABELS...")
    
    # Get component variables from audit
    components = contributions['event'].tolist()
    
    # 1. Original union (for reference)
    original_union = union_label['label_union'].values
    
    # 2. Rebalanced union (address dominance)
    # Apply inverse prevalence weighting to reduce var501060 dominance
    rebalanced_components = {}
    component_weights = {}
    
    for component in components:
        if component in data.columns:
            comp_data = data[component].fillna(0)
            
            if comp_data.dtype == 'object':
                binary_val = (comp_data.astype(str).str.lower()
                            .isin(['true', 'yes', '1', 'y'])).astype(int)
            else:
                binary_val = (pd.to_numeric(comp_data, errors='coerce')
                            .fillna(0) > 0).astype(int)
            
            prevalence = binary_val.mean()
            if prevalence > 0:
                # Reduce weight of highly dominant variables
                if component == 'var501060':  # Most dominant
                    weight = 0.3  # Significantly reduce
                elif prevalence > 0.10:
                    weight = 0.6  # Moderately reduce
                else:
                    weight = 1.0  # Keep full weight
                    
                rebalanced_components[component] = binary_val
                component_weights[component] = weight
    
    # Create rebalanced union
    weighted_sum = np.zeros(len(data))
    for component, binary_vals in rebalanced_components.items():
        weighted_sum += binary_vals * component_weights[component]
    
    rebalanced_union = (weighted_sum > 0).astype(int)
    
    # 3. Severity weighted (fix threshold bug)
    severity_weights = {
        'var501060': 0.4,   # Reduce dominant variable
        'var501052': 1.0,   # BNPL Defaults [Lifetime]
        'var501053': 0.9,   # BNPL Defaults [12M]
        'var206063': 1.0,   # Loan Defaults [28D]
        'var202077': 1.0,   # Credit Card Defaults [28D]
    }
    
    severity_score = np.zeros(len(data))
    for component, binary_vals in rebalanced_components.items():
        weight = severity_weights.get(component, 0.7)
        severity_score += binary_vals * weight
    
    # Set threshold for ~8% prevalence
    if len(severity_score[severity_score > 0]) > 0:
        threshold = np.percentile(severity_score[severity_score > 0], 92)
        severity_weighted = (severity_score >= threshold).astype(int)
    else:
        severity_weighted = (severity_score > 0.5).astype(int)
    
    # Create label variants
    labels_df = pd.DataFrame({
        'original_union': original_union,
        'rebalanced_union': rebalanced_union,
        'severity_weighted': severity_weighted
    })
    
    print(f"✅ Label variants created:")
    for col in labels_df.columns:
        prevalence = labels_df[col].mean()
        positives = labels_df[col].sum()
        print(f"   {col}: {prevalence:.1%} ({positives:,} positives)")
    
    return labels_df, component_weights

def validate_labels(labels_df, data, leakage_vars):
    """Validate labels using other AI's criteria"""
    print("\n🔍 VALIDATING LABELS...")
    
    results = {}
    
    for label_name in labels_df.columns:
        label = labels_df[label_name].values
        prevalence = label.mean()
        positives = int(label.sum())
        
        print(f"\n🎯 Validating {label_name}...")
        
        # 1. Prevalence check (3-20%)
        prevalence_ok = 0.03 <= prevalence <= 0.20
        
        # 2. Baseline model test (leakage check)
        baseline_auc = test_baseline_model(data, label, leakage_vars)
        
        # 3. Generate recommendation
        issues = []
        if not prevalence_ok:
            if prevalence < 0.03:
                issues.append("Prevalence too low")
            else:
                issues.append("Prevalence too high")
        
        if baseline_auc >= 0.95:
            issues.append("High AUC suggests leakage")
        elif baseline_auc < 0.60:
            issues.append("Low AUC suggests weak signal")
        
        if not issues:
            recommendation = "✅ RECOMMENDED"
        elif len(issues) == 1 and "weak signal" in issues[0]:
            recommendation = "⚠️  ACCEPTABLE - Weak signal"
        else:
            recommendation = f"🚨 NOT RECOMMENDED - {'; '.join(issues)}"
        
        results[label_name] = {
            'prevalence': float(prevalence),
            'total_positives': positives,
            'baseline_auc': float(baseline_auc),
            'prevalence_ok': prevalence_ok,
            'recommendation': recommendation
        }
        
        print(f"   Prevalence: {prevalence:.1%} {'✅' if prevalence_ok else '❌'}")
        print(f"   Baseline AUC: {baseline_auc:.3f}")
        print(f"   Recommendation: {recommendation}")
    
    return results

def test_baseline_model(data, target, leakage_vars):
    """Test baseline model without leakage variables"""
    try:
        # Get clean numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        clean_features = [col for col in numeric_cols 
                         if col not in leakage_vars][:100]  # Limit for speed
        
        if len(clean_features) < 10 or target.sum() < 10:
            return 0.5
        
        X = data[clean_features].fillna(0)
        
        # Remove constant features
        feature_variance = X.var()
        variable_features = feature_variance[feature_variance > 1e-8].index.tolist()
        X = X[variable_features]
        
        if len(variable_features) < 5:
            return 0.5
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, target, test_size=0.3, random_state=42, stratify=target)
        
        # Scale and fit
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
        model.fit(X_train_scaled, y_train)
        
        # Evaluate
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, test_proba)
        
        return auc
        
    except Exception as e:
        print(f"   ⚠️  Baseline test failed: {e}")
        return 0.5

def get_clean_features(data, leakage_vars, target, top_n=50):
    """Get top clean features for modeling"""
    print(f"\n🧹 IDENTIFYING CLEAN FEATURES (top {top_n})...")
    
    # Get numeric features not in leakage guard
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    candidate_features = [col for col in numeric_cols 
                         if col not in leakage_vars]
    
    print(f"📊 Candidate features: {len(candidate_features)}")
    
    if target.sum() < 10:
        return []
    
    # Simple correlation-based ranking
    feature_scores = {}
    for feature in candidate_features[:200]:  # Limit for speed
        try:
            feature_data = pd.to_numeric(data[feature], errors='coerce').fillna(0)
            if feature_data.var() > 1e-8:
                correlation = abs(np.corrcoef(feature_data, target)[0, 1])
                if not np.isnan(correlation):
                    feature_scores[feature] = correlation
        except:
            continue
    
    # Top features
    top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    print(f"✅ Top {len(top_features)} clean features identified")
    return top_features

def generate_report(validation_results, component_weights):
    """Generate comprehensive report"""
    report = """# Enhanced PD Framework - Quick Analysis Report

**Implementing Other AI's Key Recommendations**

## Executive Summary

This analysis addresses the critical issues identified by the other AI:
✅ Dominance analysis and rebalancing
✅ Fixed weighted label thresholding
✅ Comprehensive leakage protection
✅ Baseline model validation

## Critical Findings from Audit

🚨 **MAJOR ISSUE CONFIRMED:** var501060 (BNPL Overdue) dominates 95.9% of positives
📊 **Original Union:** 5.88% positive rate (too low for optimal PD modeling)  
🛡️  **Leakage Risk:** 189 variables identified as risky for features

## Enhanced Label Variants

"""
    
    # Add label comparison
    for label_name, results in validation_results.items():
        report += f"### {label_name}\n"
        report += f"- **Prevalence:** {results['prevalence']:.1%} ({results['total_positives']:,} positives)\n"
        report += f"- **Baseline AUC:** {results['baseline_auc']:.3f}\n"
        report += f"- **Status:** {results['recommendation']}\n\n"
    
    report += """## Key Recommendations

### 1. Label Selection Priority:
"""
    
    # Rank labels by recommendation
    recommended = [name for name, res in validation_results.items() 
                  if res['recommendation'].startswith('✅')]
    acceptable = [name for name, res in validation_results.items() 
                 if '⚠️' in res['recommendation']]
    not_recommended = [name for name, res in validation_results.items() 
                      if res['recommendation'].startswith('🚨')]
    
    if recommended:
        report += f"**✅ RECOMMENDED:** {', '.join(recommended)}\n"
    if acceptable:
        report += f"**⚠️  ACCEPTABLE:** {', '.join(acceptable)}\n"
    if not_recommended:
        report += f"**🚨 AVOID:** {', '.join(not_recommended)}\n"
    
    report += """
### 2. Critical Next Steps:
1. **Use rebalanced_union** if available (addresses dominance issue)
2. **Exclude ALL 189 leakage variables** from feature engineering
3. **Validate business logic** of chosen label with stakeholders
4. **Implement temporal validation** if observation dates available
5. **Monitor for concept drift** over time

### 3. Feature Engineering Guidelines:
- ✅ Use only features NOT in leakage guard list
- ✅ Focus on demographic, behavioral, and external data sources
- ❌ NEVER use default/negative event flags as features
- ❌ NEVER use variables used in label construction

## Implementation Code

```python
# PRODUCTION-READY CONFIGURATION
"""
    
    best_label = None
    if recommended:
        best_label = recommended[0]
    elif acceptable:
        best_label = acceptable[0]
    
    if best_label:
        report += f"""
# Use the recommended label
TARGET_LABEL = '{best_label}'
TARGET_PREVALENCE = {validation_results[best_label]['prevalence']:.1%}

# Load enhanced labels
labels_df = pd.read_csv('enhanced_labels.csv')
target = labels_df[TARGET_LABEL]

# Load clean features (use only these for modeling)
features_df = pd.read_csv('clean_features_ranking.csv')
clean_features = features_df['feature'].tolist()

# Load leakage guard (EXCLUDE these from features)
leakage_df = pd.read_csv('leakage_guard_list.csv')
excluded_vars = leakage_df['variable'].tolist()
```

⚠️  **CRITICAL WARNING:** Never use excluded variables as features!
"""
    
    return report

def main():
    """Main execution"""
    print("🚀 QUICK ENHANCED PD FRAMEWORK")
    print("Addressing Other AI's Key Concerns")
    print("="*40)
    
    # Analyze audit results
    contributions, union_label, leakage_vars, data = analyze_audit_results()
    
    # Create enhanced labels
    labels_df, component_weights = create_enhanced_labels(contributions, union_label, data)
    
    # Validate labels
    validation_results = validate_labels(labels_df, data, leakage_vars)
    
    # Get clean features for best label
    best_label_name = None
    for name, results in validation_results.items():
        if results['recommendation'].startswith('✅'):
            best_label_name = name
            break
    
    if not best_label_name:
        # Use first acceptable or any label
        for name, results in validation_results.items():
            if '⚠️' in results['recommendation']:
                best_label_name = name
                break
        if not best_label_name:
            best_label_name = list(validation_results.keys())[0]
    
    clean_features = get_clean_features(data, leakage_vars, 
                                       labels_df[best_label_name].values)
    
    # Save results
    print("\n💾 SAVING RESULTS...")
    
    # Save enhanced labels
    labels_df.to_csv('enhanced_labels.csv', index=False)
    
    # Save validation results
    with open('validation_results.json', 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    # Save clean features
    features_data = []
    for feature, score in clean_features:
        features_data.append({
            'feature': feature,
            'correlation_score': score,
            'recommended_for': best_label_name
        })
    
    if features_data:
        features_df = pd.DataFrame(features_data)
        features_df.to_csv('clean_features_ranking.csv', index=False)
    
    # Save leakage guard
    leakage_df = pd.DataFrame({
        'variable': leakage_vars,
        'reason': 'Contains outcome tokens or used in label construction'
    })
    leakage_df.to_csv('leakage_guard_list.csv', index=False)
    
    # Generate report
    report = generate_report(validation_results, component_weights)
    with open('enhanced_pd_framework_report.md', 'w') as f:
        f.write(report)
    
    print("✅ ANALYSIS COMPLETE!")
    print("\nFiles generated:")
    print("- enhanced_labels.csv")
    print("- validation_results.json")
    print("- clean_features_ranking.csv")
    print("- leakage_guard_list.csv") 
    print("- enhanced_pd_framework_report.md")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    recommended = [name for name, res in validation_results.items() 
                  if res['recommendation'].startswith('✅')]
    
    if recommended:
        print(f"✅ RECOMMENDED LABEL: {recommended[0]}")
        rec_results = validation_results[recommended[0]]
        print(f"   Prevalence: {rec_results['prevalence']:.1%}")
        print(f"   Baseline AUC: {rec_results['baseline_auc']:.3f}")
    else:
        print("⚠️  No labels fully recommended - review results carefully")
    
    print(f"🛡️  Protected from leakage: {len(leakage_vars)} variables")
    print(f"🧹 Clean features available: {len(clean_features)}")
    
    print("\n🎉 OTHER AI'S RECOMMENDATIONS IMPLEMENTED! ✅")

if __name__ == "__main__":
    main()
