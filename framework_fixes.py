#!/usr/bin/env python3
"""
Critical fixes for Smart Variable Framework
Addressing leakage and threshold issues identified in verification
"""

import pandas as pd
import numpy as np

def fix_weighted_label_threshold():
    """Fix the weighted label to achieve target 7-10% positive rate"""
    print("🔧 FIXING WEIGHTED LABEL THRESHOLD...")
    
    # Load composite labels
    labels_df = pd.read_csv('composite_labels.csv')
    
    # Reconstruct weighted score with proper weights
    severity_weights = {
        'var206002': 1.0,  # Missed Loan EMIS [Lifetime]
        'var308002': 1.0,  # Missed Utility Bill Payments [Lifetime]  
        'var202089': 0.8,  # Missed Minimum due Amount
        'var202003': 0.8,  # Overlimit On Credit Card
        'var701002': 0.8,  # Missed Insurance Payments
        'var701007': 0.8,  # Insurance Negative Events
        'var501003': 0.6,  # BNPL Negative Event
        'var501100': 0.6,  # BNPL Overlimit
        'var501060': 0.6   # BNPL Overdue
    }
    
    # Calculate weighted score
    weighted_score = np.zeros(len(labels_df))
    for var, weight in severity_weights.items():
        if var in labels_df.columns:
            weighted_score += labels_df[var] * weight
            print(f"Added {var} with weight {weight}")
    
    # Find threshold for ~7-10% positive rate
    target_rates = [0.07, 0.08, 0.09, 0.10]
    
    print("\nTesting thresholds for target positive rates:")
    for target_rate in target_rates:
        threshold = np.percentile(weighted_score[weighted_score > 0], 
                                (1 - target_rate) * 100)
        new_label = (weighted_score >= threshold).astype(int)
        actual_rate = new_label.mean()
        print(f"Target {target_rate:.1%} → Threshold {threshold:.2f} → Actual {actual_rate:.1%}")
    
    # Use threshold for 8% target
    optimal_threshold = np.percentile(weighted_score[weighted_score > 0], 92)
    fixed_weighted_label = (weighted_score >= optimal_threshold).astype(int)
    
    # Update the dataframe
    labels_df['label_weighted_fixed'] = fixed_weighted_label
    labels_df.to_csv('composite_labels_fixed.csv', index=False)
    
    print(f"\n✅ Fixed weighted label: {fixed_weighted_label.mean():.1%} positive rate")
    return fixed_weighted_label

def create_leakage_free_features():
    """Create clean feature list excluding all label components and outcome variables"""
    print("\n🚨 CREATING LEAKAGE-FREE FEATURE LIST...")
    
    # Load recommendations
    with open('recommended_pipeline.json', 'r') as f:
        import json
        recommendations = json.load(f)
    
    # Label construction variables (must exclude)
    label_vars = {
        'var206002', 'var308002', 'var202089', 'var202003', 'var701002', 'var701007',
        'var501003', 'var501100', 'var501060'
    }
    
    # All variables matching negative patterns (potential leakage)
    negative_pattern_df = pd.read_csv('negative_pattern_variables.csv')
    risky_vars = set(negative_pattern_df['variable'].tolist())
    
    # Current top features (many are problematic)
    current_top = [feat['feature'] for feat in recommendations['top_features']]
    
    print(f"Original top features: {len(current_top)}")
    print(f"Label construction variables: {len(label_vars)}")
    print(f"Total risky variables: {len(risky_vars)}")
    
    # Clean features (remove risky ones)
    clean_features = []
    for feat in current_top:
        base_var = feat.replace('_True', '').replace('_False', '')
        if base_var not in risky_vars and base_var not in label_vars:
            clean_features.append(feat)
    
    print(f"Clean features remaining: {len(clean_features)}")
    
    if len(clean_features) < 10:
        print("⚠️  Need to expand feature search beyond current top features")
        # Would need to re-run feature selection on clean variables only
    
    return clean_features, risky_vars

def generate_production_config():
    """Generate production-ready configuration with leakage guards"""
    print("\n🚀 GENERATING PRODUCTION-READY CONFIG...")
    
    # Fix weighted label
    fixed_weighted = fix_weighted_label_threshold()
    
    # Get clean features
    clean_features, risky_vars = create_leakage_free_features()
    
    # Production configuration
    config = {
        "RECOMMENDED_LABEL": "label_union",
        "ALTERNATIVE_LABEL": "label_weighted_fixed", 
        "POSITIVE_RATE_UNION": "7.5%",
        "POSITIVE_RATE_WEIGHTED_FIXED": f"{fixed_weighted.mean():.1%}",
        
        "LEAKAGE_GUARDS": {
            "EXCLUDE_VARIABLES": list(risky_vars),
            "EXCLUDE_PATTERNS": ["miss", "default", "overdue", "overlimit", "negative", "due", "reject", "decline"],
            "SAFE_FEATURE_COUNT": len(clean_features)
        },
        
        "NEXT_STEPS": [
            "Re-run feature selection excluding all risky variables",
            "Validate label definitions with business stakeholders", 
            "Implement time-based validation if dates available",
            "Test baseline model performance with clean features only"
        ]
    }
    
    with open('production_config.json', 'w') as f:
        import json
        json.dump(config, f, indent=2)
    
    print("✅ Production config saved to production_config.json")
    return config

if __name__ == "__main__":
    print("🔧 SMART FRAMEWORK CRITICAL FIXES")
    print("="*40)
    config = generate_production_config()
    print("\n🎯 SUMMARY: Framework fixes applied successfully!")
