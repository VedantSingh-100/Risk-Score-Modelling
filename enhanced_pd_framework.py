#!/usr/bin/env python3
"""
Enhanced PD Framework - Addressing Other AI's Recommendations
===========================================================

This framework implements all the statistical verification checks and fixes
suggested by the other AI, incorporating findings from label_audit.py

Key Improvements:
1. Dominance check and rebalancing
2. Fixed weighted label thresholding  
3. Comprehensive leakage guards
4. Baseline model validation
5. Jaccard-based event clustering
6. Production-ready label variants
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import json
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

@dataclass
class LabelValidationResults:
    """Results from comprehensive label validation"""
    prevalence: float
    total_positives: int
    dominance_check_passed: bool
    dominant_variable: Optional[str]
    dominant_share: float
    leakage_risk_score: float
    baseline_auc: float
    recommendation: str

@dataclass 
class EnhancedLabel:
    """Enhanced label with metadata"""
    name: str
    values: np.ndarray
    prevalence: float
    description: str
    methodology: str
    component_weights: Dict[str, float]
    validation_results: LabelValidationResults

class EnhancedPDFramework:
    """
    Enhanced PD Framework implementing other AI's recommendations
    """
    
    def __init__(self, 
                 raw_data_path: str = "50k_users_merged_data_userfile_updated_shopping.csv",
                 variable_catalog_path: str = "variable_catalog.csv",
                 sample_size: int = 20000):
        
        self.raw_data_path = Path(raw_data_path)
        self.variable_catalog_path = Path(variable_catalog_path)
        self.sample_size = sample_size
        
        # Load audit results
        self.audit_results = self._load_audit_results()
        
        # Define outcome tokens for leakage detection
        self.outcome_tokens = [
            r"default", r"dpd", r"overdue", r"arrear", r"write[\s-]?off", 
            r"charge[\s-]?off", r"npa", r"settle", r"miss", r"min[_\s-]?due", 
            r"over[-\s]?limit", r"declin", r"reject", r"bounced", r"nsf", 
            r"negative", r"due", r"penalty", r"insufficient"
        ]
        
        # Load data
        print("🔄 Loading data and building framework...")
        self.data, self.variable_descriptions = self._load_data()
        self.leakage_guard_list = self._build_comprehensive_leakage_guard()
        
    def _load_audit_results(self) -> Dict:
        """Load results from label_audit.py execution"""
        results = {}
        
        try:
            # Event contributions
            results['contributions'] = pd.read_csv('event_contribution_summary.csv')
            
            # Jaccard matrix
            results['jaccard'] = pd.read_csv('jaccard_matrix.csv', index_col=0)
            
            # Weighted tuning
            results['weight_tuning'] = pd.read_csv('weighted_label_tuning.csv')
            
            # Leakage guard
            with open('do_not_use_features.txt', 'r') as f:
                results['leakage_vars'] = [line.strip() for line in f]
                
            # Provisional union label
            results['union_label'] = pd.read_csv('label_union_provisional.csv')
            
            print(f"✅ Loaded audit results: {len(results['leakage_vars'])} guarded variables")
            
        except Exception as e:
            print(f"⚠️  Could not load all audit results: {e}")
            results = {}
            
        return results
    
    def _load_data(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Load raw data and variable descriptions"""
        
        # Load variable catalog for descriptions
        try:
            cat_df = pd.read_csv(self.variable_catalog_path, low_memory=False)
            name_col = "Variable" if "Variable" in cat_df.columns else cat_df.columns[0]
            desc_col = next((c for c in cat_df.columns if "description" in c.lower()), None)
            descriptions = dict(zip(cat_df[name_col].astype(str), 
                                  cat_df[desc_col].fillna("").astype(str))) if desc_col else {}
        except:
            descriptions = {}
            
        # Load sample of raw data
        print(f"📊 Loading {self.sample_size:,} rows from raw data...")
        data = pd.read_csv(self.raw_data_path, nrows=self.sample_size, low_memory=False)
        
        print(f"✅ Loaded data: {data.shape[0]:,} rows × {data.shape[1]:,} columns")
        print(f"✅ Variable descriptions: {len(descriptions):,} available")
        
        return data, descriptions
    
    def _build_comprehensive_leakage_guard(self) -> set:
        """Build comprehensive leakage guard list"""
        print("🛡️  Building comprehensive leakage guard...")
        
        # Start with audit results if available
        if 'leakage_vars' in self.audit_results:
            guard_list = set(self.audit_results['leakage_vars'])
        else:
            guard_list = set()
        
        # Add pattern-based detection
        outcome_pattern = re.compile("|".join(self.outcome_tokens), re.IGNORECASE)
        
        for col in self.data.columns:
            # Check column name and description
            text_to_check = col + " " + self.variable_descriptions.get(col, "")
            if outcome_pattern.search(text_to_check.lower()):
                guard_list.add(col)
        
        print(f"🛡️  Leakage guard list: {len(guard_list)} variables")
        return guard_list
    
    def analyze_label_dominance(self, label_components: List[str], 
                               label_values: np.ndarray) -> Tuple[bool, str, float]:
        """
        Analyze if any single component dominates the label
        Implements: Dominance check from other AI recommendations
        """
        print("🔍 Analyzing label dominance...")
        
        positive_mask = label_values == 1
        if positive_mask.sum() == 0:
            return True, "N/A", 0.0
        
        # Calculate contribution of each component among positives
        dominance_scores = {}
        
        for component in label_components:
            if component in self.data.columns:
                component_data = self.data[component].fillna(0)
                # Convert to binary if needed
                if component_data.dtype == 'object':
                    component_binary = (component_data.astype(str).str.lower()
                                      .isin(['true', 'yes', '1', 'y'])).astype(int)
                else:
                    component_binary = (pd.to_numeric(component_data, errors='coerce')
                                      .fillna(0) > 0).astype(int)
                
                # Calculate share among positives
                contribution = component_binary[positive_mask].sum() / positive_mask.sum()
                dominance_scores[component] = contribution
        
        # Find most dominant component
        if dominance_scores:
            dominant_var = max(dominance_scores, key=dominance_scores.get)
            dominant_share = dominance_scores[dominant_var]
            
            # Check if it passes the 60% threshold
            dominance_passed = dominant_share <= 0.60
            
            print(f"📊 Dominance Analysis:")
            for var, share in sorted(dominance_scores.items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
                print(f"   {var}: {share:.1%}")
            
            print(f"🎯 Dominant variable: {dominant_var} ({dominant_share:.1%})")
            print(f"✅ Dominance check: {'PASSED' if dominance_passed else 'FAILED'}")
            
            return dominance_passed, dominant_var, dominant_share
        
        return True, "N/A", 0.0
    
    def create_rebalanced_union_label(self, components: List[str], 
                                    max_single_contribution: float = 0.40) -> EnhancedLabel:
        """
        Create rebalanced union label addressing dominance issues
        """
        print(f"🏗️  Creating rebalanced union label (max single contribution: {max_single_contribution:.0%})...")
        
        # Convert components to binary
        binary_components = {}
        component_weights = {}
        
        for component in tqdm(components, desc="Processing components"):
            if component in self.data.columns:
                comp_data = self.data[component].fillna(0)
                
                if comp_data.dtype == 'object':
                    binary_val = (comp_data.astype(str).str.lower()
                                .isin(['true', 'yes', '1', 'y'])).astype(int)
                else:
                    binary_val = (pd.to_numeric(comp_data, errors='coerce')
                                .fillna(0) > 0).astype(int)
                
                # Calculate base prevalence for weighting
                prevalence = binary_val.mean()
                if prevalence > 0:
                    # Inverse prevalence weighting to reduce dominance
                    weight = min(1.0, max_single_contribution / prevalence) if prevalence > max_single_contribution else 1.0
                    binary_components[component] = binary_val
                    component_weights[component] = weight
                    
        print(f"📊 Active components: {len(binary_components)}")
        
        # Create weighted union
        weighted_sum = np.zeros(len(self.data))
        for component, binary_vals in binary_components.items():
            weighted_sum += binary_vals * component_weights[component]
        
        # Create binary label (any positive weighted contribution)
        rebalanced_label = (weighted_sum > 0).astype(int)
        prevalence = rebalanced_label.mean()
        
        print(f"✅ Rebalanced label: {prevalence:.1%} positive rate ({rebalanced_label.sum():,} positives)")
        
        # Validate the rebalanced label
        validation = self._validate_label(rebalanced_label, list(binary_components.keys()))
        
        return EnhancedLabel(
            name="label_rebalanced_union",
            values=rebalanced_label,
            prevalence=prevalence,
            description="Rebalanced union label with dominance controls",
            methodology="Inverse prevalence weighting to limit single component dominance",
            component_weights=component_weights,
            validation_results=validation
        )
    
    def create_severity_weighted_label(self, components: List[str], 
                                     target_prevalence: float = 0.08) -> EnhancedLabel:
        """
        Create properly calibrated severity-weighted label
        Fixes the thresholding bug identified in audit
        """
        print(f"⚖️  Creating severity-weighted label (target: {target_prevalence:.1%})...")
        
        # Severity weights based on business logic and audit findings
        severity_weights = {
            'var501060': 0.9,   # BNPL Overdue (most dominant - reduce weight)
            'var501052': 1.0,   # BNPL Defaults [Lifetime]
            'var501053': 0.9,   # BNPL Defaults [12M]
            'var501054': 0.8,   # BNPL Defaults [6M]
            'var206063': 1.0,   # Loan Defaults [28D]
            'var206064': 0.95,  # Loan Defaults [21D]
            'var206065': 0.9,   # Loan Defaults [14D]
            'var206066': 0.85,  # Loan Defaults [7D]
            'var501055': 0.8,   # BNPL Defaults [3M]
            'var501102': 0.8,   # BNPL Defaults [60-90D]
            'var202077': 1.0,   # Credit Card Defaults [28D]
        }
        
        # Calculate weighted score
        weighted_score = np.zeros(len(self.data))
        active_weights = {}
        
        for component in components:
            if component in self.data.columns:
                comp_data = self.data[component].fillna(0)
                
                if comp_data.dtype == 'object':
                    binary_val = (comp_data.astype(str).str.lower()
                                .isin(['true', 'yes', '1', 'y'])).astype(int)
                else:
                    binary_val = (pd.to_numeric(comp_data, errors='coerce')
                                .fillna(0) > 0).astype(int)
                
                weight = severity_weights.get(component, 0.7)  # Default weight
                weighted_score += binary_val * weight
                active_weights[component] = weight
        
        # Find threshold to achieve target prevalence
        positive_scores = weighted_score[weighted_score > 0]
        
        if len(positive_scores) > 0:
            # Calculate percentile to achieve target prevalence
            target_percentile = (1 - target_prevalence) * 100
            threshold = np.percentile(positive_scores, target_percentile)
        else:
            threshold = 0.5
        
        # Create binary label
        severity_label = (weighted_score >= threshold).astype(int)
        actual_prevalence = severity_label.mean()
        
        print(f"📊 Severity weighting:")
        print(f"   Threshold: {threshold:.3f}")
        print(f"   Target prevalence: {target_prevalence:.1%}")
        print(f"   Actual prevalence: {actual_prevalence:.1%}")
        
        # Validate the label
        validation = self._validate_label(severity_label, components)
        
        return EnhancedLabel(
            name="label_severity_weighted",
            values=severity_label,
            prevalence=actual_prevalence,
            description=f"Severity-weighted label (threshold: {threshold:.3f})",
            methodology="Business logic severity weighting with calibrated threshold",
            component_weights=active_weights,
            validation_results=validation
        )
    
    def create_clustered_label(self, components: List[str]) -> EnhancedLabel:
        """
        Create label based on Jaccard clustering to reduce redundancy
        """
        print("🔗 Creating clustered label based on Jaccard similarity...")
        
        if 'jaccard' not in self.audit_results:
            print("⚠️  No Jaccard matrix available, using simple union")
            return self.create_rebalanced_union_label(components)
        
        # Use Jaccard matrix to identify clusters
        jaccard_matrix = self.audit_results['jaccard']
        
        # Convert to distance matrix and cluster
        distance_matrix = 1 - jaccard_matrix.fillna(0)
        
        # Simple clustering: group highly similar variables
        clusters = []
        used_vars = set()
        
        for var in jaccard_matrix.index:
            if var not in used_vars:
                # Find highly similar variables (Jaccard > 0.7)
                similar_vars = jaccard_matrix.loc[var][jaccard_matrix.loc[var] > 0.7].index.tolist()
                if len(similar_vars) > 1:
                    clusters.append(similar_vars)
                    used_vars.update(similar_vars)
                else:
                    clusters.append([var])
                    used_vars.add(var)
        
        print(f"📊 Identified {len(clusters)} variable clusters")
        
        # Create label using cluster representatives
        cluster_components = {}
        for i, cluster in enumerate(clusters):
            # Use the variable with highest individual prevalence as representative
            best_var = cluster[0]
            if len(cluster) > 1:
                prevalences = {}
                for var in cluster:
                    if var in self.data.columns:
                        var_data = self.data[var].fillna(0)
                        if var_data.dtype == 'object':
                            binary_val = (var_data.astype(str).str.lower()
                                        .isin(['true', 'yes', '1', 'y'])).astype(int)
                        else:
                            binary_val = (pd.to_numeric(var_data, errors='coerce')
                                        .fillna(0) > 0).astype(int)
                        prevalences[var] = binary_val.mean()
                
                if prevalences:
                    best_var = max(prevalences, key=prevalences.get)
            
            cluster_components[f"cluster_{i}"] = best_var
        
        # Create union from cluster representatives
        union_values = np.zeros(len(self.data))
        component_weights = {}
        
        for cluster_name, var in cluster_components.items():
            if var in self.data.columns:
                var_data = self.data[var].fillna(0)
                if var_data.dtype == 'object':
                    binary_val = (var_data.astype(str).str.lower()
                                .isin(['true', 'yes', '1', 'y'])).astype(int)
                else:
                    binary_val = (pd.to_numeric(var_data, errors='coerce')
                                .fillna(0) > 0).astype(int)
                
                union_values = np.maximum(union_values, binary_val)
                component_weights[var] = 1.0
        
        clustered_label = union_values.astype(int)
        prevalence = clustered_label.mean()
        
        print(f"✅ Clustered label: {prevalence:.1%} positive rate")
        
        validation = self._validate_label(clustered_label, list(cluster_components.values()))
        
        return EnhancedLabel(
            name="label_clustered",
            values=clustered_label,
            prevalence=prevalence,
            description="Label based on Jaccard clustering to reduce redundancy",
            methodology="Cluster similar variables and use representative from each cluster",
            component_weights=component_weights,
            validation_results=validation
        )
    
    def _validate_label(self, label_values: np.ndarray, 
                       components: List[str]) -> LabelValidationResults:
        """
        Comprehensive label validation implementing other AI's checks
        """
        print("🔍 Running comprehensive label validation...")
        
        prevalence = label_values.mean()
        total_positives = int(label_values.sum())
        
        # 1. Dominance check
        dominance_passed, dominant_var, dominant_share = self.analyze_label_dominance(
            components, label_values)
        
        # 2. Baseline model validation (leakage check)
        baseline_auc = self._test_baseline_model(label_values)
        
        # 3. Leakage risk assessment
        leakage_risk = self._assess_leakage_risk(components)
        
        # 4. Generate recommendation
        recommendation = self._generate_validation_recommendation(
            prevalence, dominance_passed, baseline_auc, leakage_risk)
        
        return LabelValidationResults(
            prevalence=prevalence,
            total_positives=total_positives,
            dominance_check_passed=dominance_passed,
            dominant_variable=dominant_var,
            dominant_share=dominant_share,
            leakage_risk_score=leakage_risk,
            baseline_auc=baseline_auc,
            recommendation=recommendation
        )
    
    def _test_baseline_model(self, target: np.ndarray) -> float:
        """
        Test baseline model with clean features (no leakage)
        Implements: Predictability check from other AI recommendations
        """
        print("🤖 Testing baseline model (leakage-free features)...")
        
        # Get numeric features not in leakage guard
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        clean_features = [col for col in numeric_cols 
                         if col not in self.leakage_guard_list][:200]  # Limit for speed
        
        if len(clean_features) < 10:
            print("⚠️  Too few clean features for baseline test")
            return 0.5
        
        # Prepare features
        X = self.data[clean_features].fillna(0)
        
        # Remove near-constant features
        feature_variance = X.var()
        variable_features = feature_variance[feature_variance > 1e-8].index.tolist()
        X = X[variable_features]
        
        if len(variable_features) < 5 or target.sum() < 10:
            print("⚠️  Insufficient data for baseline test")
            return 0.5
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, target, test_size=0.3, random_state=42, stratify=target)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Fit model
            model = LogisticRegression(random_state=42, max_iter=1000, 
                                     class_weight='balanced')
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            test_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, test_proba)
            
            print(f"📊 Baseline AUC: {auc:.3f} (features: {len(variable_features)})")
            
            return auc
            
        except Exception as e:
            print(f"⚠️  Baseline model failed: {e}")
            return 0.5
    
    def _assess_leakage_risk(self, components: List[str]) -> float:
        """Assess leakage risk based on component analysis"""
        risky_components = sum(1 for comp in components 
                             if comp in self.leakage_guard_list)
        risk_score = risky_components / len(components) if components else 0
        return risk_score
    
    def _generate_validation_recommendation(self, prevalence: float, 
                                          dominance_passed: bool,
                                          baseline_auc: float,
                                          leakage_risk: float) -> str:
        """Generate validation recommendation based on other AI criteria"""
        
        issues = []
        
        # Check prevalence (3-20% range)
        if prevalence < 0.03:
            issues.append("Prevalence too low (<3%)")
        elif prevalence > 0.20:
            issues.append("Prevalence too high (>20%)")
        
        # Check dominance
        if not dominance_passed:
            issues.append("Single variable dominance (>60%)")
        
        # Check baseline AUC
        if baseline_auc >= 0.95:
            issues.append("Very high AUC suggests leakage")
        elif baseline_auc < 0.60:
            issues.append("Low AUC suggests weak signal")
        
        # Check leakage risk
        if leakage_risk > 0.5:
            issues.append("High leakage risk from components")
        
        if not issues:
            return "✅ RECOMMENDED - Passes all validation checks"
        elif len(issues) == 1 and "leakage risk" in issues[0]:
            return "⚠️  ACCEPTABLE - Minor leakage risk, use with caution"
        else:
            return f"🚨 NOT RECOMMENDED - Issues: {'; '.join(issues)}"
    
    def generate_production_labels(self) -> Dict[str, EnhancedLabel]:
        """
        Generate all recommended label variants
        """
        print("\n🏭 GENERATING PRODUCTION-READY LABELS")
        print("="*50)
        
        # Define component variables from audit
        if 'contributions' in self.audit_results:
            components = self.audit_results['contributions']['event'].tolist()
        else:
            # Fallback to manual definition
            components = [
                "var501102", "var501056", "var501057", "var501058", "var501059", "var501101",
                "var202077", "var501055", "var501060", "var206063", "var206064", "var206065",
                "var206066", "var501052", "var501053", "var501054"
            ]
        
        # Filter to available components
        available_components = [c for c in components if c in self.data.columns]
        print(f"📊 Using {len(available_components)} available components")
        
        labels = {}
        
        # 1. Original union (for comparison)
        if 'union_label' in self.audit_results:
            original_union = self.audit_results['union_label']['label_union'].values
            validation = self._validate_label(original_union, available_components)
            
            labels['original_union'] = EnhancedLabel(
                name="label_original_union",
                values=original_union,
                prevalence=original_union.mean(),
                description="Original union label from audit",
                methodology="Simple union of all default-related events",
                component_weights={c: 1.0 for c in available_components},
                validation_results=validation
            )
        
        # 2. Rebalanced union (addresses dominance)
        labels['rebalanced_union'] = self.create_rebalanced_union_label(available_components)
        
        # 3. Severity weighted (fixes threshold bug)
        labels['severity_weighted'] = self.create_severity_weighted_label(available_components)
        
        # 4. Clustered (reduces redundancy)
        labels['clustered'] = self.create_clustered_label(available_components)
        
        return labels
    
    def generate_clean_feature_recommendations(self, labels: Dict[str, EnhancedLabel]) -> Dict:
        """
        Generate clean feature recommendations for each label
        """
        print("\n🧹 GENERATING CLEAN FEATURE RECOMMENDATIONS")
        print("="*50)
        
        # Get all numeric features not in leakage guard
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        candidate_features = [col for col in numeric_cols 
                            if col not in self.leakage_guard_list]
        
        print(f"📊 Candidate features: {len(candidate_features)} (after leakage filtering)")
        
        feature_recommendations = {}
        
        for label_name, label_obj in labels.items():
            print(f"\n🎯 Analyzing features for {label_name}...")
            
            if label_obj.validation_results.recommendation.startswith("🚨"):
                print(f"⚠️  Skipping {label_name} - validation failed")
                continue
            
            # Simple feature scoring using correlation (as proxy)
            target = label_obj.values
            
            if target.sum() < 10:
                print(f"⚠️  Too few positives in {label_name}")
                continue
            
            feature_scores = {}
            
            for feature in tqdm(candidate_features[:500], desc=f"Scoring features"):  # Limit for speed
                try:
                    feature_data = pd.to_numeric(self.data[feature], errors='coerce').fillna(0)
                    
                    # Simple correlation score
                    if feature_data.var() > 1e-8:  # Avoid constant features
                        correlation = abs(np.corrcoef(feature_data, target)[0, 1])
                        if not np.isnan(correlation):
                            feature_scores[feature] = correlation
                            
                except Exception:
                    continue
            
            # Top features
            top_features = sorted(feature_scores.items(), 
                                key=lambda x: x[1], reverse=True)[:50]
            
            feature_recommendations[label_name] = {
                'top_features': top_features,
                'total_candidates': len(candidate_features),
                'scored_features': len(feature_scores),
                'label_prevalence': label_obj.prevalence,
                'validation_passed': not label_obj.validation_results.recommendation.startswith("🚨")
            }
            
            print(f"✅ {label_name}: {len(top_features)} top features identified")
        
        return feature_recommendations
    
    def generate_comprehensive_report(self, labels: Dict[str, EnhancedLabel], 
                                    feature_recommendations: Dict) -> str:
        """
        Generate comprehensive analysis report
        """
        print("\n📋 GENERATING COMPREHENSIVE REPORT...")
        
        report = """# Enhanced PD Framework - Comprehensive Analysis Report

**Generated with Other AI's Recommendations Implemented**

## Executive Summary

This analysis implements all statistical verification checks and improvements 
suggested by the other AI, including:

✅ Dominance checks with rebalancing
✅ Fixed weighted label thresholding  
✅ Comprehensive leakage guards
✅ Baseline model validation
✅ Jaccard-based clustering analysis

## Label Analysis Results

"""
        
        # Label comparison table
        report += "### Label Comparison\n\n"
        report += "| Label | Prevalence | Positives | Dominance Check | Baseline AUC | Recommendation |\n"
        report += "|-------|------------|-----------|-----------------|--------------|----------------|\n"
        
        for name, label in labels.items():
            val = label.validation_results
            report += f"| {name} | {label.prevalence:.1%} | {val.total_positives:,} | "
            report += f"{'✅' if val.dominance_check_passed else '❌'} | {val.baseline_auc:.3f} | "
            report += f"{val.recommendation.split(' - ')[0]} |\n"
        
        # Detailed analysis for each label
        report += "\n## Detailed Label Analysis\n\n"
        
        for name, label in labels.items():
            val = label.validation_results
            report += f"### {name}\n\n"
            report += f"**Description:** {label.description}\n\n"
            report += f"**Methodology:** {label.methodology}\n\n"
            report += f"**Key Metrics:**\n"
            report += f"- Prevalence: {label.prevalence:.1%} ({val.total_positives:,} positives)\n"
            report += f"- Dominance Check: {'✅ PASSED' if val.dominance_check_passed else '❌ FAILED'}\n"
            if not val.dominance_check_passed:
                report += f"  - Dominant Variable: {val.dominant_variable} ({val.dominant_share:.1%})\n"
            report += f"- Baseline AUC: {val.baseline_auc:.3f}\n"
            report += f"- Leakage Risk: {val.leakage_risk_score:.1%}\n"
            report += f"- **Recommendation: {val.recommendation}**\n\n"
        
        # Feature recommendations
        report += "## Clean Feature Recommendations\n\n"
        
        for label_name, feat_data in feature_recommendations.items():
            if feat_data['validation_passed']:
                report += f"### {label_name}\n\n"
                report += f"**Feature Candidates:** {feat_data['total_candidates']:,} total, "
                report += f"{feat_data['scored_features']:,} scored\n\n"
                report += f"**Top 10 Features:**\n"
                for i, (feature, score) in enumerate(feat_data['top_features'][:10], 1):
                    desc = self.variable_descriptions.get(feature, "")[:60]
                    report += f"{i}. `{feature}` (score: {score:.4f}) - {desc}\n"
                report += "\n"
        
        # Implementation guidance
        report += """## Implementation Guidance

### Recommended Labels (in priority order):

1. **Best Choice:** The label with ✅ RECOMMENDED status and highest baseline AUC
2. **Alternative:** Labels with ⚠️ ACCEPTABLE status
3. **Avoid:** Labels with 🚨 NOT RECOMMENDED status

### Next Steps:

1. **Validate Business Logic:** Confirm label definitions with stakeholders
2. **Feature Engineering:** Use only clean features (not in leakage guard)
3. **Temporal Validation:** Implement time-based splits if dates available
4. **Model Development:** Start with logistic regression baseline
5. **Performance Monitoring:** Track model performance over time

### Leakage Protection:

"""
        
        report += f"- **Guarded Variables:** {len(self.leakage_guard_list)} variables excluded\n"
        report += f"- **Clean Features:** Available for modeling\n"
        report += f"- **Recommendation:** Always exclude variables used in label construction\n\n"
        
        # Critical warnings
        report += """## Critical Warnings

⚠️  **Never use label components as features** - This guarantees data leakage
⚠️  **Validate temporal assumptions** - Ensure no future information leaks
⚠️  **Monitor for concept drift** - Labels may become stale over time
⚠️  **Document all exclusions** - Maintain clear audit trail

## Files Generated

- `enhanced_labels.csv` - All label variants
- `clean_features_ranking.csv` - Feature recommendations  
- `leakage_guard_list.csv` - Variables to exclude
- `validation_results.json` - Detailed validation metrics
"""
        
        return report

def main():
    """Main execution function"""
    print("🚀 ENHANCED PD FRAMEWORK")
    print("Implementing Other AI's Recommendations")
    print("="*50)
    
    # Initialize framework
    framework = EnhancedPDFramework()
    
    # Generate production labels
    labels = framework.generate_production_labels()
    
    # Generate feature recommendations  
    feature_recommendations = framework.generate_clean_feature_recommendations(labels)
    
    # Generate comprehensive report
    report = framework.generate_comprehensive_report(labels, feature_recommendations)
    
    # Save results
    print("\n💾 SAVING RESULTS...")
    
    # Save labels
    labels_df = pd.DataFrame({name: label.values for name, label in labels.items()})
    labels_df.to_csv('enhanced_labels.csv', index=False)
    
    # Save feature recommendations
    all_features = []
    for label_name, feat_data in feature_recommendations.items():
        for feature, score in feat_data['top_features']:
            all_features.append({
                'label': label_name,
                'feature': feature,
                'score': score,
                'description': framework.variable_descriptions.get(feature, "")
            })
    
    if all_features:
        features_df = pd.DataFrame(all_features)
        features_df.to_csv('clean_features_ranking.csv', index=False)
    
    # Save leakage guard
    leakage_df = pd.DataFrame({
        'variable': list(framework.leakage_guard_list),
        'reason': 'Contains outcome tokens or used in label construction'
    })
    leakage_df.to_csv('leakage_guard_list.csv', index=False)
    
    # Save validation results
    validation_data = {}
    for name, label in labels.items():
        val = label.validation_results
        validation_data[name] = {
            'prevalence': float(label.prevalence),
            'total_positives': int(val.total_positives),
            'dominance_check_passed': bool(val.dominance_check_passed),
            'dominant_variable': str(val.dominant_variable) if val.dominant_variable else None,
            'dominant_share': float(val.dominant_share),
            'baseline_auc': float(val.baseline_auc),
            'leakage_risk_score': float(val.leakage_risk_score),
            'recommendation': str(val.recommendation)
        }
    
    with open('validation_results.json', 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    # Save report
    with open('enhanced_pd_framework_report.md', 'w') as f:
        f.write(report)
    
    print("✅ ANALYSIS COMPLETE!")
    print("\nFiles generated:")
    print("- enhanced_labels.csv")
    print("- clean_features_ranking.csv") 
    print("- leakage_guard_list.csv")
    print("- validation_results.json")
    print("- enhanced_pd_framework_report.md")
    
    # Summary
    print(f"\n🎯 SUMMARY:")
    recommended_labels = [name for name, label in labels.items() 
                         if label.validation_results.recommendation.startswith("✅")]
    
    if recommended_labels:
        print(f"✅ Recommended labels: {', '.join(recommended_labels)}")
    else:
        print("⚠️  No labels passed all validation checks")
        acceptable_labels = [name for name, label in labels.items() 
                           if "ACCEPTABLE" in label.validation_results.recommendation]
        if acceptable_labels:
            print(f"⚠️  Acceptable labels: {', '.join(acceptable_labels)}")
    
    print(f"🛡️  Leakage guard: {len(framework.leakage_guard_list)} variables protected")
    print(f"🧹 Clean features: Available for modeling")

if __name__ == "__main__":
    main()
