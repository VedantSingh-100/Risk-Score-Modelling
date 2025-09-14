# PIPELINE – Labels → Features → Models

This document describes the complete end-to-end pipeline used for the Parshvi Risk Model, including all decisions, thresholds, and methodologies.

## 1. Label Construction

### Target Variable: `label_union`

The target variable was constructed using a sophisticated union-based approach that combines multiple risk indicators.

**Key Configuration:**
- **Include lifetime signals**: Yes
- **Deduplication threshold**: Jaccard ≤ 0.85
- **Dominance cutoff**: ≥ 0.6
- **Quality threshold**: ≥ 0.6
- **Final sources**: 9 eligible sources
- **Overall prevalence**: 0.5203 (25,695 positives out of 49,389 samples)

**Label Sources:**
The final label sources are documented in `configs/Selected_label_sources.csv`. These were selected through a comprehensive sweep process that evaluated different combinations and thresholds.

**References:**
- `configs/best_config_used.json` - Winning sweep configuration
- `configs/build_summary.json` - Build metadata and statistics

## 2. Feature Engineering

### Initial Feature Selection

**Source**: Internal_Algo360 feature dictionary
**Selection Rule**: Keep only features with **Fill Rate ≥ 0.85**

This threshold was chosen to ensure data quality while maintaining sufficient feature diversity. The fill rate threshold is captured in `configs/build_summary.json`.

### Feature Exclusions

**Guard Lists:**
Features are excluded based on multiple guard mechanisms:

1. **Identifier Variables**: User IDs, row IDs, and other non-predictive identifiers
2. **Outcome-Related Features**: Variables that could cause leakage (e.g., overdue flags, charge-off indicators)
3. **Label Sources**: Variables used in label construction are excluded from features

**Guard Files:**
- `configs/guard_Set.txt` - Explicit exclusion list
- `dropped_features.json` - Categorized dropped features with reasons

### Transform Rules

Feature transformations are defined in `configs/transforms_config.json`:

**Count/Amount Variables:**
- Transform: `log1p(x)` (log(1 + x))
- Winsorization: Clip at 0.1% and 99.9% quantiles
- Rationale: Handle right-skewed distributions and outliers

**Ratio Variables:**
- Transform: `asinh(x)` (inverse hyperbolic sine)
- Imputation: Median imputation for missing values
- Rationale: Handle both positive and negative ratios symmetrically

**Score/Vintage Variables:**
- Imputation: Median imputation
- Scaling: Robust scaling for score variables
- Rationale: Preserve interpretability while handling outliers

### Quality Control

**Null/Infinite Value Policy:**
- Pre-transformation QC: Document baseline null rates
- Post-transformation QC: **0 null/infinite values** achieved
- Files: `feature_engineering_report_pre.csv` vs `feature_engineering_report_post.csv`

**Distribution Validation:**
- Expected shifts after log/asinh transforms are documented
- Outlier detection and treatment applied consistently
- QC summary available in `data/processed/qc_summary.md`

### Redundancy Management

**Detection:**
- Correlation threshold: |r| ≥ 0.97
- Method: Pairwise correlation analysis
- Output: `qc_redundancy_pairs.csv` (17 pairs identified)

**Resolution Strategy:**
1. **Graph-based approach** (baseline models): 
   - Build correlation graph of redundant features
   - Within each connected component, keep feature with highest single-variable AUC
   - Preserve interpretability and predictive power

2. **Threshold-based approach** (booster models):
   - Simple correlation threshold dropping
   - Faster for large feature sets

**Final Feature Count:**
- Initial (post-filter): 56 features
- After redundancy pruning: ~39 features (typical for boosters)

### Leakage Detection

**Automated Scanning:**
- Pattern matching for outcome-related terms
- Cross-reference with known problematic variables
- Output: `leakage_check.csv`

**Manual Review:**
- Domain expert review of feature names and definitions
- Validation against business logic
- No features flagged in current dataset

## 3. Model Training

### Cross-Validation Strategy

**Method**: Stratified K-Fold
- **Folds**: 5
- **Stratification**: Maintains class balance across folds
- **Seeds**: Multiple random seeds for stability analysis [42, 202, 404, 808, 1337]

### Baseline Models

**Logistic Regression (ElasticNet):**
- Penalty: ElasticNet (L1 + L2 combination)
- L1 ratio: 0.5 (equal L1/L2 weighting)
- Regularization: C = 1.0
- Preprocessing: RobustScaler with centering
- Solver: SAGA (handles ElasticNet penalty)

**Gradient Boosting (sklearn):**
- Estimators: 300 trees
- Learning rate: 0.05
- Max depth: 3 (conservative to prevent overfitting)
- No preprocessing (tree-based model)

**Stability Analysis:**
- PCS (Prediction Consistency Score) via Jaccard similarity
- Top-20 features compared across different random seeds
- High stability indicates robust feature selection

### Advanced Boosting Models

**XGBoost (Preferred):**
- Objective: binary:logistic
- Estimators: 5000 (with early stopping)
- Learning rate: 0.02 (conservative)
- Max depth: 6
- Regularization: L1=0.1, L2=0.2
- Subsampling: 0.8 (rows and columns)
- Early stopping: 300 rounds without improvement

**LightGBM (Alternative):**
- Similar hyperparameters to XGBoost
- Leaf-wise tree growth
- Automatic handling of categorical features

**HistGradientBoosting (Fallback):**
- sklearn's native gradient boosting
- Used when XGBoost/LightGBM unavailable
- Max iterations: 800
- L2 regularization: 1.0

### Probability Calibration

**Method**: Platt Scaling
- Logistic regression fitted on out-of-fold predictions
- Improves probability estimates for decision-making
- Maintains ranking while improving calibration

## 4. Model Evaluation

### Metrics

**Primary Metrics:**
- **AUC-ROC**: Area under ROC curve (discrimination)
- **AP (PR-AUC)**: Average Precision / Area under Precision-Recall curve

**Reporting:**
- Cross-validation mean ± standard deviation
- Out-of-fold predictions (raw and calibrated)
- Per-fold breakdown for variance analysis

### Feature Importance

**Methods:**
- **Tree-based models**: Built-in feature importance (gain-based)
- **Linear models**: Absolute coefficient magnitudes
- **Ranking**: Sorted by importance for interpretability

### Stability Assessment

**Feature Consistency:**
- Jaccard similarity of top-K features across CV folds
- High consistency indicates robust feature selection
- Reported for both baseline and boosting models

## 5. Output Artifacts

### Model Reports
- `model_cv_report.json` - Comprehensive performance metrics
- `cv_fold_metrics.csv` - Per-fold performance breakdown
- `feature_importance_cv.csv` - Feature importance rankings

### Predictions
- `oof_predictions.csv` - Out-of-fold predictions (raw and calibrated)
- Enables ensemble methods and further analysis

### Quality Control
- `qc_guard_removed.csv` - Features removed by guard lists
- `redundancy_drops.csv` - Features removed by redundancy pruning
- `pcs_stability_summary.json` - Stability analysis results

## 6. Reproducibility

### Configuration Management
- All hyperparameters stored in `configs/training_config.yaml`
- Random seeds fixed for reproducible results
- Transform rules explicitly documented

### Version Control
- Feature engineering rules versioned
- Model configurations tracked
- Guard lists and exclusions documented

### Environment
- Dependencies pinned in `requirements.txt`
- Python package versions specified
- Cross-platform compatibility ensured

## 7. Next Steps and Extensions

### Model Improvements
- **Neural Networks**: TabNet, NODE, or other deep tabular models
- **Ensemble Methods**: Combine multiple model types
- **Feature Selection**: Advanced selection algorithms

### Pipeline Enhancements
- **Automated Retraining**: Scheduled model updates
- **Drift Detection**: Monitor feature and target distributions
- **A/B Testing**: Framework for model comparison in production

### Monitoring
- **Performance Tracking**: Monitor AUC/AP over time
- **Feature Stability**: Track importance changes
- **Data Quality**: Automated QC checks

---

## Configuration References

- **Training Parameters**: `configs/training_config.yaml`
- **Feature Guards**: `configs/guard_Set.txt`
- **Label Sources**: `configs/Selected_label_sources.csv`
- **Transform Rules**: `configs/transforms_config.json` (if available)

## Quality Assurance Checklist

- ✅ Zero null/infinite values after preprocessing
- ✅ No leakage-prone features included
- ✅ Redundant features properly handled
- ✅ Cross-validation properly stratified
- ✅ Multiple random seeds for stability
- ✅ Probability calibration applied
- ✅ All artifacts properly documented
