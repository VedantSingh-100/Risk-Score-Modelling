# Enhanced Model Training Pipeline

## Overview
The enhanced `train.py` script now features comprehensive progress tracking and HPC integration for robust model training with redundancy pruning and cross-validation.

## What's New

### Comprehensive Progress Tracking
- **7-phase overall progress bar**: Tracks the entire training pipeline
- **Detailed sub-progress bars** for:
  - Redundancy graph construction
  - Connected component discovery
  - Feature pruning decisions
  - Cross-validation training (seeds × folds)
  - Model performance evaluation

### Enhanced Training Workflow
```
=== Model Training Pipeline with Progress Tracking ===
Overall Progress: 43%|████▎     | 3/7 [02:15<03:10, Training Logistic Regression]

Training logit_en with 5 seeds × 5-fold CV...
logit_en seeds: 60%|██████    | 3/5 [01:45<01:10, seed=404]
  Fold 404: 100%|██████████| 5/5 [00:12<00:00, fold=3]

logit_en results: AUC = 0.8234 ± 0.0156, AP = 0.4567 ± 0.0234
```

### HPC Integration
- **Optimized resource allocation**: 20 CPUs, 64GB RAM, 4-hour limit
- **Intelligent file validation**: Checks for all required inputs with fallbacks
- **Comprehensive performance reporting**: Model metrics, stability analysis
- **Resource usage monitoring**: Memory, CPU, timing statistics

## Enhanced Features

### 1. **Intelligent Data Loading**
```python
print("Loading feature matrix and labels...")
X = pd.read_parquet(DATA_DIR / "X_features.parquet")
print(f"Loaded {X.shape[0]:,} samples with {X.shape[1]} features")
```

### 2. **Progress-Tracked Redundancy Analysis**
```python
print("Processing redundancy pairs...")
for _,r in tqdm(pairs.iterrows(), desc="Building graph", total=len(pairs)):
    # Build adjacency graph with progress

print("Finding connected components...")
for node in tqdm(nodes, desc="Finding components"):
    # BFS component discovery with progress
```

### 3. **Enhanced Cross-Validation**
```python
def cv_eval_model(clf_name, make_clf, X_df, y_vec, n_splits=5, seeds=[42,202,404,808,1337]):
    print(f"Training {clf_name} with {len(seeds)} seeds × {n_splits}-fold CV...")
    
    for seed in tqdm(seeds, desc=f"{clf_name} seeds"):
        for fold, (tr_idx, va_idx) in enumerate(tqdm(fold_splits, desc=f"Fold {seed}", leave=False)):
            # Nested progress bars for seeds and folds
```

### 4. **Comprehensive Results Reporting**
```python
print(f"{clf_name} results: AUC = {stab['mean_auc']:.4f} ± {stab['std_auc']:.4f}, "
      f"AP = {stab['mean_ap']:.4f} ± {stab['std_ap']:.4f}")
```

## File Requirements

### Primary Input Files
```
data/
├── X_features.parquet          # Engineered feature matrix
├── y_label.csv                 # Target labels
├── qc_redundancy_pairs.csv     # Feature redundancy analysis
└── qc_single_feature_metrics.csv  # Individual feature AUC scores
```

### Optional Configuration Files
```
data/
├── dropped_features.json      # Pre-dropped features metadata
├── guard_Set.txt             # Manual guard list
├── guard_set.txt             # Alternative guard file
├── do_not_use_features.txt   # Additional guard features
└── Selected_label_sources.csv # Label source variables (for guarding)
```

### Output Files Generated
```
data/outputs/
├── X_features_model.parquet      # Final pruned feature matrix
├── model_eval_summary.csv        # Cross-validation results
├── pcs_stability_summary.json    # Performance & stability metrics
├── feature_importance_lr.csv     # Logistic regression coefficients
├── feature_importance_gb.csv     # Gradient boosting importances
├── kept_features.csv            # Final feature list
└── redundancy_drops.csv         # Features removed by pruning
```

## Model Training Details

### Models Trained
1. **Logistic Regression (ElasticNet)**
   - Penalty: ElasticNet (L1 + L2) with `l1_ratio=0.5`
   - Regularization: `C=1.0`
   - Solver: SAGA (supports ElasticNet)
   - Preprocessing: RobustScaler

2. **Gradient Boosting Classifier**
   - Estimators: 300 trees
   - Learning rate: 0.05
   - Max depth: 3
   - No preprocessing (tree-based)

### Cross-Validation Strategy
- **5-fold stratified CV** × **5 random seeds** = 25 total fits per model
- Seeds: [42, 202, 404, 808, 1337]
- Out-of-fold predictions aggregated for final metrics
- Feature importance stability via top-20 Jaccard similarity

### Redundancy Pruning Algorithm
1. **Graph Construction**: Build adjacency graph from correlation pairs
2. **Component Discovery**: Find connected components via BFS
3. **Feature Selection**: Keep highest AUC feature from each component
4. **Systematic Removal**: Drop all other features in components

## Usage Instructions

### 1. Prerequisites
Ensure all input files are in the `data/` directory:
```bash
# Copy/symlink feature engineering outputs
ln -s /home/vhsingh/deterministic_fe_outputs/X_features.parquet data/
ln -s /home/vhsingh/deterministic_fe_outputs/y_label.csv data/

# Copy QC outputs
ln -s /home/vhsingh/qc_outputs/qc_*.csv data/

# Copy guard files
ln -s /home/vhsingh/deterministic_fe_outputs/dropped_features.json data/
```

### 2. Local Execution (for testing)
```bash
cd /home/vhsingh/Parshvi_project
python train.py
```

### 3. HPC Execution
```bash
cd /home/vhsingh/Parshvi_project
sbatch submit_train_job.sh
```

## Progress Tracking Examples

### Overall Pipeline Progress
```
=== Model Training Pipeline with Progress Tracking ===
Overall Progress: 57%|█████▋    | 4/7 [05:23<04:02, Training Gradient Boosting]

Redundancy pruning: keeping 45, dropping 8 features
Final feature matrix: 49,389 samples × 45 features
```

### Cross-Validation Progress
```
Training logit_en with 5 seeds × 5-fold CV...
logit_en seeds: 100%|██████████| 5/5 [03:45<00:00, seed=1337]
logit_en results: AUC = 0.8234 ± 0.0156, AP = 0.4567 ± 0.0234

Training gbdt with 5 seeds × 5-fold CV...
gbdt seeds: 80%|████████  | 4/5 [07:12<01:48, seed=808]
```

### Final Results Summary
```
=== Training Complete ===
Final model features: 45 (dropped 8 redundant)
Logistic Regression: AUC = 0.8234 ± 0.0156
Gradient Boosting:   AUC = 0.8456 ± 0.0134
Results saved to: data/outputs
```

## HPC Resource Configuration

### Current Settings
- **CPUs**: 20 cores (good for parallel CV)
- **Memory**: 64GB RAM (sufficient for feature matrices)
- **Time**: 4 hours (generous for 50 total model fits)
- **Partition**: cpu

### Resource Scaling Guidelines
```bash
# For larger datasets (>100k samples)
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00

# For more cross-validation (10 seeds)
#SBATCH --time=06:00:00

# For larger feature sets (>500 features)
#SBATCH --mem=96G
```

## Performance Metrics

### Model Evaluation
- **AUC-ROC**: Area under ROC curve
- **Average Precision (AP)**: Area under Precision-Recall curve
- **Stability**: Top-20 feature Jaccard similarity across seeds
- **Timing**: Fit and prediction times per fold

### Output Analysis
```python
import pandas as pd
import json

# Load results
eval_df = pd.read_csv("data/outputs/model_eval_summary.csv")
with open("data/outputs/pcs_stability_summary.json") as f:
    summary = json.load(f)

# Performance comparison
print(f"Logistic: {summary['logit_en']['mean_auc']:.4f} ± {summary['logit_en']['std_auc']:.4f}")
print(f"GBoosting: {summary['gbdt']['mean_auc']:.4f} ± {summary['gbdt']['std_auc']:.4f}")
```

## Best Practices

### Data Preparation
1. **Run feature engineering first** to generate X_features.parquet
2. **Complete QC analysis** to get redundancy pairs
3. **Verify file paths** match expected locations
4. **Check label distribution** for class imbalance

### Resource Management
1. **Monitor memory usage** for large feature matrices
2. **Adjust CPU allocation** based on dataset size
3. **Set appropriate time limits** for cross-validation workload
4. **Use progress bars** to monitor long-running jobs

### Results Interpretation
1. **Compare model performance** (AUC, AP) with uncertainty
2. **Assess feature stability** via Jaccard similarity
3. **Review feature importance** for interpretability
4. **Check redundancy pruning** effectiveness

This enhanced training pipeline provides production-ready model development with complete visibility into the training process, robust redundancy handling, and comprehensive performance evaluation.
