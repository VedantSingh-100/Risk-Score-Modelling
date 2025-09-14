# Deterministic Feature Engineering Usage Guide

## Overview
This guide explains the enhanced `determinstic_fe_build.py` script with intelligent feature type routing, progress tracking, and HPC submission capabilities.

## Script Highlights
This is an **exceptionally sophisticated** feature engineering pipeline with:
- **Intelligent feature type detection** based on variable names and descriptions
- **Automated transformation routing** (scores, vintages, ratios, counts, amounts)
- **Comprehensive leakage detection** using multiple metrics (AUC, Spearman, Jaccard)
- **Complete reproducibility** with detailed audit trails
- **Progress tracking** for all major operations

## What's New in the Enhanced Version

### Progress Tracking Features
- **8-step overall progress bar**: Tracks the entire pipeline from start to finish
- **Feature-level progress bars** for:
  - Feature filtering and selection
  - Pre-engineering statistical profiling
  - Intelligent feature engineering with type routing
  - Post-engineering statistical profiling  
  - Comprehensive leakage analysis
- **Detailed logging**: Clear descriptions of each transformation applied

### HPC Integration
- **`submit_deterministic_job.sh`**: SLURM batch script optimized for this workload
- **Smart file validation**: Checks for multiple input file variations
- **Resource optimization**: 12 CPUs, 48GB RAM, 2.5-hour time limit
- **Comprehensive reporting**: Detailed job statistics and output summaries

## Intelligent Feature Engineering

### Automatic Feature Type Detection
The script automatically detects feature types based on variable names and descriptions:

#### 1. **Scores** (`looks_like_score`)
- **Detection**: Contains "score", "propensity", or "affluence"
- **Processing**: Median imputation + Robust scaling
- **Rationale**: Scores are already normalized, just need centering/scaling

#### 2. **Vintages** (`looks_like_vintage`) 
- **Detection**: Contains "vintage" or "days"
- **Processing**: Median imputation, no transformation
- **Rationale**: Time periods are naturally interpretable

#### 3. **Ratios [0,1]** (`looks_like_ratio_01`)
- **Detection**: Contains "ratio" and "vintage" OR "balance is less than"
- **Processing**: Zero imputation + clipping to [0,1] + quantile clipping
- **Rationale**: Bounded ratios need constraint enforcement

#### 4. **Ratios (can exceed 1)** (`ratio_can_exceed_1`)
- **Detection**: "ratio of debit amount/credit amount"
- **Processing**: Median imputation + arcsinh transformation + quantile clipping
- **Rationale**: Signed log handles skewness and negative values

#### 5. **Counts** (`looks_like_count`)
- **Detection**: Starts with "no. of" or contains "count"
- **Processing**: Zero imputation + log1p transformation + quantile clipping
- **Rationale**: Counts are non-negative integers, log1p handles zeros

#### 6. **Amounts** (`looks_like_amount`, `looks_like_per_txn_avg`)
- **Detection**: Contains "amount", "sum", "upi" or "per transaction"
- **Processing**: Zero imputation + log1p transformation + quantile clipping
- **Rationale**: Financial amounts are non-negative, highly skewed

#### 7. **Fallback Routing**
- **Signed values**: arcsinh transformation (handles negatives)
- **Non-negative**: log1p transformation (standard for positive skewed data)

### Advanced Leakage Detection
Three complementary metrics for comprehensive leakage analysis:

1. **AUC (1D)**: Direct predictive power (`>= 0.92` flags)
2. **Spearman Correlation**: Monotonic relationship (`|r| >= 0.60` flags)  
3. **Jaccard Similarity**: Overlap in binary indicators (`>= 0.70` flags)

## Prerequisites

### Required Input Files
The script intelligently searches for files with multiple naming patterns:

#### Data Files
- **Main dataset**: `50k_users_merged_data_userfile_updated_shopping.csv` OR `raw.csv`

#### Feature Selection (one required)
- `selected_features._finalcsv` (your naming)
- `feature_list_final.csv` (standard naming)

#### Labels (one required)
- `y_label.csv` (direct label file)
- `label_union.csv` (union label file) 
- `Selected_label_sources.csv` (reconstruct from sources)

#### Optional Configuration Files
- **Guard sets**: `guard_Set.txt`, `guard_set.txt`, `do_not_use_features.txt`
- **Build config**: `build_summary.json`, `best_config_used.json`

### Dependencies
```bash
pip install pandas numpy scikit-learn tqdm pyarrow
```

## Usage Instructions

### 1. Local Execution (for testing)
```bash
# Basic usage - script auto-detects all input files
python determinstic_fe_build.py
```

### 2. HPC Execution  
```bash
# Submit to SLURM scheduler
sbatch submit_deterministic_job.sh

# Monitor job progress
squeue -u $USER

# Follow real-time progress
tail -f logs/deterministic_JOBID.out

# Check for errors
tail -f logs/deterministic_JOBID.err
```

## Progress Tracking Example

The enhanced script provides rich progress feedback:

```
=== Deterministic Feature Engineering Build ===
Overall Progress: 25%|██▌       | 2/8 [00:15<00:45, Building target labels]

Loading build configuration...
Using fill rate threshold: 0.85
Loading selected features list...
Initial selected features: 58
Features after fill rate filter: 55

Applying feature selection filters...
Filtering features: 100%|██████████| 55/55 [00:02<00:00, 25.3it/s]
Features to keep: 45
Dropped breakdown: {'guard': 3, 'identifier': 2, 'outcome_guard': 3, 'not_in_raw': 0}

Computing detailed statistics...
Pre-engineering stats: 100%|██████████| 45/45 [00:08<00:00, 5.2it/s]

Running intelligent feature engineering...
Engineering features: 100%|██████████| 45/45 [00:12<00:00, 3.7it/s]
Engineered features shape: (49390, 45)

Post-engineering stats: 100%|██████████| 45/45 [00:08<00:00, 5.4it/s]

Running comprehensive leakage analysis...
Leakage analysis: 100%|██████████| 45/45 [00:15<00:00, 2.9it/s]
```

## Output Files

### Primary ML-Ready Outputs
- **`X_features.parquet`** - Final engineered feature matrix (ready for modeling)
- **`y_label.csv`** - Target label vector

### Detailed Analysis Reports  
- **`feature_engineering_report_pre.csv`** - Raw feature statistics
- **`feature_engineering_report_post.csv`** - Engineered feature statistics
- **`leakage_check.csv`** - Comprehensive leakage analysis with flags

### Audit and Reproducibility
- **`transforms_config.json`** - Complete transformation record with parameters
- **`dropped_features.json`** - Features removed and reasons
- **`clip_quantiles`** - Quantile clipping thresholds used

## Resource Requirements

### Current HPC Settings
- **CPUs**: 12 cores (balanced for feature engineering)
- **Memory**: 48GB RAM (handles statistical computations)
- **Time**: 2.5 hours (generous for comprehensive analysis)

### Scaling Guidelines
```bash
# For larger datasets (>100k rows)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# For more features (>100)  
#SBATCH --cpus-per-task=20
#SBATCH --mem=96G
#SBATCH --time=06:00:00
```

## Feature Engineering Quality Metrics

### Transform Type Distribution
The script reports which transformations were applied:
```
Transform types used: {'score', 'vintage', 'count', 'amount', 'real_nonneg'}
```

### Leakage Flags
High-risk features are automatically flagged:
- **AUC > 0.92**: Suspiciously predictive
- **|Spearman| > 0.60**: Strong correlation  
- **Jaccard > 0.70**: High overlap
- **Outcome guard**: Contains outcome-related terms
- **Guard set**: Manually excluded features

## Advanced Configuration

### Customizing Detection Rules
Edit the helper functions to adjust feature type detection:

```python
def looks_like_score(name, desc):
    text = f"{name} {desc}".lower()
    return "score" in text or "propensity" in text or "affluence" in text
```

### Modifying Transformation Parameters
```python
CLIP_QUANTILES = (0.001, 0.999)  # Outlier clipping thresholds
FILL_RATE_THRESHOLD = 0.85       # Minimum fill rate
```

### Leakage Detection Thresholds
```python
# In the leakage analysis section
if not np.isnan(auc) and auc >= 0.92:          # AUC threshold
if not np.isnan(spearman) and abs(spearman) >= 0.60:  # Correlation threshold  
if not np.isnan(jac) and jac >= 0.70:          # Jaccard threshold
```

## Troubleshooting

### Common Issues

#### Missing Input Files
```
ERROR: Missing required files:
  ✗ feature selection file (one of: selected_features._finalcsv feature_list_final.csv)
```
**Solution**: Ensure you have the right feature selection file, possibly from running `build_labels_and_features.py` first.

#### No Features Remain
```
RuntimeError: No features left after guard/filters
```
**Solutions**:
- Check guard sets aren't too restrictive
- Verify feature selection files have valid entries
- Review outcome guard terms
- Check identifier detection patterns

#### Memory Issues
**Symptoms**: Job killed, "Exceeded memory limit"
**Solutions**:
- Increase `--mem` in SLURM script
- Use data type optimization (int32 vs int64)
- Consider chunked processing for very large datasets

#### Slow Statistical Computations
**Symptoms**: Long runtimes on profiling steps
**Solutions**:
- Increase CPU allocation
- Check for features with extreme cardinality
- Consider sampling for initial development

### Performance Optimization

#### For Financial/Transactional Data
- The intelligent routing is optimized for financial features
- Amount and ratio detection works well for banking data
- Score detection handles risk/propensity features

#### For High-Cardinality Features
- Identifier detection removes non-predictive IDs
- Quantile clipping handles extreme outliers
- Transform auditing tracks all changes

## Integration with ML Pipeline

The outputs are immediately usable for machine learning:

```python
import pandas as pd
import json

# Load engineered data
X = pd.read_parquet("X_features.parquet")
y = pd.read_csv("y_label.csv")["label"]

# Review transformations applied
with open("transforms_config.json") as f:
    transforms = json.load(f)
    
print(f"Features: {X.shape[1]}")
print(f"Transform types: {set(t['type'] for t in transforms['transforms'].values())}")

# Check for leakage flags
leakage = pd.read_csv("leakage_check.csv")
flagged = leakage[leakage["flagged"]]
print(f"Flagged features: {len(flagged)}")

# Ready for modeling - no further preprocessing needed
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Best Practices

1. **Review feature types**: Check that detection logic matches your domain
2. **Validate transformations**: Use pre/post reports to verify changes
3. **Monitor leakage flags**: Investigate flagged features before modeling
4. **Save configurations**: Keep transform configs for applying to new data
5. **Test incrementally**: Start with subset of features for development

This sophisticated pipeline provides production-ready feature engineering with full transparency and reproducibility - perfect for regulated environments requiring audit trails.
