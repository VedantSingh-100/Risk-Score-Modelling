# Enhanced Engineer Features Usage Guide

## Overview
This guide explains how to use the enhanced `engineer_features.py` script with progress tracking and HPC submission capabilities.

## What's New in the Enhanced Version

### Progress Tracking Features
- **8-step overall progress bar**: Tracks major pipeline phases
- **Feature-level progress bars**: Shows progress for:
  - Feature statistics computation (pre and post engineering)
  - Numeric feature engineering (imputation, clipping, log transforms)
  - Categorical feature engineering (one-hot/frequency encoding)
  - Feature filtering and selection
  - Leakage analysis computations
- **Enhanced logging**: Clear step descriptions and better console output

### HPC Integration
- **Dedicated submission script**: `submit_engineer_job.sh` for SLURM clusters
- **Resource optimization**: 16 CPUs, 64GB RAM, 3-hour time limit
- **Comprehensive validation**: Checks for all required input files
- **Smart error handling**: Graceful failure reporting and diagnostics

## Files Overview

### Enhanced: `engineer_features.py`
Key improvements:
- Added `tqdm` import for progress bars
- 8-step overall progress tracking
- Progress bars for all major loops and computations
- Better error messages and status updates

### New: `submit_engineer_job.sh`
- SLURM batch script for HPC execution
- Automatic file validation and error checking
- Resource usage monitoring and reporting
- Handles both required and optional input files

### New: `ENGINEER_FEATURES_USAGE.md`
- This comprehensive usage guide

## Prerequisites

### Required Input Files
These files must exist before running the engineer script:
1. **`50k_users_merged_data_userfile_updated_shopping.csv`** - Main dataset
2. **`deterministic_build/selected_features_final.csv`** - Features selected by build script
3. **`deterministic_build/selected_label_sources.csv`** - Label sources from sweep
4. **`deterministic_build/guard_set.txt`** - Variables to exclude from features

### Optional Input Files
These will be used if present:
- **`deterministic_build/build_summary.json`** - Configuration used in build process

### Dependencies
```bash
pip install pandas numpy tqdm pyarrow openpyxl
```

## Usage Instructions

### 1. Setup and Validation
```bash
# Navigate to project directory
cd /home/vhsingh/Parshvi_project

# Ensure you've run the build script first
python build_labels_and_features.py

# Verify all required files exist
ls -la deterministic_build/
```

### 2. Local Execution (for testing)
```bash
# Basic usage with defaults
python engineer_features.py

# With custom parameters
python engineer_features.py \
    --data "50k_users_merged_data_userfile_updated_shopping.csv" \
    --selected-features "deterministic_build/selected_features_final.csv" \
    --label-sources "deterministic_build/selected_label_sources.csv" \
    --guard "deterministic_build/guard_set.txt" \
    --outdir "engineered" \
    --fill-threshold 0.85 \
    --auto-drop-high-leakage
```

### 3. HPC Execution
```bash
# Submit to SLURM scheduler
sbatch submit_engineer_job.sh

# Monitor job status
squeue -u $USER

# Follow progress in real-time
tail -f logs/engineer_JOBID.out

# Check for errors
tail -f logs/engineer_JOBID.err
```

## Progress Tracking Details

### Overall Pipeline Progress
The script shows an 8-step progress bar tracking:
1. **Loading inputs and configuration** - File reading and validation
2. **Building target labels** - Union of label sources
3. **Selecting and filtering features** - Apply guards and filters
4. **Computing pre-engineering statistics** - Raw feature analysis
5. **Engineering features** - Transformations and encoding
6. **Computing post-engineering statistics** - Final feature analysis
7. **Running leakage analysis** - Jaccard similarity checks
8. **Saving final outputs** - Write results to disk

### Detailed Progress Bars
- **Summarizing features**: Statistics computation for each feature
- **Filtering features**: Guard and selection rule application
- **Numeric engineering**: Imputation, clipping, and log transforms
- **Categorical engineering**: One-hot or frequency encoding
- **Leakage check**: Jaccard computation vs target label

### Example Output
```
=== Deterministic Feature Engineering Pipeline ===
Overall Progress: 12%|█▎        | 1/8 [00:15<01:45, Loading inputs and configuration]
Loading raw data...
   Data: 49,390 rows x 1,533 columns
Loading selected features (with descriptions & fill rate)...
   Selected by fill rate ≥ 0.85: 57 features (from 58)

Computing feature statistics...
Summarizing features: 100%|██████████| 57/57 [00:12<00:00, 4.5it/s]

Overall Progress: 62%|██████▎   | 5/8 [01:23<00:52, Engineering features]
Processing numeric features...
Numeric engineering: 100%|██████████| 57/57 [00:08<00:00, 6.8it/s]

Computing leakage analysis...
Leakage check: 100%|██████████| 57/57 [00:05<00:00, 10.3it/s]
```

## Resource Requirements

### Current HPC Settings
- **CPUs**: 16 cores (increased for feature engineering workload)
- **Memory**: 64GB RAM (higher for statistical computations)
- **Time limit**: 3 hours (generous for large datasets)
- **Partition**: cpu

### Customizing Resources
Edit `submit_engineer_job.sh` to modify:
```bash
#SBATCH --cpus-per-task=32     # For more CPU cores
#SBATCH --mem=128G             # For larger datasets
#SBATCH --time=06:00:00        # For longer processing
```

## Output Files

All outputs are saved to the `engineered/` directory:

### Primary Outputs
- **`X_features.parquet`** - Final engineered feature matrix (ready for ML)
- **`y_label.csv`** - Target label vector
- **`feature_list_final.csv`** - List of features included in final matrix

### Analysis Reports
- **`feature_engineering_report_pre.csv`** - Raw feature statistics
- **`feature_engineering_report_post.csv`** - Engineered feature statistics
- **`leakage_check.csv`** - Potential leakage analysis
- **`transforms_config.json`** - Complete transformation record

### Metadata
- **`label_stats.json`** - Target label distribution
- **`dropped_features.json`** - Features removed and reasons
- **`best_config_used.json`** - Configuration provenance (if available)

## Command Line Options

```bash
python engineer_features.py [OPTIONS]

Options:
  --data TEXT                    Data CSV path [default: 50k_users_merged_data_userfile_updated_shopping.csv]
  --selected-features TEXT       Selected features CSV [default: selected_features._finalcsv]
  --label-sources TEXT           Label sources CSV [default: Selected_label_sources.csv]
  --guard TEXT                   Guard set file [default: guard_Set.txt]
  --build-summary TEXT           Build summary JSON [default: build_summary.json]
  --outdir TEXT                  Output directory [default: engineered]
  --fill-threshold FLOAT         Fill rate threshold [default: 0.85]
  --auto-drop-high-leakage       Auto-drop high leakage features [default: False]
```

## Feature Engineering Pipeline Details

### Numeric Features
1. **Imputation**: Missing values filled with median
2. **Clipping**: Outliers clipped to 1st-99th percentile range
3. **Log Transform**: Applied if:
   - Feature is non-negative
   - Skewness > 2.0
   - Not identified as a ratio/percentage

### Categorical Features
1. **Imputation**: Missing values filled with mode
2. **Encoding Strategy**:
   - **One-hot**: ≤12 unique values
   - **Frequency**: >12 unique values (maps to relative frequency)

### Guard Rules
Features are excluded if they:
- Are missing from the dataset
- Are used as label sources
- Are in the guard set (leakage prevention)
- Look like identifiers (account numbers, IDs, etc.)

## Troubleshooting

### Common Issues

#### Missing Input Files
```
ERROR: Missing required files:
  ✗ deterministic_build/selected_features_final.csv
```
**Solution**: Run `build_labels_and_features.py` first to generate required files.

#### Out of Memory
**Symptoms**: Job killed, "Exceeded memory limit" in logs
**Solutions**:
- Increase `--mem` in SLURM script
- Reduce dataset size for testing
- Use chunked processing for very large datasets

#### Feature Engineering Fails
**Symptoms**: Assert errors, transformation failures
**Solutions**:
- Check data quality and missing values
- Verify feature selection didn't remove all features
- Review guard rules and thresholds

#### Slow Performance
**Symptoms**: Long runtimes, progress bars stalling
**Solutions**:
- Increase CPU allocation
- Check I/O performance (disk speed)
- Monitor memory usage

### Performance Optimization

#### For Large Datasets (>100k rows)
- Increase memory allocation to 128GB+
- Use 32+ CPU cores
- Consider data type optimization (int32 vs int64)

#### For Many Features (>1000)
- Progress bars help track bottlenecks
- Categorical encoding can be memory-intensive
- Consider feature pre-filtering

### Monitoring Job Progress

#### Real-time Monitoring
```bash
# Watch overall progress
tail -f logs/engineer_JOBID.out | grep "Overall Progress\|==="

# Monitor resource usage
sstat $JOBID

# Check memory usage
ssh $NODE_NAME "htop"
```

#### Post-Job Analysis
```bash
# Complete job statistics
sacct -j $JOBID --format=JobID,Elapsed,MaxRSS,MaxVMSize,CPUTime,State

# Review outputs
ls -la engineered/
du -h engineered/*
```

## Integration with ML Pipeline

The engineered outputs are ready for machine learning:

```python
import pandas as pd

# Load engineered features and labels
X = pd.read_parquet("engineered/X_features.parquet")
y = pd.read_csv("engineered/y_label.csv")["label"]

# Features are already:
# - Imputed (no missing values)
# - Scaled/transformed appropriately 
# - Encoded (all numeric)
# - Leakage-checked

# Ready for train/test split and modeling
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Best Practices

1. **Always run build script first** to generate required inputs
2. **Test locally** with small data samples before HPC submission
3. **Monitor resource usage** to optimize allocation
4. **Review leakage reports** before training models
5. **Save transformation configs** for applying to new data
6. **Use version control** for reproducible results

This enhanced pipeline provides comprehensive progress tracking and robust HPC execution for production-scale feature engineering workflows.

