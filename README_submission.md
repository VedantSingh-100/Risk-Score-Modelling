# Risk Feature Selection - Submission Guide

This directory contains scripts for running the risk feature selection analysis on different computing environments.

## 📁 Files Overview

- `risk_feature_selection.py` - Main analysis script with progress tracking
- `submit_risk_analysis.sh` - SLURM job submission script for HPC clusters
- `run_local.sh` - Local execution script for workstations/laptops
- `README_submission.md` - This guide

## 🚀 Quick Start

### For HPC Clusters (SLURM)

```bash
# Submit job to SLURM queue
sbatch submit_risk_analysis.sh

# Check job status
squeue -u $USER

# View job output (replace JOBID with actual job ID)
tail -f logs/risk_analysis_JOBID.out
```

### For Local Execution

```bash
# Run directly on local machine
./run_local.sh

# Or if you prefer:
bash run_local.sh
```

## ⚙️ Configuration

### SLURM Script Configuration

Edit `submit_risk_analysis.sh` to adjust:

```bash
#SBATCH --time=02:00:00          # Walltime (adjust based on data size)
#SBATCH --partition=gpu          # Partition (change to cpu/gpu/etc.)
#SBATCH --cpus-per-task=8        # CPU cores
#SBATCH --mem=32G                # Memory allocation
#SBATCH --mail-user=$USER@domain # Email notifications
```

### Environment Setup

Both scripts will:
1. Create a Python virtual environment (`venv/`)
2. Install required packages automatically
3. Check for required input files

### Required Input Files

Ensure these files exist in the project directory:
- `50k_users_merged_data_userfile_updated_shopping.csv` (main dataset)
- `Internal_Algo360VariableDictionary_WithExplanation.xlsx` (data dictionary)
- `Variable_Classification_Table_v2.xlsx` (optional classification table)

## 📊 Expected Outputs

Both scripts will generate:
- `variable_catalog.csv` - Complete variable audit
- `candidate_targets_ranked.csv` - Potential target variables
- `feature_importance_consensus.csv` - Feature rankings (if target found)
- `model_card.md` - Analysis summary and recommendations
- `logs/` - Execution logs with timestamps

## 🔧 Customization

### Modify Resource Requirements

For larger datasets, adjust in the Python script:
```python
MAX_ROWS_FOR_MI = 120_000        # Increase for more data
MAX_FEATURES_MI = 3000           # Increase for more features
MAX_FEATURES_L1 = 2000           # Adjust based on memory
```

### Add Custom Modules

In the SLURM script, uncomment and modify:
```bash
module load python/3.9
module load anaconda3
conda activate your_environment
```

## 📝 Monitoring Progress

The scripts include comprehensive progress tracking:
- Real-time progress bars for long operations
- Step-by-step status messages
- Memory and timing information
- Detailed error logging

## ⚠️ Troubleshooting

### Common Issues

1. **Missing Files**: Check that all required input files are present
2. **Memory Issues**: Reduce `MAX_ROWS_FOR_MI` or request more memory
3. **Permission Errors**: Ensure scripts are executable (`chmod +x`)
4. **Module Loading**: Adjust module commands for your cluster

### Getting Help

Check the logs in `logs/` directory for detailed error messages:
```bash
# SLURM job logs
ls logs/risk_analysis_*.out
ls logs/risk_analysis_*.err

# Local execution logs
ls logs/analysis_output_*.log
```

## 📈 Performance Guidelines

### Dataset Size Recommendations

- **Small** (<100K rows, <1K features): Use default settings
- **Medium** (100K-1M rows, 1K-5K features): Increase memory to 64GB
- **Large** (>1M rows, >5K features): Consider distributed computing

### Typical Runtimes

- Small datasets: 5-15 minutes
- Medium datasets: 30-60 minutes  
- Large datasets: 1-3 hours

## 📧 Notifications

The SLURM script includes email notifications for:
- Job start (`BEGIN`)
- Job completion (`END`) 
- Job failure (`FAIL`)

Update the email address in the script:
```bash
#SBATCH --mail-user=your.email@domain.com
```
