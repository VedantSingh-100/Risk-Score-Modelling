# HPC Usage Guide for Build Labels and Features

## Overview
This guide explains how to run the enhanced `build_labels_and_features.py` script on HPC systems with progress tracking.

## What's New
- **Progress bars**: Added tqdm progress bars to track long-running operations
- **HPC submission script**: Ready-to-use SLURM batch script for HPC clusters
- **Better logging**: Enhanced output with step-by-step progress tracking

## Files Added/Modified

### Modified: `build_labels_and_features.py`
- Added `tqdm` import for progress bars
- Enhanced main progress tracking with 7-step overall progress bar
- Added progress bars for:
  - Deduplication process
  - Leakage check computations
- Better console output with clearer step descriptions

### New: `submit_build_job.sh`
- SLURM batch script for HPC submission
- Resource allocation: 8 CPUs, 32GB RAM, 2-hour time limit
- Automatic file validation before execution
- Comprehensive logging and error handling
- Job statistics reporting

### New: `requirements.txt`
- Lists required Python packages with minimum versions
- Ensures tqdm and other dependencies are available

### New: `HPC_USAGE.md`
- This usage guide

## Usage Instructions

### 1. Setup (One-time)
```bash
# Navigate to project directory
cd /home/vhsingh/Parshvi_project

# Install required packages (if not already available)
pip install -r requirements.txt

# Or on HPC systems, you might need:
# pip install --user -r requirements.txt
```

### 2. Running Locally (for testing)
```bash
python build_labels_and_features.py
```

### 3. Running on HPC
```bash
# Submit the job to SLURM scheduler
sbatch submit_build_job.sh

# Check job status
squeue -u $USER

# Check job output (replace JOBID with actual job ID)
tail -f logs/build_JOBID.out

# Check for errors
tail -f logs/build_JOBID.err
```

## Resource Requirements

### Current Settings:
- **CPUs**: 8 cores
- **Memory**: 32GB RAM
- **Time limit**: 2 hours
- **Partition**: cpu (adjust based on your HPC)

### To Modify Resources:
Edit the `#SBATCH` directives in `submit_build_job.sh`:
```bash
#SBATCH --cpus-per-task=16     # For more CPU cores
#SBATCH --mem=64G              # For more memory
#SBATCH --time=04:00:00        # For longer time limit
```

## Progress Tracking Features

### Overall Progress
- 7-step main progress bar showing overall completion
- Clear step descriptions for each major phase

### Detailed Progress
- **Deduplication**: Shows progress when checking for near-duplicate label sources
- **Leakage Check**: Tracks Jaccard similarity computation across all features
- **File Operations**: Better feedback when loading and saving data

### Example Output
```
=== Deterministic label & feature build ===
Overall Progress: 14%|█▍        | 1/7 [00:05<00:32, Loading configuration and dictionary]
Loading variable dictionary...
Loaded dictionary sheet='Sheet1', name_col='Variables', fill_col='Fill_Rate'
Fill-rate filter @ 85.00%: kept 1247/1534 variables

Overall Progress: 29%|██▊       | 2/7 [00:12<00:28, Selecting label sources]
Loading label candidates...
Initial label pool size: 23 (negatives prioritized where recommended=True)

Checking for near-duplicate label sources...
Deduplicating: 100%|██████████| 23/23 [00:03<00:00, 7.2it/s]

Computing Jaccard similarity for leakage detection...
Leakage check: 100%|██████████| 1247/1247 [01:23<00:00, 15.0it/s]
```

## Output Files
All outputs are saved to `deterministic_build/` directory:
- `selected_label_sources.csv`: Final label sources with contribution shares
- `label_union.csv`: The target label vector
- `selected_features_initial.csv`: Features before leakage filtering
- `selected_features_final.csv`: Final feature list after all filters
- `guard_leakage_report.csv`: Detailed leakage analysis
- `guard_set.txt`: Variables excluded from features
- `build_summary.json`: Complete run summary and parameters

## Troubleshooting

### Job Fails to Start
- Check if all required input files exist
- Verify SLURM partition name matches your HPC system
- Ensure you have sufficient quota/permissions

### Out of Memory
- Increase `--mem` in the SLURM script
- Consider processing data in chunks if dataset is very large

### Job Times Out
- Increase `--time` limit
- Check if data loading is the bottleneck
- Consider using faster storage if available

### Import Errors
- Ensure all packages in `requirements.txt` are installed
- Check if your HPC system requires module loading
- Uncomment and modify module load lines in `submit_build_job.sh`

## Monitoring Progress

### Real-time Monitoring
```bash
# Watch job queue
watch squeue -u $USER

# Follow output in real-time
tail -f logs/build_JOBID.out
```

### Post-Job Analysis
```bash
# Check job efficiency
sacct -j JOBID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,State

# View complete output
less logs/build_JOBID.out
```

## Performance Notes
- Progress bars add minimal overhead (~1-2% runtime increase)
- Memory usage should be similar to original script
- Deduplication and leakage checks are the most time-consuming steps
- For very large datasets (>100k features), consider increasing time limits

