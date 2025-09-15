# Risk Model Hyperparameter Sweep Usage Guide

This guide explains how to run the comprehensive hyperparameter sweep and MLP stacking pipeline with Weights & Biases logging.

## 🎯 Overview

The pipeline consists of two main stages:
1. **GBDT Hyperparameter Sweep**: Optimizes XGBoost or LightGBM using Optuna
2. **MLP + Stacking**: Trains neural network and creates calibrated ensemble

All runs are logged to Weights & Biases project "Risk Score" with clear run names for easy comparison.

## 🚀 Quick Start

### Local Testing (Quick)
```bash
# Quick test with 10 trials, 3-fold CV
bash scripts/run_quick_sweep.sh xgb
```

### Local Full Run
```bash
# Full XGBoost sweep (60 trials, 5-fold CV)
bash scripts/run_risk_model_sweep.sh xgb 60

# Full LightGBM sweep
bash scripts/run_risk_model_sweep.sh lgb 60
```

### HPC Deployment
```bash
# Submit XGBoost sweep to SLURM
sbatch submit_risk_sweep_job.sh xgb 60

# Submit LightGBM sweep to SLURM  
sbatch submit_risk_sweep_job.sh lgb 60
```

## 📊 Weights & Biases Integration

### Project Configuration
- **Project Name**: "Risk Score"
- **API Key**: Configured in scripts (update if needed)
- **Run Names**: 
  - GBDT: `XGB_Sweep_60trials_YYYYMMDD_HHMMSS`
  - Stacking: `Stack_XGB+MLP_YYYYMMDD_HHMMSS`

### Logged Metrics
**GBDT Sweep Runs:**
- `cv_auc_mean` - Cross-validation AUC (primary metric)
- `cv_auc_std` - AUC standard deviation
- `param_*` - All hyperparameters being optimized
- `trial` - Trial number
- `best_cv_auc` - Best score achieved
- `final_oof_auc` - Out-of-fold AUC with best params

**Stacking Runs:**
- Model comparison metrics (XGB vs MLP vs Stack)
- Final ensemble performance
- Calibration improvements

## 🔧 Configuration Options

### Algorithm Selection
```bash
# XGBoost (default)
bash scripts/run_risk_model_sweep.sh xgb

# LightGBM
bash scripts/run_risk_model_sweep.sh lgb
```

### Hyperparameter Search Space

**XGBoost Parameters:**
- `n_estimators`: 500-2000
- `learning_rate`: 0.01-0.3 (log scale)
- `max_depth`: 3-10
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `reg_alpha`: 1e-8 to 10.0 (log scale)
- `reg_lambda`: 1e-8 to 10.0 (log scale)
- `min_child_weight`: 1-10

**LightGBM Parameters:**
- `n_estimators`: 500-2000
- `learning_rate`: 0.01-0.3 (log scale)
- `num_leaves`: 10-300
- `max_depth`: 3-10
- `subsample`: 0.6-1.0
- `colsample_bytree`: 0.6-1.0
- `reg_alpha`: 1e-8 to 10.0 (log scale)
- `reg_lambda`: 1e-8 to 10.0 (log scale)
- `min_child_samples`: 5-100

### Custom Parameters
```bash
# Custom number of trials and CV folds
bash scripts/run_risk_model_sweep.sh xgb 100 5 42
#                                    ^   ^   ^ ^
#                                    |   |   | └─ seed
#                                    |   |   └─── cv_folds  
#                                    |   └─────── trials
#                                    └─────────── algorithm
```

## 📁 Output Structure

```
model_outputs/
├── gbdt_sweep_xgb/                 # GBDT optimization results
│   ├── best_params.json           # Optimal hyperparameters
│   ├── final_results.json         # Complete results summary
│   ├── all_trials.csv             # All optimization trials
│   ├── oof_predictions.csv        # Out-of-fold predictions
│   └── decile_analysis.csv        # Performance by decile
└── stack_xgb/                     # Stacking results
    ├── summary.csv                # Model comparison
    ├── oof_predictions.csv        # All model predictions
    ├── stacker_logit.json         # Stacking coefficients
    └── deciles_*.csv              # Decile analysis per model
```

## 🎯 Key Files Explained

### `best_params.json`
```json
{
  "algorithm": "xgb",
  "best_score": 0.876543,
  "best_params": {
    "n_estimators": 1200,
    "learning_rate": 0.05,
    "max_depth": 6,
    ...
  },
  "final_oof_auc": 0.875123,
  "final_oof_ap": 0.881234
}
```

### `summary.csv` (Stacking Results)
```csv
model,auc,ap,gini,ks
xgb,0.8751,0.8812,0.7502,0.6234
mlp,0.8698,0.8756,0.7396,0.6123
stack,0.8789,0.8845,0.7578,0.6289
```

## 🔍 Monitoring Progress

### Real-time Monitoring
1. **W&B Dashboard**: https://wandb.ai/[username]/Risk%20Score
2. **Local Logs**: Watch script output for progress
3. **HPC Logs**: `tail -f artifacts/logs/risk_sweep_[JOBID].out`

### Key Metrics to Watch
- **CV AUC**: Primary optimization metric (higher is better)
- **Trial Progress**: Number of completed trials
- **Best Score**: Current best CV AUC found
- **Parameter Trends**: Which parameters lead to better performance

## 🚨 Troubleshooting

### Common Issues

**Missing Dependencies:**
```bash
pip install optuna wandb xgboost lightgbm
```

**W&B Authentication:**
```bash
wandb login
# Or set WANDB_API_KEY in script
```

**Memory Issues:**
- Reduce `--trials` parameter
- Reduce `--n-splits` for CV
- Use smaller dataset for testing

**HPC Job Failures:**
- Check SLURM logs in `artifacts/logs/`
- Verify data files exist in `data/processed/`
- Ensure conda environment is activated

### Performance Tips

**Speed Optimization:**
- Use `run_quick_sweep.sh` for testing
- Reduce trials for initial validation
- Use fewer CV folds for faster iteration

**Quality Optimization:**
- Increase trials for better hyperparameter search
- Use 5-fold CV for robust validation
- Run multiple seeds for stability analysis

## 📈 Expected Results

### Typical Performance
- **Baseline XGBoost**: ~0.85-0.87 AUC
- **Optimized XGBoost**: ~0.87-0.89 AUC
- **MLP**: ~0.86-0.88 AUC
- **Stacked Ensemble**: ~0.88-0.90 AUC

### Runtime Estimates
- **Quick Test (10 trials)**: 10-20 minutes
- **Full Sweep (60 trials)**: 4-8 hours
- **MLP Stacking**: 30-60 minutes

## 🔄 Iterative Improvement

### Workflow Recommendations
1. Start with `run_quick_sweep.sh` for validation
2. Run full sweep with best algorithm
3. Analyze W&B results for parameter insights
4. Adjust search space if needed
5. Run production sweep with optimized parameters

### Parameter Analysis
Use W&B parallel coordinates plot to understand:
- Which parameters matter most
- Parameter interaction effects
- Optimal parameter ranges
- Diminishing returns on trials

## 🎉 Next Steps

After successful sweep completion:
1. **Review W&B Dashboard** for detailed analysis
2. **Extract Best Parameters** from `best_params.json`
3. **Validate Stacking Performance** in `summary.csv`
4. **Deploy Best Model** using optimal configuration
5. **Document Results** for future reference

