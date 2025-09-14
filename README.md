# Parshvi Risk Model

Production-grade repository for building labels, engineering features, and training binary risk models using baseline and gradient boosting approaches.

## 🎯 Project Overview

This repository contains a complete machine learning pipeline for risk modeling, including:
- **Label Engineering**: Union-based target construction from multiple sources
- **Feature Engineering**: Transform-based preprocessing with redundancy pruning
- **Model Training**: Baseline (Logistic + GBDT) and advanced boosting models
- **Quality Control**: Comprehensive validation and leakage detection

## 📊 Key Results

- **Target**: `label_union` with prevalence ≈ 0.52 (25,695 positives from 49,389 samples)
- **Features**: 39 final features (after redundancy pruning from 56 initial features)
- **Performance**: XGBoost CV AUC ~0.868, AP ~0.884
- **Stability**: High feature consistency across CV folds

## 🏗️ Repository Structure

```
parshvi-risk-model/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
├── configs/                    # Configuration files
│   ├── training_config.yaml    # Training parameters
│   ├── guard_Set.txt          # Features to exclude
│   └── Selected_label_sources.csv # Label source configuration
├── data/
│   ├── raw/                   # Raw data files (not tracked)
│   ├── interim/               # Intermediate processing
│   └── processed/             # Clean, processed data
│       ├── X_features.parquet # Feature matrix
│       └── y_label.csv        # Target labels
├── src/                       # Source code
│   ├── models/
│   │   ├── train_baselines.py # Logistic + GBDT training
│   │   ├── train_boosters.py  # XGBoost/LightGBM training
│   │   └── train_advanced.py  # Original advanced training script
│   ├── preprocessing/         # Data preprocessing modules
│   └── utils/                 # Utility functions
├── scripts/                   # Execution scripts
│   └── repo_relayout.sh      # Repository organization script
├── docs/                      # Documentation
│   ├── PIPELINE.md           # Detailed pipeline documentation
│   └── *.md                  # Other documentation files
└── artifacts/                 # Generated outputs (not tracked)
    ├── models/               # Trained model files
    ├── reports/              # Training reports and metrics
    └── logs/                 # Execution logs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone <repository-url>
cd parshvi-risk-model

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training

**Baseline Models (Logistic + GBDT):**
```bash
python -m src.models.train_baselines \
    --data-dir data/processed \
    --config-dir configs \
    --out-dir artifacts/reports/baselines
```

**Boosting Models (XGBoost/LightGBM):**
```bash
python -m src.models.train_boosters \
    --data-dir data/processed \
    --out-dir artifacts/reports/boosters \
    --redundancy-r 0.97
```

### 3. View Results

Results are saved in `artifacts/reports/` with:
- Cross-validation metrics
- Feature importance rankings
- Out-of-fold predictions
- Model performance summaries

## 📋 Pipeline Overview

### Labels
- **Target**: `label_union` constructed from 9 eligible sources
- **Strategy**: Lifetime signals included with deduplication
- **Quality**: Dominance ≥ 0.6, quality ≥ 0.6, Jaccard ≤ 0.85

### Features
- **Selection**: Fill rate ≥ 0.85 from Internal_Algo dictionary
- **Transforms**: log1p for counts, asinh for ratios, robust scaling
- **Guards**: Identifier and outcome-related feature removal
- **Redundancy**: Graph-based pruning of highly correlated features (r ≥ 0.97)

### Models
- **Baselines**: Logistic Regression (ElasticNet) + sklearn GradientBoosting
- **Boosters**: XGBoost (preferred) or LightGBM with early stopping
- **Validation**: 5-fold stratified cross-validation
- **Calibration**: Platt scaling for probability calibration

## 🔧 Configuration

Key parameters are controlled via `configs/training_config.yaml`:

```yaml
feature_engineering:
  redundancy_threshold: 0.97
  fill_rate_threshold: 0.85

cross_validation:
  n_splits: 5
  random_seeds: [42, 202, 404, 808, 1337]

models:
  xgboost:
    learning_rate: 0.02
    max_depth: 6
    early_stopping_rounds: 300
```

## 📈 Model Performance

| Model | CV AUC | CV AP | Stability |
|-------|--------|-------|-----------|
| Logistic | ~0.85 | ~0.86 | High |
| GBDT | ~0.86 | ~0.87 | High |
| XGBoost | ~0.87 | ~0.88 | High |

## 🛡️ Quality Assurance

- **Leakage Detection**: Automated scanning for outcome-related features
- **Guard Lists**: Explicit exclusion of problematic variables
- **Redundancy Control**: Correlation-based feature pruning
- **Null Handling**: Zero null/infinite values after preprocessing
- **Stability Analysis**: Cross-seed feature importance consistency

## 📚 Documentation

- [`docs/PIPELINE.md`](docs/PIPELINE.md) - Detailed pipeline documentation
- [`configs/training_config.yaml`](configs/training_config.yaml) - Training parameters
- Model cards and reports in `docs/` directory

## 🤝 Contributing

1. Follow the existing code structure
2. Update documentation for any changes
3. Run both baseline and booster training for validation
4. Ensure all outputs are generated in `artifacts/`

## 📄 License

[Add your license information here]

## 📞 Contact

[Add contact information here]
