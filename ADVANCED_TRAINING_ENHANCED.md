# Advanced Model Training Pipeline - Enhanced with Progress Tracking

## 🚀 Overview

The `train_advanced.py` script provides a sophisticated machine learning pipeline with automatic gradient boosting library detection, comprehensive progress tracking, and advanced model calibration. This enhanced version includes detailed progress bars and an optimized HPC submission workflow.

## ✨ Key Features

### 🔄 **Automatic Library Detection**
- **Primary**: LightGBM (fastest, most memory efficient)
- **Fallback 1**: XGBoost (robust, widely supported)  
- **Fallback 2**: sklearn HistGradientBoosting (always available)

### 📊 **Comprehensive Progress Tracking**
- **Overall pipeline progress**: 7 main phases tracked
- **Feature correlation analysis**: Progress bar for redundancy detection
- **Guard enforcement**: Progress tracking for feature scanning
- **Cross-validation**: Fold-by-fold progress for each model type
- **Real-time updates**: Live feedback on training progress

### 🛡️ **Advanced Data Protection**
- **Correlation-based redundancy removal**: Removes features with |r| ≥ 0.97
- **Multi-source guard file detection**: Flexible guard set loading
- **Automatic leakage detection**: Scans for outcome-related feature names
- **Robust file path handling**: Multiple fallback locations

### 🎯 **Model Calibration**
- **Platt scaling**: Calibrates predictions for better probability estimates
- **Out-of-fold predictions**: Prevents overfitting during calibration
- **Dual metrics**: Raw and calibrated performance comparison

## 📁 File Structure

```
/home/vhsingh/Parshvi_project/
├── train_advanced.py                    # Enhanced training script
├── submit_advanced_train_job.sh         # HPC submission script
├── ADVANCED_TRAINING_ENHANCED.md        # This documentation
└── logs/                               # Job logs (auto-created)
    ├── advanced_train_[job_id].out     # Standard output
    └── advanced_train_[job_id].err     # Error output
```

## 🔧 **Input Requirements**

### **Required Files**
The script automatically searches for files in multiple locations:

#### **Feature Matrix** (one of):
- `/data/X_features.parquet`
- `/home/vhsingh/Parshvi_project/data/X_features.parquet`
- `/home/vhsingh/Parshvi_project/engineered/X_features.parquet`
- `/home/vhsingh/deterministic_fe_outputs/X_features.parquet`

#### **Labels** (one of):
- `/data/y_label.csv`
- `/home/vhsingh/Parshvi_project/data/y_label.csv`
- `/home/vhsingh/Parshvi_project/engineered/y_label.csv`
- `/home/vhsingh/deterministic_fe_outputs/y_label.csv`

#### **Guard Set** (optional, one of):
- `/data/guard_Set.txt` or `/data/guard_set.txt`
- `/home/vhsingh/Parshvi_project/data/guard_Set.txt`
- `/home/vhsingh/Parshvi_project/engineered/guard_set.txt`

### **Data Format Requirements**
- **Features**: Parquet format with numeric columns
- **Labels**: CSV with target column (non-ID columns auto-detected)
- **Guard set**: Text file with one feature name per line

## 🖥️ **HPC Submission**

### **Resource Allocation**
```bash
#SBATCH --cpus-per-task=24      # 24 CPU cores
#SBATCH --mem=96GB              # 96GB RAM
#SBATCH --time=06:00:00         # 6 hours max
#SBATCH --partition=gpu         # GPU partition (for faster compute)
```

### **Submit Job**
```bash
cd /home/vhsingh/Parshvi_project
sbatch submit_advanced_train_job.sh
```

### **Monitor Progress**
```bash
# Check job status
squeue -u $USER

# View live output
tail -f logs/advanced_train_[job_id].out

# Check for errors
tail -f logs/advanced_train_[job_id].err
```

## 📊 **Output Files**

All outputs are saved to `/model_outputs/`:

### **Performance Metrics**
- `model_cv_report.json` - Comprehensive performance summary
- `cv_fold_metrics.csv` - Fold-by-fold detailed metrics
- `oof_predictions.csv` - Out-of-fold predictions (raw + calibrated)

### **Feature Analysis**
- `feature_importance_cv.csv` - Cross-validated feature importance rankings

### **Example Output Structure**
```
/model_outputs/
├── model_cv_report.json         # Main performance report
├── cv_fold_metrics.csv          # Detailed fold metrics
├── oof_predictions.csv          # Prediction probabilities
└── feature_importance_cv.csv    # Feature rankings
```

## 📈 **Performance Interpretation**

### **Key Metrics Explained**

#### **AUC (Area Under ROC Curve)**
- **Range**: 0.0 - 1.0
- **Interpretation**: 
  - 0.5 = Random performance
  - 0.7 = Good performance
  - 0.8 = Very good performance
  - 0.9+ = Excellent performance

#### **AP (Average Precision)**
- **Range**: 0.0 - 1.0  
- **Focus**: Performance on positive class
- **Better for**: Imbalanced datasets

#### **Standard Deviation**
- **Low values** (< 0.01): Very stable model
- **High values** (> 0.05): Unstable, may need more data

### **Sample Performance Report**
```json
{
  "model_kind": "lgb",
  "n_rows": 49389,
  "n_features": 127,
  "auc_cv_mean": 0.8756,
  "auc_cv_std": 0.0023,
  "ap_cv_mean": 0.8912,
  "ap_cv_std": 0.0019,
  "auc_oof_raw": 0.8751,
  "auc_oof_cal": 0.8753
}
```

## 🔍 **Progress Tracking Details**

### **Main Pipeline Phases**
1. **Loading data and preprocessing** - File I/O and initial setup
2. **Feature redundancy analysis** - Correlation-based feature removal  
3. **Guard set enforcement** - Leakage prevention
4. **Model training with cross-validation** - Core ML training
5. **Model calibration** - Platt scaling for better probabilities
6. **Computing final metrics** - Performance evaluation
7. **Saving outputs and artifacts** - File writing

### **Detailed Progress Bars**
- **Correlation checking**: Feature-by-feature redundancy analysis
- **Guard scanning**: Term-by-term leakage detection  
- **CV folds**: Fold-by-fold training progress
- **Library detection**: Automatic fallback progression

## ⚡ **Performance Optimizations**

### **Computational Efficiency**
- **Multi-threading**: Uses all allocated CPU cores
- **Memory optimization**: Efficient data structures
- **Early stopping**: Prevents overfitting and saves time
- **Vectorized operations**: NumPy/pandas optimizations

### **HPC Optimizations**
- **Environment variables**: Optimal thread allocation
- **Module loading**: Proper library initialization
- **Resource monitoring**: Memory and CPU usage tracking

## 🐛 **Troubleshooting**

### **Common Issues**

#### **Library Installation Failures**
```bash
# Manual installation
pip install --user lightgbm xgboost
```

#### **Memory Issues**
- Reduce `--mem` allocation if job fails
- Consider feature selection to reduce dimensionality

#### **File Not Found Errors**
- Check file paths in the validation section
- Ensure proper file permissions
- Verify data format (Parquet vs CSV)

#### **Performance Issues**
- Check for data leakage in features
- Verify target variable distribution
- Consider feature engineering

### **Debug Mode**
Add debug prints to the script:
```python
print(f"Debug: X shape = {X.shape}")
print(f"Debug: y distribution = {np.bincount(y)}")
```

## 🎯 **Best Practices**

### **Data Preparation**
1. **Clean data**: Remove missing values and outliers
2. **Feature engineering**: Create meaningful derived features
3. **Guard sets**: Maintain comprehensive leakage prevention lists
4. **Validation**: Always validate data quality before training

### **Model Training**
1. **Cross-validation**: Use stratified folds for imbalanced data
2. **Calibration**: Always calibrate for probability-based decisions
3. **Feature importance**: Review and validate important features
4. **Stability**: Check performance consistency across folds

### **Production Deployment**
1. **Model versioning**: Track model versions and performance
2. **Monitoring**: Set up performance monitoring in production
3. **Retraining**: Establish regular retraining schedules
4. **A/B testing**: Compare new models against baselines

## 📞 **Support**

For issues or questions:
1. Check the error logs in `/home/vhsingh/Parshvi_project/logs/`
2. Review the troubleshooting section above
3. Validate input data format and paths
4. Consider resource allocation adjustments

---

**Enhanced Advanced Training Pipeline** - Providing production-ready machine learning with comprehensive progress tracking and robust error handling.


