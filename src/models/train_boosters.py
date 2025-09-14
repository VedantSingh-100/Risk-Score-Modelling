#!/usr/bin/env python
"""
Booster Model Training Script
Trains XGBoost, LightGBM, or HistGradientBoosting with cross-validation and calibration.
"""
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression

def load_y(y_path):
    """Load target variable, preferring label_union if available."""
    ydf = pd.read_csv(y_path)
    for c in ("label_union","label_weighted","label_hierarchical","label_clustered"):
        if c in ydf.columns: 
            return ydf[c].astype(int).values, c
    cols = [c for c in ydf.columns if c.lower() not in {"id","row_id","user_id"}]
    return ydf[cols[-1]].astype(int).values, cols[-1]

def train_lgb_cv(X, y, n_splits=5, seed=42):
    """Train LightGBM with cross-validation."""
    import lightgbm as lgb
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    feats = X.columns.tolist()
    fi = np.zeros(len(feats))
    rows = []
    
    print(f"Training LightGBM with {n_splits}-fold cross-validation...")
    for fold,(tr,va) in enumerate(tqdm(cv.split(X,y), total=n_splits, desc="LGB CV folds"), 1):
        model = lgb.LGBMClassifier(
            objective="binary", 
            n_estimators=5000, 
            learning_rate=0.02,
            num_leaves=31, 
            subsample=0.8, 
            colsample_bytree=0.8,
            reg_alpha=0.1, 
            reg_lambda=0.2, 
            random_state=seed, 
            n_jobs=-1
        )
        model.fit(
            X.iloc[tr], y[tr], 
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc", 
            callbacks=[lgb.early_stopping(300, verbose=False)]
        )
        p = model.predict_proba(X.iloc[va])[:,1]
        oof[va] = p
        fi += model.feature_importances_
        
        auc = roc_auc_score(y[va], p)
        ap = average_precision_score(y[va], p)
        iters = getattr(model, "best_iteration_", model.n_estimators)
        rows.append({"fold":fold,"auc":auc,"ap":ap,"iters":iters})
        print(f"[LGB] fold {fold}: AUC={auc:.4f}, AP={ap:.4f}, iters={iters}")
    
    fi /= n_splits
    fi_df = pd.DataFrame({"feature":feats,"importance":fi}).sort_values("importance",ascending=False)
    fold_df = pd.DataFrame(rows)
    
    return "lgb", oof, fi_df, fold_df

def train_xgb_cv(X, y, n_splits=5, seed=42):
    """Train XGBoost with cross-validation."""
    import xgboost as xgb
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    feats = X.columns.tolist()
    fi = np.zeros(len(feats))
    rows = []
    
    print(f"Training XGBoost with {n_splits}-fold cross-validation...")
    for fold,(tr,va) in enumerate(tqdm(cv.split(X,y), total=n_splits, desc="XGB CV folds"), 1):
        model = xgb.XGBClassifier(
            objective="binary:logistic", 
            n_estimators=5000,
            learning_rate=0.02, 
            max_depth=6, 
            subsample=0.8,
            colsample_bytree=0.8, 
            reg_alpha=0.1, 
            reg_lambda=0.2,
            tree_method="hist", 
            eval_metric="auc", 
            random_state=seed,
            n_jobs=-1, 
            early_stopping_rounds=300
        )
        model.fit(
            X.iloc[tr], y[tr], 
            eval_set=[(X.iloc[va], y[va])], 
            verbose=False
        )
        p = model.predict_proba(X.iloc[va])[:,1]
        oof[va] = p
        
        try: 
            fi += model.feature_importances_
        except: 
            pass
        
        # Handle different XGBoost versions
        best = getattr(model, "best_iteration", getattr(model,"best_ntree_limit", model.n_estimators))
        auc = roc_auc_score(y[va], p)
        ap = average_precision_score(y[va], p)
        rows.append({"fold":fold,"auc":auc,"ap":ap,"iters":best})
        print(f"[XGB] fold {fold}: AUC={auc:.4f}, AP={ap:.4f}, iters={best}")
    
    fi_df = pd.DataFrame({"feature":feats,"importance":fi}).sort_values("importance",ascending=False)
    fold_df = pd.DataFrame(rows)
    
    return "xgb", oof, fi_df, fold_df

def train_hgb_cv(X, y, n_splits=5, seed=42):
    """Train HistGradientBoosting with cross-validation."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    rows = []
    
    print(f"Training HistGradientBoosting with {n_splits}-fold cross-validation...")
    for fold,(tr,va) in enumerate(tqdm(cv.split(X,y), total=n_splits, desc="HGB CV folds"), 1):
        model = HistGradientBoostingClassifier(
            max_depth=None, 
            max_iter=800,
            learning_rate=0.05, 
            l2_regularization=1.0,
            random_state=seed
        )
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:,1]
        oof[va] = p
        
        auc = roc_auc_score(y[va], p)
        ap = average_precision_score(y[va], p)
        rows.append({"fold":fold,"auc":auc,"ap":ap})
        print(f"[HGB] fold {fold}: AUC={auc:.4f}, AP={ap:.4f}")
    
    fold_df = pd.DataFrame(rows)
    return "hgb", oof, None, fold_df

def main(args):
    print("Loading data...")
    X = pd.read_parquet(Path(args.data_dir)/"X_features.parquet")
    y, label_col = load_y(Path(args.data_dir)/"y_label.csv")
    
    print(f"Loaded X: {X.shape}, y: {len(y)} (prevalence: {y.mean():.4f})")
    print(f"Target column: {label_col}")

    # Optional redundancy drop by correlation
    if args.redundancy_r > 0:
        print(f"Applying correlation-based redundancy pruning (r >= {args.redundancy_r})...")
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if (upper[c] >= args.redundancy_r).any()]
        if to_drop:
            print(f"Dropping {len(to_drop)} redundant features")
            X = X.drop(columns=to_drop)
        else:
            print("No redundant features found")

    print(f"Final feature set: {X.shape[1]} features")

    # Determine which library to use
    print("Determining available gradient boosting library...")
    try:
        import lightgbm as _
        print("✓ LightGBM available - using LightGBM")
        kind, oof, fi_df, folds = train_lgb_cv(X, y, args.folds, args.seed)
    except ImportError:
        try:
            import xgboost as _
            print("✓ XGBoost available - using XGBoost")
            kind, oof, fi_df, folds = train_xgb_cv(X, y, args.folds, args.seed)
        except ImportError:
            print("✓ Using sklearn HistGradientBoosting (fallback)")
            kind, oof, fi_df, folds = train_hgb_cv(X, y, args.folds, args.seed)

    # Create output directory
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    # Calibrate using Platt scaling
    print("Applying Platt calibration...")
    cal = LogisticRegression(max_iter=1000).fit(oof.reshape(-1,1), y)
    oof_cal = cal.predict_proba(oof.reshape(-1,1))[:,1]

    # Calculate metrics
    auc_cv, ap_cv = folds["auc"].mean(), folds["ap"].mean()
    auc_oof_raw = roc_auc_score(y, oof)
    ap_oof_raw = average_precision_score(y, oof)
    auc_oof_cal = roc_auc_score(y, oof_cal)
    ap_oof_cal = average_precision_score(y, oof_cal)
    
    # Create report
    report = {
        "model_kind": kind,
        "n_rows": int(X.shape[0]), 
        "n_features": int(X.shape[1]),
        "auc_cv_mean": float(auc_cv), 
        "auc_cv_std": float(folds["auc"].std()),
        "ap_cv_mean":  float(ap_cv),  
        "ap_cv_std":  float(folds["ap"].std()),
        "auc_oof_raw": float(auc_oof_raw),
        "ap_oof_raw":  float(ap_oof_raw),
        "auc_oof_cal": float(auc_oof_cal),
        "ap_oof_cal":  float(ap_oof_cal),
        "label": label_col
    }
    
    # Save outputs
    print("Saving results...")
    pd.DataFrame({"oof_pred":oof,"oof_pred_cal":oof_cal,"y":y}).to_csv(out/"oof_predictions.csv", index=False)
    folds.to_csv(out/"cv_fold_metrics.csv", index=False)
    if fi_df is not None: 
        fi_df.to_csv(out/"feature_importance_cv.csv", index=False)
    (out/"model_cv_report.json").write_text(json.dumps(report, indent=2))
    
    print("\n=== Results ===")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train gradient boosting models with cross-validation")
    ap.add_argument("--data-dir", default="data/processed", help="Directory containing processed data")
    ap.add_argument("--out-dir",  default="artifacts/reports/boosters", help="Output directory")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--redundancy-r", type=float, default=0.0, help="Correlation threshold for redundancy pruning (e.g. 0.97)")
    main(ap.parse_args())
