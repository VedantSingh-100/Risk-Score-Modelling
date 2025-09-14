import os, json, warnings, math
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)

# ---------- I/O ----------
print("=== Advanced Model Training Pipeline with Progress Tracking ===")

# Setup progress tracking for main phases
main_phases = [
    "Loading data and preprocessing",
    "Feature redundancy analysis",
    "Guard set enforcement", 
    "Model training with cross-validation",
    "Model calibration",
    "Computing final metrics",
    "Saving outputs and artifacts"
]

main_progress = tqdm(main_phases, desc="Overall Progress")

X_path = "/home/vhsingh/Parshvi_project/data/X_features.parquet"
y_path = "/home/vhsingh/Parshvi_project/data/y_label.csv"
guard_path_candidates = ["/data/guard_Set.txt", "/data/guard_set.txt", "/data/guard_Set.txt"]
out_dir = Path("model_outputs"); out_dir.mkdir(parents=True, exist_ok=True)

print("Loading feature matrix and labels...")
X = pd.read_parquet(X_path)
y_df = pd.read_csv(y_path)
# pick the first non-id column as target (your y file contains label_union)
target_col = [c for c in y_df.columns if c.lower() not in {"id", "row_id", "user_id"}][0]
y = y_df[target_col].astype(int).values

print(f"Loaded X: {X.shape}, y positives: {int(y.sum())} / {len(y)} (prevalence={y.mean():.4f})")
main_progress.update(1)

# ---------- OPTIONAL: drop redundancy by correlation ----------
def drop_redundant_features(X, thr=0.97):
    print(f"Computing correlation matrix for {X.shape[1]} features...")
    # pairwise correlation on numeric (all)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    
    print("Identifying redundant features...")
    to_drop = []
    for column in tqdm(upper.columns, desc="Checking correlations"):
        if any(upper[column] >= thr):
            to_drop.append(column)
    
    kept = [c for c in X.columns if c not in to_drop]
    return X[kept], kept, to_drop

APPLY_REDUNDANCY_PRUNE = True
if APPLY_REDUNDANCY_PRUNE:
    X, kept, dropped = drop_redundant_features(X, thr=0.97)
    print(f"Redundancy prune: dropped {len(dropped)} features (|r|>=0.97) → X now {X.shape[1]} features")

main_progress.update(1)

# ---------- Guard enforcement ----------
print("Building guard set...")
guard_set = set()
for p in guard_path_candidates:
    if Path(p).exists():
        guard_set |= {ln.strip() for ln in Path(p).read_text().splitlines() if ln.strip()}

# also add basic outcome-like patterns as a safety net
print("Checking for outcome-related features...")
guard_terms = ["default", "dpd", "overdue", "arrear", "write", "chargeoff", "npa", "settle",
               "miss", "min_due", "overlimit", "declin", "reject", "bounced", "nsf", "negative", "flag"]
auto_guard = set()
for c in tqdm(X.columns, desc="Scanning for guard terms"):
    if any(t in c.lower() for t in guard_terms):
        auto_guard.add(c)
guard_set |= auto_guard

guard_hits = [c for c in X.columns if c in guard_set]
if guard_hits:
    print(f"[GUARD] Removing {len(guard_hits)} guarded features: {guard_hits[:5]}{'...' if len(guard_hits)>5 else ''}")
    X = X.drop(columns=guard_hits)

print(f"Final feature set: {X.shape[1]} features after guard enforcement")
main_progress.update(1)

# ---------- Models ----------
def train_lgb_cv(X, y, n_splits=5, seed=SEED):
    import lightgbm as lgb
    
    print(f"Training LightGBM with {n_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    feats = X.columns.tolist()
    fi = np.zeros(len(feats))
    metrics = []

    fold_splits = list(cv.split(X, y))
    for fold, (tr, va) in enumerate(tqdm(fold_splits, desc="LGB CV folds"), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=5000,
            learning_rate=0.02,
            num_leaves=31,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            random_state=seed,
            n_jobs=-1
        )
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )
        p = model.predict_proba(Xva)[:, 1]
        oof[va] = p
        fi += model.feature_importances_
        auc = roc_auc_score(yva, p)
        ap  = average_precision_score(yva, p)
        metrics.append({"fold": fold, "auc": auc, "ap": ap, "iters": model.best_iteration_})
        print(f"[LGB] fold {fold}: AUC={auc:.4f}, AP={ap:.4f}, iters={model.best_iteration_}")

    fi /= n_splits
    return oof, pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False), pd.DataFrame(metrics)

def train_xgb_cv(X, y, n_splits=5, seed=SEED):
    import xgboost as xgb
    
    print(f"Training XGBoost with {n_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    feats = X.columns.tolist()
    fi = np.zeros(len(feats))
    metrics = []

    fold_splits = list(cv.split(X, y))
    for fold, (tr, va) in enumerate(tqdm(fold_splits, desc="XGB CV folds"), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=5000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.2,
            tree_method="hist",  # fast
            eval_metric="auc",
            random_state=seed,
            n_jobs=-1,
            early_stopping_rounds=300
        )
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False
        )
        p = model.predict_proba(Xva)[:, 1]
        oof[va] = p
        try:
            fi += model.feature_importances_
        except Exception:
            pass
        auc = roc_auc_score(yva, p)
        ap  = average_precision_score(yva, p)
        # In XGBoost 2.0+, best_ntree_limit is replaced with best_iteration
        best_iter = getattr(model, 'best_iteration', getattr(model, 'best_ntree_limit', model.n_estimators))
        metrics.append({"fold": fold, "auc": auc, "ap": ap, "iters": best_iter})
        print(f"[XGB] fold {fold}: AUC={auc:.4f}, AP={ap:.4f}, iters={best_iter}")

    return oof, pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False), pd.DataFrame(metrics)

def train_hgb_cv(X, y, n_splits=5, seed=SEED):
    print(f"Training HistGradientBoosting with {n_splits}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    feats = X.columns.tolist()
    metrics = []

    fold_splits = list(cv.split(X, y))
    for fold, (tr, va) in enumerate(tqdm(fold_splits, desc="HGB CV folds"), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        model = HistGradientBoostingClassifier(
            max_depth=None, max_iter=800, learning_rate=0.05, l2_regularization=1.0,
            random_state=seed
        )
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xva)[:, 1]
        oof[va] = p
        auc = roc_auc_score(yva, p)
        ap  = average_precision_score(yva, p)
        metrics.append({"fold": fold, "auc": auc, "ap": ap})
        print(f"[HGB] fold {fold}: AUC={auc:.4f}, AP={ap:.4f}")
    return oof, None, pd.DataFrame(metrics)

# Try LightGBM → XGBoost → HGB fallback
print("Determining available gradient boosting library...")
model_kind = None
try:
    import lightgbm as _; model_kind = "lgb"
    print("✓ LightGBM available - using LightGBM")
except Exception:
    try:
        import xgboost as _; model_kind = "xgb"
        print("✓ XGBoost available - using XGBoost")
    except Exception:
        model_kind = "hgb"
        print("✓ Using sklearn HistGradientBoosting")

print(f"Starting {model_kind.upper()} cross-validation training...")
if model_kind == "lgb":
    oof, fi_df, fold_df = train_lgb_cv(X, y, n_splits=5, seed=SEED)
elif model_kind == "xgb":
    oof, fi_df, fold_df = train_xgb_cv(X, y, n_splits=5, seed=SEED)
else:
    oof, fi_df, fold_df = train_hgb_cv(X, y, n_splits=5, seed=SEED)

main_progress.update(1)

# ---------- Calibration (Platt) ----------
print("Performing Platt calibration on out-of-fold predictions...")
# We calibrate a simple LogisticRegression on OOF predictions vs. true labels.
clf_cal = LogisticRegression(max_iter=1000).fit(oof.reshape(-1,1), y)
oof_cal = clf_cal.predict_proba(oof.reshape(-1,1))[:,1]

main_progress.update(1)

# ---------- Metrics ----------
print("Computing final performance metrics...")
def summarize(scores):
    return float(np.mean(scores)), float(np.std(scores))

auc_cv_mean, auc_cv_std = summarize(fold_df["auc"].values)
ap_cv_mean,  ap_cv_std  = summarize(fold_df["ap"].values)

auc_oof  = roc_auc_score(y, oof)
ap_oof   = average_precision_score(y, oof)
auc_cal  = roc_auc_score(y, oof_cal)
ap_cal   = average_precision_score(y, oof_cal)

print(f"\n== CV Summary ({model_kind.upper()}) ==")
print(f"AUC (fold mean±sd): {auc_cv_mean:.4f} ± {auc_cv_std:.4f}")
print(f"AP  (fold mean±sd): {ap_cv_mean:.4f} ± {ap_cv_std:.4f}")
print(f"AUC (OOF raw): {auc_oof:.4f} | AUC (OOF calibrated): {auc_cal:.4f}")
print(f"AP  (OOF raw):  {ap_oof:.4f} | AP  (OOF calibrated):  {ap_cal:.4f}")

main_progress.update(1)

# ---------- Save artifacts ----------
print("Saving model artifacts and reports...")
pd.DataFrame({"oof_pred": oof, "oof_pred_cal": oof_cal, "y": y}).to_csv(out_dir/"oof_predictions.csv", index=False)
fold_df.to_csv(out_dir/"cv_fold_metrics.csv", index=False)
if fi_df is not None:
    fi_df.to_csv(out_dir/"feature_importance_cv.csv", index=False)

report = {
    "model_kind": model_kind,
    "n_rows": int(X.shape[0]),
    "n_features": int(X.shape[1]),
    "auc_cv_mean": auc_cv_mean, "auc_cv_std": auc_cv_std,
    "ap_cv_mean": ap_cv_mean,   "ap_cv_std": ap_cv_std,
    "auc_oof_raw": auc_oof, "ap_oof_raw": ap_oof,
    "auc_oof_cal": auc_cal, "ap_oof_cal": ap_cal,
}
json.dump(report, open(out_dir/"model_cv_report.json","w"), indent=2)

main_progress.update(1)
main_progress.close()

print(f"\n🎉 Training completed successfully!")
print(f"📁 Artifacts written to: {out_dir}")
print(f"🏆 Best model: {model_kind.upper()} with {auc_cv_mean:.4f} AUC")
