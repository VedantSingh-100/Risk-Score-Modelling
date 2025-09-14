# === MODEL STARTER WITH REDUNDANCY PRUNE + PCS QUICK CHECK ===
import os, json, time, math, gc, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

# ------------------------------------------------------------
# I/O
# ------------------------------------------------------------
print("=== Model Training Pipeline with Progress Tracking ===")

# Setup progress tracking for main phases
main_phases = [
    "Loading data and preprocessing",
    "Building redundancy graph",
    "Pruning redundant features", 
    "Training Logistic Regression",
    "Training Gradient Boosting",
    "Computing feature importances",
    "Saving final outputs"
]

main_progress = tqdm(main_phases, desc="Overall Progress")

DATA_DIR = Path("data")
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Load features & target
print("Loading feature matrix and labels...")
X = pd.read_parquet(DATA_DIR / "X_features.parquet")
y_df = pd.read_csv(DATA_DIR / "y_label.csv")
print(f"Loaded {X.shape[0]:,} samples with {X.shape[1]} features")

# Choose label column (prefer label_union, else first boolean/int column)
label_candidates = [c for c in ["label_union","label_weighted","label_hierarchical","label_clustered"] if c in y_df.columns]
label_col = label_candidates[0] if label_candidates else y_df.columns[-1]
y = y_df[label_col].astype(int).values
print(f"Using label column: {label_col} (prevalence: {y.mean():.4f})")

# ------------------------------------------------------------
# Load redundancy pairs + 1D metrics (be robust to file name typos)
# ------------------------------------------------------------
print("Loading QC files...")
pairs_path = None
for cand in ["qc_redundancy_pairs.csv","qc_redundancyt_pairs.csv"]:
    p = DATA_DIR / cand
    if p.exists():
        pairs_path = p; break
if pairs_path is None:
    raise FileNotFoundError("qc_redundancy_pairs.csv not found (also tried qc_redundancyt_pairs.csv).")

metrics_path = None
for cand in ["qc_single_feature_matrix.csv","qc_single_feature_metrics.csv"]:
    p = DATA_DIR / cand
    if p.exists():
        metrics_path = p; break
if metrics_path is None:
    raise FileNotFoundError("qc_single_feature_matrix.csv not found (also tried qc_single_feature_metrics.csv).")

pairs = pd.read_csv(pairs_path)  # columns: feature_i, feature_j, pearson_r
metrics = pd.read_csv(metrics_path)  # columns: feature, auc_1d, ...
print(f"Loaded {len(pairs)} redundancy pairs and {len(metrics)} feature metrics")

auc_map = dict(zip(metrics["feature"], metrics["auc_1d"]))

# ------------------------------------------------------------
# Guard set
# ------------------------------------------------------------
print("Building guard set...")
guard = set()
# from guard_Set.txt (optional)
for cand in ["guard_Set.txt","guard_set.txt","do_not_use_features.txt"]:
    p = DATA_DIR / cand
    if p.exists():
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if ln:
                guard.add(ln)

# from dropped_features.json (identifiers, pre-removed columns)
drop_meta_path = DATA_DIR / "dropped_features.json"
if drop_meta_path.exists():
    jf = json.loads(drop_meta_path.read_text())
    for k in ["guard","identifier","outcome_guard","not_in_raw","fill_rate_lt_threshold"]:
        for v in jf.get(k, []):
            guard.add(v)

# If label sources list exists, add as guard (optional)
sel_lab_path = DATA_DIR / "Selected_label_sources.csv"
if sel_lab_path.exists():
    sel_df = pd.read_csv(sel_lab_path)
    for v in sel_df["variable"].astype(str):
        guard.add(v)

# Assert: no guard features present
guard_in_X = sorted(set(X.columns) & guard)
if guard_in_X:
    # We will drop them defensively
    X = X.drop(columns=guard_in_X)
    pd.Series(guard_in_X, name="dropped_guard_feature").to_csv(OUT_DIR/"qc_guard_removed.csv", index=False)
    print(f"Defensively dropped {len(guard_in_X)} guard features")

print(f"Guard set size: {len(guard)}, Features after guard: {X.shape[1]}")
main_progress.update(1)

# ------------------------------------------------------------
# Redundancy prune: build connected components and keep the best auc_1d in each
# ------------------------------------------------------------
print("Building redundancy graph...")
# Build adjacency
adj = {}
def add_edge(a,b):
    adj.setdefault(a,set()).add(b); adj.setdefault(b,set()).add(a)

print("Processing redundancy pairs...")
for _,r in tqdm(pairs.iterrows(), desc="Building graph", total=len(pairs)):
    a, b = r["feature_i"], r["feature_j"]
    if a in X.columns and b in X.columns:
        add_edge(a,b)

print("Finding connected components...")
visited, comps = set(), []
nodes = list(adj.keys())
for node in tqdm(nodes, desc="Finding components"):
    if node in visited: continue
    # BFS
    q, comp = [node], set([node]); visited.add(node)
    while q:
        u = q.pop()
        for v in adj.get(u, []):
            if v not in visited:
                visited.add(v); comp.add(v); q.append(v)
    comps.append(sorted(comp))

print(f"Found {len(comps)} connected components")
main_progress.update(1)

# Decide keep/drop
print("Selecting best features from each component...")
keep, drop = set(), set()
for comp in tqdm(comps, desc="Pruning features"):
    # choose feature with highest auc_1d; fallback to column present in X
    scored = [(f, auc_map.get(f, 0.0)) for f in comp if f in X.columns]
    if not scored: continue
    best = max(scored, key=lambda t: t[1])[0]
    keep.add(best)
    for f,_ in scored:
        if f != best:
            drop.add(f)

# Also check pair endpoints that are singletons not in components (no action)
drop_sorted = sorted(drop)
keep_sorted = sorted(keep)
pd.DataFrame({"dropped_feature": drop_sorted}).to_csv(OUT_DIR / "redundancy_drops.csv", index=False)

print(f"Redundancy pruning: keeping {len(keep_sorted)}, dropping {len(drop_sorted)} features")

# Apply drops
X_model = X.drop(columns=drop_sorted) if drop_sorted else X.copy()
X_model.to_parquet(OUT_DIR / "X_features_model.parquet")
print(f"Final feature matrix: {X_model.shape[0]:,} samples × {X_model.shape[1]} features")
main_progress.update(1)

# ------------------------------------------------------------
# Train/evaluate: Logistic(ElasticNet) + GradientBoosting with 5-fold CV
# ------------------------------------------------------------
def cv_eval_model(clf_name, make_clf, X_df, y_vec, n_splits=5, seeds=[42,202,404,808,1337]):
    print(f"Training {clf_name} with {len(seeds)} seeds × {n_splits}-fold CV...")
    rows = []
    topk_sets = []
    
    for seed in tqdm(seeds, desc=f"{clf_name} seeds"):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof_pred = np.zeros(len(y_vec), dtype=float)
        fit_times, pred_times = [], []

        # Refit per fold to collect OOF
        fold_splits = list(skf.split(X_df, y_vec))
        for fold, (tr_idx, va_idx) in enumerate(tqdm(fold_splits, desc=f"Fold {seed}", leave=False)):
            X_tr, X_va = X_df.iloc[tr_idx], X_df.iloc[va_idx]
            y_tr, y_va = y_vec[tr_idx], y_vec[va_idx]

            clf = make_clf(random_state=seed)
            t0 = time.time()
            clf.fit(X_tr, y_tr)
            fit_times.append(time.time()-t0)

            t1 = time.time()
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(X_va)[:,1]
            else:
                # decision_function fallback
                s = clf.decision_function(X_va)
                p = (s - s.min())/(s.max()-s.min() + 1e-12)
            pred_times.append(time.time()-t1)
            oof_pred[va_idx] = p

        auc = roc_auc_score(y_vec, oof_pred)
        ap  = average_precision_score(y_vec, oof_pred)

        # Fit once on full data to extract "top-20" features for stability
        clf_full = make_clf(random_state=seed)
        clf_full.fit(X_df, y_vec)

        if clf_name == "logit_en":
            coefs = np.abs(clf_full.named_steps["clf"].coef_[0])
            top_idx = np.argsort(coefs)[::-1][:20]
            top_names = X_df.columns[top_idx]
        else:
            imp = clf_full.named_steps["clf"].feature_importances_
            top_idx = np.argsort(imp)[::-1][:20]
            top_names = X_df.columns[top_idx]

        topk_sets.append(set(top_names))
        rows.append({
            "model": clf_name, "seed": seed, "auc": auc, "ap": ap,
            "fit_time_s": float(np.sum(fit_times)), "pred_time_s": float(np.sum(pred_times))
        })

    # Stability: average pairwise Jaccard across seeds
    jacc = []
    for a, b in itertools.combinations(topk_sets, 2):
        inter = len(a & b); union = len(a | b)
        jacc.append(inter/union if union else 1.0)
    stab = {
        "model": clf_name,
        "mean_auc": float(np.mean([r["auc"] for r in rows])),
        "std_auc":  float(np.std([r["auc"] for r in rows], ddof=1)) if len(rows)>1 else 0.0,
        "mean_ap":  float(np.mean([r["ap"] for r in rows])),
        "std_ap":   float(np.std([r["ap"] for r in rows], ddof=1)) if len(rows)>1 else 0.0,
        "mean_jaccard_top20": float(np.mean(jacc)) if jacc else 1.0
    }
    
    print(f"{clf_name} results: AUC = {stab['mean_auc']:.4f} ± {stab['std_auc']:.4f}, "
          f"AP = {stab['mean_ap']:.4f} ± {stab['std_ap']:.4f}")
    
    return pd.DataFrame(rows), stab, topk_sets

from sklearn.pipeline import Pipeline

def make_logit(random_state=42):
    return Pipeline([
        ("scaler", RobustScaler(with_centering=True)),
        ("clf", LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            max_iter=5000, random_state=random_state, n_jobs=None
        ))
    ])

def make_gb(random_state=42):
    return Pipeline([
        ("scaler", "passthrough"),
        ("clf", GradientBoostingClassifier(
            random_state=random_state, n_estimators=300, learning_rate=0.05, max_depth=3
        ))
    ])

cv_logit, stab_logit, logit_topsets = cv_eval_model("logit_en", make_logit, X_model, y)
main_progress.update(1)

cv_gb,    stab_gb,    gb_topsets    = cv_eval_model("gbdt",    make_gb,    X_model, y)
main_progress.update(1)

# Save eval rows
eval_df = pd.concat([cv_logit, cv_gb], ignore_index=True)
eval_df.to_csv(OUT_DIR / "model_eval_summary.csv", index=False)

# Save stability summary
pcs_summary = {"logit_en": stab_logit, "gbdt": stab_gb,
               "n_features_after_prune": int(X_model.shape[1]),
               "n_features_before_prune": int(X.shape[1]),
               "label": label_col}
(Path(OUT_DIR / "pcs_stability_summary.json")).write_text(json.dumps(pcs_summary, indent=2))

print("Computing final feature importances...")
# Save feature importances from a final fit (on full data with default seed)
final_logit = make_logit(random_state=42).fit(X_model, y)
final_gb    = make_gb(random_state=42).fit(X_model, y)

coef_abs = np.abs(final_logit.named_steps["clf"].coef_[0])
fi_lr = pd.DataFrame({"feature": X_model.columns, "importance": coef_abs}).sort_values("importance", ascending=False)
fi_lr.to_csv(OUT_DIR / "feature_importance_lr.csv", index=False)

fi_gb = pd.DataFrame({"feature": X_model.columns, "importance": final_gb.named_steps["clf"].feature_importances_}).sort_values("importance", ascending=False)
fi_gb.to_csv(OUT_DIR / "feature_importance_gb.csv", index=False)

# Keep a human-friendly drop plan
pd.DataFrame({
    "kept_feature": sorted(X_model.columns)
}).to_csv(OUT_DIR / "kept_features.csv", index=False)

main_progress.update(1)
main_progress.update(1)  # Complete final phase
main_progress.close()

print("\n=== Training Complete ===")
print(f"Final model features: {X_model.shape[1]} (dropped {len(drop_sorted)} redundant)")
print(f"Logistic Regression: AUC = {stab_logit['mean_auc']:.4f} ± {stab_logit['std_auc']:.4f}")
print(f"Gradient Boosting:   AUC = {stab_gb['mean_auc']:.4f} ± {stab_gb['std_auc']:.4f}")
print(f"Results saved to: {OUT_DIR}")
