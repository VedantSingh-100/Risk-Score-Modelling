#!/usr/bin/env python
"""
Baseline Model Training Script
Trains Logistic Regression (ElasticNet) and sklearn Gradient Boosting with cross-validation.
Includes redundancy pruning and PCS stability analysis.
"""
import os, json, time, itertools, argparse
from pathlib import Path
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def load_y(y_path, prefer=("label_union","label_weighted","label_hierarchical","label_clustered")):
    """Load target variable, preferring label_union if available."""
    ydf = pd.read_csv(y_path)
    for c in prefer:
        if c in ydf.columns: 
            return ydf[c].astype(int).values, c
    # else last non-id column
    cols = [c for c in ydf.columns if c.lower() not in {"id","row_id","user_id"}]
    return ydf[cols[-1]].astype(int).values, cols[-1]

def build_guard(config_dir: Path):
    """Build guard set from configuration files."""
    guard = set()
    for cand in ("guard_Set.txt","guard_set.txt","do_not_use_features.txt"):
        p = config_dir / cand
        if p.exists():
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln: guard.add(ln)
    
    # also drop label sources if present
    sel = config_dir / "Selected_label_sources.csv"
    if sel.exists():
        try:
            sdf = pd.read_csv(sel)
            if "variable" in sdf.columns:
                guard |= set(sdf["variable"].astype(str))
        except Exception:
            pass
    
    # dropped_features.json (identifiers)
    drop_meta = config_dir.parent / "dropped_features.json"
    if drop_meta.exists():
        try:
            jf = json.loads(drop_meta.read_text())
            for k in ("guard","identifier","outcome_guard","not_in_raw","fill_rate_lt_threshold"):
                for v in jf.get(k, []): guard.add(v)
        except Exception:
            pass
    return guard

def load_redundancy_pairs(data_dir: Path):
    """Load redundancy pairs from QC analysis."""
    for name in ("qc_redundancy_pairs.csv","qc_redundancyt_pairs.csv"):
        p = data_dir / name
        if p.exists():
            df = pd.read_csv(p)
            if set(["feature_i","feature_j"]) <= set(df.columns):
                return df
    return None

def load_metrics(data_dir: Path):
    """Load single feature AUC metrics for redundancy resolution."""
    for name in ("qc_single_feature_metrics.csv","qc_single_feature_matrix.csv"):
        p = data_dir / name
        if p.exists():
            m = pd.read_csv(p)
            if "feature" in m.columns and "auc_1d" in m.columns:
                return dict(zip(m["feature"], m["auc_1d"]))
    return {}

def prune_by_pairs(X, pairs_df, auc_map):
    """Prune redundant features using graph-based approach."""
    # Graph of highly-correlated groups from QC
    adj = {}
    def add(a,b):
        adj.setdefault(a,set()).add(b)
        adj.setdefault(b,set()).add(a)
    
    for _,r in pairs_df.iterrows():
        a,b = r["feature_i"], r["feature_j"]
        if a in X.columns and b in X.columns: 
            add(a,b)
    
    keep, drop, seen = set(), set(), set()
    for node in adj:
        if node in seen: continue
        # BFS to find connected component
        q=[node]; comp=set([node]); seen.add(node)
        while q:
            u=q.pop()
            for v in adj.get(u, []):
                if v not in seen: 
                    seen.add(v); comp.add(v); q.append(v)
        
        # keep best auc_1d within component
        scored=[(f, auc_map.get(f,0.0)) for f in comp if f in X.columns]
        if not scored: continue
        best=max(scored, key=lambda t:t[1])[0]
        keep.add(best)
        for f,_ in scored:
            if f!=best: drop.add(f)
    
    return sorted(keep), sorted(drop)

def cv_eval(name, make_clf, X, y, n_splits=5, seeds=(42,202,404,808,1337)):
    """Cross-validation evaluation with PCS stability analysis."""
    rows, topk_sets = [], []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y), float)
        for tr,va in skf.split(X,y):
            clf = make_clf(seed)
            clf.fit(X.iloc[tr], y[tr])
            if hasattr(clf,"predict_proba"):
                p = clf.predict_proba(X.iloc[va])[:,1]
            else:
                p = clf.decision_function(X.iloc[va])
            oof[va] = p
        
        auc, ap = roc_auc_score(y,oof), average_precision_score(y,oof)
        
        # fit once full to extract top20
        model = make_clf(seed).fit(X,y)
        if name=="logit_en":
            coefs = np.abs(model.named_steps["clf"].coef_[0])
            idx=np.argsort(coefs)[::-1][:20]
        else:
            imp = model.named_steps["clf"].feature_importances_
            idx=np.argsort(imp)[::-1][:20]
        
        topk_sets.append(set(X.columns[idx]))
        rows.append({"model":name,"seed":seed,"auc":auc,"ap":ap})
    
    # stability (Jaccard similarity of top-20 features across seeds)
    jacc=[]
    for a,b in itertools.combinations(topk_sets,2):
        u=len(a|b)
        jacc.append(len(a&b)/u if u else 1.0)
    
    return pd.DataFrame(rows), float(np.mean(jacc))

def make_logit(seed):
    """Create logistic regression pipeline."""
    return Pipeline([
        ("scaler", RobustScaler(with_centering=True)),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga",
                                  l1_ratio=0.5, C=1.0, max_iter=5000,
                                  random_state=seed))
    ])

def make_gbdt(seed):
    """Create gradient boosting pipeline."""
    return Pipeline([
        ("scaler","passthrough"),
        ("clf", GradientBoostingClassifier(random_state=seed,
                                           n_estimators=300, learning_rate=0.05, max_depth=3))
    ])

def main(args):
    data_dir   = Path(args.data_dir)
    config_dir = Path(args.config_dir)
    out_dir    = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    X = pd.read_parquet(data_dir/"X_features.parquet")
    y, label_col = load_y(data_dir/"y_label.csv")
    
    print(f"Loaded X: {X.shape}, y: {len(y)} (prevalence: {y.mean():.4f})")
    print(f"Target column: {label_col}")

    # Guard
    guard = build_guard(config_dir)
    guard_in_X = sorted(set(X.columns)&guard)
    if guard_in_X:
        print(f"Dropping {len(guard_in_X)} guarded features")
        pd.Series(guard_in_X, name="dropped_guard").to_csv(out_dir/"qc_guard_removed.csv", index=False)
        X = X.drop(columns=guard_in_X)

    # Redundancy prune using QC graph if present
    auc_map = load_metrics(data_dir)
    pairs   = load_redundancy_pairs(data_dir)
    drop = []
    if pairs is not None:
        print("Applying graph-based redundancy pruning...")
        keep, drop = prune_by_pairs(X, pairs, auc_map)
        if drop:
            print(f"Dropping {len(drop)} redundant features")
            X = X[keep]
            pd.DataFrame({"dropped_redundant": drop}).to_csv(out_dir/"redundancy_drops.csv", index=False)
    
    print(f"Final feature set: {X.shape[1]} features")

    # Cross-validation evaluation
    print("Running cross-validation...")
    cv_logit, jacc_l = cv_eval("logit_en", make_logit, X, y, n_splits=args.folds)
    cv_gbdt, jacc_g  = cv_eval("gbdt",     make_gbdt,  X, y, n_splits=args.folds)

    # Summaries
    eval_df = pd.concat([cv_logit, cv_gbdt], ignore_index=True)
    eval_df.to_csv(out_dir/"model_eval_summary.csv", index=False)
    
    pcs = {
        "label": label_col,
        "n_features_after_prune": int(X.shape[1]),
        "guard_dropped": guard_in_X,
        "redundancy_dropped": drop,
        "stability": {
            "logit_en_mean_jaccard_top20": jacc_l,
            "gbdt_mean_jaccard_top20": jacc_g
        },
        "mean_auc": {
            "logit_en": float(cv_logit["auc"].mean()),
            "gbdt": float(cv_gbdt["auc"].mean())
        },
        "mean_ap": {
            "logit_en": float(cv_logit["ap"].mean()),
            "gbdt": float(cv_gbdt["ap"].mean())
        }
    }
    
    (out_dir/"pcs_stability_summary.json").write_text(json.dumps(pcs, indent=2))
    
    print("\n=== Results ===")
    print(json.dumps(pcs, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train baseline models with stability analysis")
    ap.add_argument("--data-dir",   default="data/processed", help="Directory containing processed data")
    ap.add_argument("--config-dir", default="configs", help="Directory containing configuration files")
    ap.add_argument("--out-dir",    default="artifacts/reports/baselines", help="Output directory")
    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    main(ap.parse_args())
