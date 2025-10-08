# train_gdbt_sweep.py
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, roc_curve
from sklearn.linear_model import LogisticRegression

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available. Install with: pip install wandb")

def load_xy(data_root: Path, label_col: str | None):
    X = pd.read_parquet(data_root / "X_features.parquet")
    X = X.astype(np.float32) # memory + speed
    ydf = pd.read_csv(data_root / "y_label.csv")
    if label_col is None:
        label_col = "label" if "label" in ydf.columns else ydf.columns[0]
    y = ydf[label_col].astype(int).values
    if len(y) != len(X):
        raise ValueError(f"Row mismatch: X={len(X)} vs y={len(y)}")
    return X, y, label_col

def summarize_all(y, p, label="oof"):
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
    fpr, tpr, _ = roc_curve(y, p); ks = float(np.max(tpr - fpr))
    gini = 2*auc - 1; brier = np.mean((p - y)**2); ll = log_loss(y, p)
    return {"label": label, "auc": float(auc), "ap": float(ap), "ks": ks, "gini": gini, "brier": brier, "logloss": float(ll)}

def decile_table(y, p):
    n = len(y); order = np.argsort(-p); y = y[order]; p = p[order]
    bins = np.linspace(0, n, 11, dtype=int); rows = []
    for d in range(10):
        a, b = bins[d], bins[d+1]
        yy, pp = y[a:b], p[a:b]
        rows.append({"decile": d+1, "n": len(yy), "positives": int(yy.sum()),
                     "rate": float(yy.mean()), "score_min": float(pp.min()), "score_max": float(pp.max()), "score_mean": float(pp.mean())})
    return pd.DataFrame(rows)

def make_param_sampler_xgb(prev):
    spw = max((1.0-prev)/max(prev,1e-9), 1.0)
    rng = np.random.default_rng(42)
    def sample():
        return dict(
            eta=float(10**rng.uniform(-2.0, -0.7)),
            max_depth=int(rng.integers(3, 9)),
            min_child_weight=float(10**rng.uniform(-1, 2)),
            subsample=float(rng.uniform(0.6, 1.0)),
            colsample_bytree=float(rng.uniform(0.6, 1.0)),
            gamma=float(10**rng.uniform(-3, 1)),
            reg_alpha=float(10**rng.uniform(-4, 0)),
            reg_lambda=float(10**rng.uniform(-3, 1)),
            scale_pos_weight=float(spw),
            n_estimators=int(rng.integers(800, 2000)),
        )
    return sample

def cv_xgb(X, y, params, n_splits=5, seed=42, metric="auto"):
    import xgboost as xgb
    if metric == "auto": metric = "aucpr" if y.mean() <= 0.20 else "auc"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)
    feats = X.columns.tolist()
    fi = np.zeros(len(feats), dtype=np.float64)
    for tr, va in skf.split(X, y):
        model = xgb.XGBClassifier(
            objective="binary:logistic", tree_method="hist", random_state=seed,
            eval_metric=metric, early_stopping_rounds=200, **params
        )
        model.fit(X.iloc[tr], y[tr], eval_set=[(X.iloc[va], y[va])], verbose=False)
        p = model.predict_proba(X.iloc[va])[:,1]; oof[va] = p
        try: fi += model.feature_importances_
        except: pass
    fi /= n_splits
    return oof, pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False)

def cv_lgb(X, y, params, n_splits=5, seed=42, metric="auto"):
    import lightgbm as lgb
    if metric == "auto": metric = "average_precision" if y.mean() <= 0.20 else "auc"
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y), dtype=np.float32)
    feats = X.columns.tolist()
    fi = np.zeros(len(feats), dtype=np.float64)
    for tr, va in skf.split(X, y):
        model = lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=-1, **params)
        model.fit(X.iloc[tr], y[tr],
                  eval_set=[(X.iloc[va], y[va])], eval_metric=metric,
                  callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
        p = model.predict_proba(X.iloc[va])[:,1]; oof[va] = p
        try: fi += model.booster_.feature_importance(importance_type="gain")
        except: pass
    fi /= n_splits
    return oof, pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--label-column", default="label_clustered")
    ap.add_argument("--algo", choices=["xgb","lgb"], default="xgb")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-metric", choices=["auto","auc","aucpr"], default="auto")
    ap.add_argument("--splits-csv", default=None)
    ap.add_argument("--out-dir", default="model_outputs/gbdt_cv")
    args = ap.parse_args()

    run = None
    if WANDB_AVAILABLE:
        run = wandb.init(
            entity="ved100-carnegie-mellon-university",
            project="Risk_Score",
            name=f"gbdt-{args.algo}-{args.label_column}",
            config={**vars(args)},
            tags=["gbdt", args.algo, args.label_column, "hyperparameter-sweep"]
        )

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    X, y, label = load_xy(Path(args.data_root), args.label_column)
    prev = y.mean()
    print(f"[GBDT] X={X.shape} positives={y.sum()}/{len(y)} prev={prev:.2%} target={label}")
    if WANDB_AVAILABLE and run is not None:
        wandb.log({"dataset/n_samples": len(X), "dataset/n_features": X.shape[1], "dataset/n_positives": int(y.sum()), "dataset/prevalence": float(prev)})

    best = {"auc": -1}; records = []
    if args.algo == "xgb": sampler = make_param_sampler_xgb(prev)
    else:
        def sampler():
            rng = np.random.default_rng(42)
            return dict(
                learning_rate=float(10**rng.uniform(-2.0, -0.7)),
                num_leaves=int(2**rng.uniform(4, 7.5)),
                max_depth=int(rng.integers(-1, 11)),
                min_child_samples=int(2**rng.uniform(3, 8)),
                subsample=float(rng.uniform(0.6,1.0)),
                colsample_bytree=float(rng.uniform(0.6,1.0)),
                reg_alpha=float(10**rng.uniform(-4, 0)),
                reg_lambda=float(10**rng.uniform(-3, 1)),
                n_estimators=int(rng.integers(800, 2000)),
            )

    for t in range(1, args.trials+1):
        params = sampler()
        if args.algo == "xgb":
            oof, fi = cv_xgb(X, y, params, n_splits=args.n_splits, seed=args.seed, metric=args.eval_metric)
        else:
            oof, fi = cv_lgb(X, y, params, n_splits=args.n_splits, seed=args.seed, metric=args.eval_metric)
        smry = summarize_all(y, oof, label="oof")
        rec = {"trial": t, **params, **smry}; records.append(rec)
        if smry["auc"] > best["auc"]:
            best = {"trial": t, "params": params, **smry}
            fi.to_csv(out / "feature_importance_running_best.csv", index=False)
            pd.DataFrame({"oof": oof, "y": y}).to_csv(out / "oof_running_best.csv", index=False)
        print(f"[{t:03d}] AUC={smry['auc']:.4f} AP={smry['ap']:.4f}")

    pd.DataFrame(records).sort_values("auc", ascending=False).to_csv(out/"trials_summary.csv", index=False)
    (out/"best_params.json").write_text(json.dumps(best, indent=2))

    if WANDB_AVAILABLE and run is not None:
        wandb.log({"best_model/auc": best["auc"], "best_model/ap": best["ap"],
                   "best_model/ks": best["ks"], "best_model/gini": best["gini"],
                   "best_model/brier": best["brier"], "best_model/logloss": best["logloss"]})

    # Calibrate OOF
    oof_df = pd.read_csv(out/"oof_running_best.csv")
    lr = LogisticRegression(max_iter=1000).fit(oof_df[["oof"]].values, oof_df["y"].values)
    oof_cal = lr.predict_proba(oof_df[["oof"]].values)[:,1]
    smry_cal = summarize_all(oof_df["y"].values, oof_cal, label="oof_cal")
    pd.DataFrame({"oof_cal": oof_cal, "y": oof_df["y"].values}).to_csv(out/"oof_running_best_calibrated.csv", index=False)
    (out/"calibration_summary.json").write_text(json.dumps(smry_cal, indent=2))
    if WANDB_AVAILABLE and run is not None:
        wandb.log({"calibrated/auc": smry_cal["auc"], "calibrated/ap": smry_cal["ap"], "calibrated/ks": smry_cal["ks"], "calibrated/logloss": smry_cal["logloss"]})

    # Optional test evaluation
    if args.splits_csv and Path(args.splits_csv).exists():
        sp = pd.read_csv(args.splits_csv)
        test_idx = np.where(sp["is_test"].values==1)[0]
        if len(test_idx) > 0:
            if args.algo == "xgb":
                import xgboost as xgb
                final = xgb.XGBClassifier(objective="binary:logistic", tree_method="hist",
                                          random_state=args.seed, **best["params"])
            else:
                import lightgbm as lgb
                final = lgb.LGBMClassifier(objective="binary", random_state=args.seed, **best["params"])
            final.fit(X, y)
            ptest = final.predict_proba(X.iloc[test_idx])[:,1]
            smry_test = summarize_all(y[test_idx], ptest, label="test")
            (out/"test_summary.json").write_text(json.dumps(smry_test, indent=2))
            dec = decile_table(y[test_idx], ptest); dec.to_csv(out/"deciles_test.csv", index=False)
            if WANDB_AVAILABLE and run is not None:
                wandb.log({"test/auc": smry_test["auc"], "test/ap": smry_test["ap"], "test/ks": smry_test["ks"],
                           "test/brier": smry_test["brier"], "test/logloss": smry_test["logloss"]})
            print(f"[TEST] AUC={smry_test['auc']:.4f} AP={smry_test['ap']:.4f}")

    if WANDB_AVAILABLE and run is not None:
        wandb.finish()
    print("\n== DONE ==")
    print(json.dumps(best, indent=2))

if __name__ == "__main__":
    main()
