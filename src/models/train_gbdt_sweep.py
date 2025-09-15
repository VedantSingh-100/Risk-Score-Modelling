# train_gbdt_sweep.py
import os, json, time, math, random, warnings
import numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from ..utils.metrics import summarize_all, decile_table
from ..utils.io_utils import load_xy, maybe_load_monotone, dump_json, save_df

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(42)

def try_import_xgb():
    try:
        import xgboost as xgb
        return xgb
    except Exception:
        return None

def try_import_lgb():
    try:
        import lightgbm as lgb
        return lgb
    except Exception:
        return None

def make_param_sampler(algo, X, y):
    # class balance (note: your prevalence ~= 0.52, so this is mild)
    pos = y.sum(); neg = len(y)-pos
    spw = max(neg/(pos+1e-9), 0.5)  # ~0.92 here

    if algo == "xgb":
        space = {
            "eta":        lambda: 10**RNG.uniform(-2.0, -0.7),   # 0.01–0.2
            "max_depth":  lambda: RNG.integers(3, 9),
            "min_child_weight": lambda: 10**RNG.uniform(-1, 2),  # 0.1–100
            "subsample":  lambda: RNG.uniform(0.6, 1.0),
            "colsample_bytree": lambda: RNG.uniform(0.6, 1.0),
            "gamma":      lambda: 10**RNG.uniform(-3, 1),        # 0.001–10
            "reg_alpha":  lambda: 10**RNG.uniform(-4, 0),        # 1e-4–1
            "reg_lambda": lambda: 10**RNG.uniform(-3, 1),        # 0.001–10
            "scale_pos_weight": lambda: RNG.choice([1.0, spw, 1.25*spw]),
            "n_estimators": lambda: RNG.integers(800, 2500)
        }
    else:  # lightgbm
        space = {
            "learning_rate": lambda: 10**RNG.uniform(-2.0, -0.7),
            "num_leaves":    lambda: int(2**RNG.uniform(4, 7.5)),  # 16–181
            "max_depth":     lambda: RNG.integers(-1, 11),
            "min_child_samples": lambda: int(2**RNG.uniform(3, 8)), # 8–256
            "subsample":     lambda: RNG.uniform(0.6, 1.0),
            "colsample_bytree": lambda: RNG.uniform(0.6, 1.0),
            "reg_alpha":     lambda: 10**RNG.uniform(-4, 0),
            "reg_lambda":    lambda: 10**RNG.uniform(-3, 1),
            "n_estimators":  lambda: RNG.integers(800, 2500),
            "scale_pos_weight": lambda: RNG.choice([1.0, spw, 1.25*spw])
        }
    return space

def sample_params(space):
    return {k: v() for k, v in space.items()}

def run_cv_xgb(X, y, params, monotone_vec=None, seed=42, n_splits=5, eval_metric="auc"):
    import xgboost as xgb
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y)); feats = X.columns.tolist(); fi = np.zeros(len(feats))
    mono_str = None
    if monotone_vec is not None:
        mono_str = "(" + ",".join(str(int(v)) for v in monotone_vec) + ")"
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=eval_metric,           # << changed
            tree_method="hist",
            n_jobs=os.cpu_count(),             # << added
            random_state=seed,
            early_stopping_rounds=300,
            **params
        )
        if mono_str is not None:
            model.set_params(monotone_constraints=mono_str)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        p = model.predict_proba(Xva)[:,1]
        oof[va_idx] = p
        try: fi += model.feature_importances_
        except Exception: pass
    fi /= n_splits
    return oof, pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False)

def run_cv_lgb(X, y, params, monotone_vec=None, seed=42, n_splits=5):
    import lightgbm as lgb
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    feats = X.columns.tolist()
    fi = np.zeros(len(feats))
    for tr_idx, va_idx in skf.split(X, y):
        Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]
        model = lgb.LGBMClassifier(
            objective="binary",
            random_state=seed,
            n_jobs=os.cpu_count(),
            **params
        )
        if monotone_vec is not None:
            model.set_params(monotone_constraints=monotone_vec)
        model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)])
        p = model.predict_proba(Xva)[:,1]
        oof[va_idx] = p
        try:
            fi += model.booster_.feature_importance(importance_type="gain")
        except Exception:
            pass
    fi /= n_splits
    return oof, pd.DataFrame({"feature": feats, "importance": fi}).sort_values("importance", ascending=False)

def maybe_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
        # Use specific entity and project for your CMU workspace
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project or "Risk_Score", 
            config=vars(args)
        )
        return wandb
    except Exception as e:
        print(f"[wandb] disabled ({e})")
        return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="model_outputs/gbdt_sweep")
    ap.add_argument("--algo", choices=["xgb","lgb"], default="xgb")
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default="ved100-carnegie-mellon-university", 
                    help="W&B entity (organization/username)")
    ap.add_argument("--use-monotone", action="store_true",
                    help="loads data/monotone_config.json if present")
    ap.add_argument("--target", default=None, help="Explicit target column in y_label.csv")
    ap.add_argument("--prune-corr", action="store_true", help="Apply correlation pruning if keep list not present")
    ap.add_argument("--corr-thr", type=float, default=0.95, help="Correlation threshold for pruning")
    ap.add_argument("--eval-metric", choices=["auc", "aucpr", "auto"], default="auto",
                help="Early-stopping metric for XGB. 'auto' picks aucpr if prevalence<=0.20")

    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    X, y, label_name = load_xy(args.data_root, target_name=args.target)
    print(f"Loaded X={X.shape}, positives={int(y.sum())}/{len(y)}  (target={label_name})")
    X = X.astype(np.float32)  # speed/memory friendly
    prev = float(y.mean())
    metric = args.eval_metric
    if metric == "auto":
        metric = "aucpr" if prev <= 0.20 else "auc"
    print(f"[INFO] Prevalence={prev:.4%}; XGB eval_metric={metric}")
    # Respect feature_keep_list.txt if present (produced by framework)
    keep_path = Path(args.data_root)/"feature_keep_list.txt"
    drop_path = Path(args.data_root)/"feature_drop_corr.txt"

    if keep_path.exists():
        keep = [ln.strip() for ln in keep_path.read_text().splitlines() if ln.strip()]
        missing = [c for c in keep if c not in X.columns]
        if missing:
            print(f"[WARN] {len(missing)} kept features not in X; ignoring first few: {missing[:5]}")
        X = X[[c for c in keep if c in X.columns]]
        print(f"[KEEP-LIST] Using {X.shape[1]} features from feature_keep_list.txt")
    else:
        # If an explicit drop list exists, apply it defensively
        if drop_path.exists():
            to_drop = [ln.strip() for ln in drop_path.read_text().splitlines() if ln.strip()]
            have = [c for c in to_drop if c in X.columns]
            if have:
                X = X.drop(columns=have)
                print(f"[DROP-LIST] Dropped {len(have)} features from feature_drop_corr.txt")
    if args.prune_corr:
        # Light pruning at training time if no keep list
        num_cols = X.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) >= 2:
            imp = X[num_cols].copy().fillna(X[num_cols].median(numeric_only=True))
            # Simple priority by variance
            priority = imp.var(numeric_only=True)
            # Local prune function (same as in framework)
            def _prune(Xdf, thr, prio):
                corr = Xdf.corr().abs()
                upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                to_drop = set()
                for col in upper.columns:
                    highs = upper.index[upper[col] > thr].tolist()
                    for h in highs:
                        if h in to_drop or col in to_drop: 
                            continue
                        drop = col if float(prio.get(col,0)) < float(prio.get(h,0)) else h
                        to_drop.add(drop)
                kept = [c for c in Xdf.columns if c not in to_drop]
                return kept, sorted(list(to_drop))
            kept, dropped = _prune(imp, args.corr_thr, priority)
            X = X[kept + [c for c in X.columns if c not in num_cols]]  # keep non-numerics if any
            print(f"[CORR-PRUNE] Dropped {len(dropped)} numeric features (> r={args.corr_thr})")

    # select backend
    use_xgb = args.algo == "xgb" and (try_import_xgb() is not None)
    use_lgb = args.algo == "lgb" and (try_import_lgb() is not None)
    if not (use_xgb or use_lgb):
        raise RuntimeError("Neither XGBoost nor LightGBM available. Please install one.")

    wandb = maybe_wandb(args)
    param_space = make_param_sampler("xgb" if use_xgb else "lgb", X, y)
    monotone_vec = maybe_load_monotone(args.data_root, X.columns) if args.use_monotone else None

    trial_rows, best = [], {"auc": -1}
    for t in range(1, args.trials+1):
        params = sample_params(param_space)
        if use_xgb:
            oof, fi = run_cv_xgb(X, y, params, monotone_vec, seed=args.seed, n_splits=args.n_splits, eval_metric=metric)
        else:
            oof, fi = run_cv_lgb(X, y, params, monotone_vec, seed=args.seed, n_splits=args.n_splits)

        smry, dec = summarize_all(y, oof, label="oof")
        row = {"trial": t, **params, **smry}
        trial_rows.append(row)

        if smry["auc"] > best["auc"]:
            best = {"trial": t, "params": params, "auc": smry["auc"], "ap": smry["ap"]}
            # save running best artifacts
            fi.to_csv(Path(args.out_dir)/"feature_importance_running_best.csv", index=False)
            dec.to_csv(Path(args.out_dir)/"deciles_running_best.csv", index=False)
            pd.DataFrame({"oof_pred": oof, "y": y}).to_csv(Path(args.out_dir)/"oof_running_best.csv", index=False)

        if wandb:
            wandb.log({**smry, "trial": t})

        print(f"[{t:03d}/{args.trials}] AUC={smry['auc']:.4f}  AP={smry['ap']:.4f}")

    trials_df = pd.DataFrame(trial_rows).sort_values("auc", ascending=False)
    trials_df.to_csv(Path(args.out_dir)/"trials_summary.csv", index=False)
    dump_json(best, Path(args.out_dir)/"best_params.json")

    # Platt calibration on best OOF
    oof_df = pd.read_csv(Path(args.out_dir)/"oof_running_best.csv")
    clf_cal = LogisticRegression(max_iter=1000).fit(oof_df[["oof_pred"]].values, oof_df["y"].values)
    oof_cal = clf_cal.predict_proba(oof_df[["oof_pred"]].values)[:,1]
    cal_smry, cal_dec = summarize_all(oof_df["y"].values, oof_cal, label="oof_cal")
    cal_dec.to_csv(Path(args.out_dir)/"deciles_running_best_calibrated.csv", index=False)
    dump_json({"calibration_auc": cal_smry["auc"], "calibration_ap": cal_smry["ap"]},
              Path(args.out_dir)/"calibration_summary.json")

    if wandb:
        wandb.log({"oof_cal_auc": cal_smry["auc"], "oof_cal_ap": cal_smry["ap"]})
        wandb.finish()

    print("\n== DONE ==")
    print(f"Best trial #{best['trial']}  AUC={best['auc']:.4f}, AP={best['ap']:.4f}")
    print(f"Artifacts → {args.out_dir}")

if __name__ == "__main__":
    main()
