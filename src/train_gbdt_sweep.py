# train_gbdt_sweep.py
import json, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score

from .metrics import summarize_all
from .io_utils import load_xy, dump_json, save_df

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

def cv_score_xgb(X, y, params, n_splits=5, seed=42):
    """Cross-validation scoring for XGBoost with early stopping."""
    import xgboost as xgb
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = xgb.XGBClassifier(**params)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            verbose=False
        )
        pred = model.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y[va], pred)
        scores.append(auc)
    
    return np.mean(scores), np.std(scores)

def cv_score_lgb(X, y, params, n_splits=5, seed=42):
    """Cross-validation scoring for LightGBM with early stopping."""
    import lightgbm as lgb
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        pred = model.predict_proba(X.iloc[va])[:, 1]
        auc = roc_auc_score(y[va], pred)
        scores.append(auc)
    
    return np.mean(scores), np.std(scores)

def objective_xgb(trial, X, y, n_splits, seed, use_wandb=False):
    """Optuna objective for XGBoost hyperparameter optimization."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'random_state': seed,
        'n_jobs': -1,
        
        # Hyperparameters to optimize
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'early_stopping_rounds': 100
    }
    
    mean_auc, std_auc = cv_score_xgb(X, y, params, n_splits, seed)
    
    # Log to wandb if enabled
    if use_wandb and HAS_WANDB:
        wandb.log({
            'trial': trial.number,
            'cv_auc_mean': mean_auc,
            'cv_auc_std': std_auc,
            **{f'param_{k}': v for k, v in params.items() if k not in ['objective', 'eval_metric', 'tree_method', 'random_state', 'n_jobs']}
        })
    
    return mean_auc

def objective_lgb(trial, X, y, n_splits, seed, use_wandb=False):
    """Optuna objective for LightGBM hyperparameter optimization."""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'random_state': seed,
        'n_jobs': -1,
        'verbose': -1,
        
        # Hyperparameters to optimize
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }
    
    mean_auc, std_auc = cv_score_lgb(X, y, params, n_splits, seed)
    
    # Log to wandb if enabled
    if use_wandb and HAS_WANDB:
        wandb.log({
            'trial': trial.number,
            'cv_auc_mean': mean_auc,
            'cv_auc_std': std_auc,
            **{f'param_{k}': v for k, v in params.items() if k not in ['objective', 'metric', 'boosting_type', 'random_state', 'n_jobs', 'verbose']}
        })
    
    return mean_auc

def main():
    import argparse
    ap = argparse.ArgumentParser(description="GBDT Hyperparameter Sweep with Optuna")
    ap.add_argument("--data-root", default="data/processed", help="Data directory")
    ap.add_argument("--out-dir", default="model_outputs/gbdt_sweep", help="Output directory")
    ap.add_argument("--algo", choices=["xgb", "lgb"], default="xgb", help="Algorithm to optimize")
    ap.add_argument("--trials", type=int, default=60, help="Number of optimization trials")
    ap.add_argument("--n-splits", type=int, default=5, help="CV folds")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    args = ap.parse_args()
    
    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    X, y, target_col = load_xy(args.data_root)
    print(f"Loaded X: {X.shape}, y: {len(y)} (prevalence: {y.mean():.4f})")
    print(f"Target column: {target_col}")
    
    # Initialize wandb if requested
    use_wandb = args.wandb and HAS_WANDB
    if use_wandb:
        wandb.init(
            project="Risk Score",
            name=f"{args.algo.upper()}_sweep_{args.trials}trials",
            config={
                "algorithm": args.algo,
                "trials": args.trials,
                "n_splits": args.n_splits,
                "seed": args.seed,
                "n_samples": len(y),
                "n_features": X.shape[1],
                "prevalence": float(y.mean())
            }
        )
    
    # Create study
    study = optuna.create_study(direction='maximize', seed=args.seed)
    
    # Define objective function
    if args.algo == "xgb":
        objective_func = lambda trial: objective_xgb(trial, X, y, args.n_splits, args.seed, use_wandb)
        print(f"Starting XGBoost hyperparameter optimization with {args.trials} trials...")
    else:
        objective_func = lambda trial: objective_lgb(trial, X, y, args.n_splits, args.seed, use_wandb)
        print(f"Starting LightGBM hyperparameter optimization with {args.trials} trials...")
    
    # Run optimization
    study.optimize(objective_func, n_trials=args.trials, show_progress_bar=True)
    
    # Get best parameters
    best_params = study.best_params.copy()
    best_score = study.best_value
    
    print(f"\nOptimization complete!")
    print(f"Best CV AUC: {best_score:.6f}")
    print(f"Best parameters: {json.dumps(best_params, indent=2)}")
    
    # Save results
    results = {
        "algorithm": args.algo,
        "best_score": float(best_score),
        "best_params": best_params,
        "n_trials": args.trials,
        "target_column": target_col
    }
    
    dump_json(results, Path(args.out_dir) / "best_params.json")
    
    # Save all trials
    trials_df = study.trials_dataframe()
    save_df(trials_df, Path(args.out_dir) / "all_trials.csv")
    
    # Train final model with best parameters and get OOF predictions
    print("Training final model with best parameters...")
    
    if args.algo == "xgb":
        import xgboost as xgb
        final_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'tree_method': 'hist',
            'random_state': args.seed,
            'n_jobs': -1,
            **best_params
        }
        
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y))
        
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            model = xgb.XGBClassifier(**final_params)
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[va], y[va])],
                eval_metric="auc",
                verbose=False
            )
            pred = model.predict_proba(X.iloc[va])[:, 1]
            oof[va] = pred
    
    else:  # LightGBM
        import lightgbm as lgb
        final_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'random_state': args.seed,
            'n_jobs': -1,
            'verbose': -1,
            **best_params
        }
        
        skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        oof = np.zeros(len(y))
        
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            model = lgb.LGBMClassifier(**final_params)
            model.fit(
                X.iloc[tr], y[tr],
                eval_set=[(X.iloc[va], y[va])],
                eval_metric="auc",
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            pred = model.predict_proba(X.iloc[va])[:, 1]
            oof[va] = pred
    
    # Calculate final metrics
    final_auc = roc_auc_score(y, oof)
    final_ap = average_precision_score(y, oof)
    
    print(f"Final OOF AUC: {final_auc:.6f}")
    print(f"Final OOF AP: {final_ap:.6f}")
    
    # Save OOF predictions
    oof_df = pd.DataFrame({
        'oof_pred': oof,
        'y_true': y
    })
    save_df(oof_df, Path(args.out_dir) / "oof_predictions.csv")
    
    # Generate summary metrics
    summary, deciles = summarize_all(y, oof, label=f"{args.algo}_best")
    save_df(deciles, Path(args.out_dir) / "decile_analysis.csv")
    
    # Update results with final metrics
    results.update({
        "final_oof_auc": float(final_auc),
        "final_oof_ap": float(final_ap),
        "summary_metrics": summary
    })
    
    dump_json(results, Path(args.out_dir) / "final_results.json")
    
    # Log final results to wandb
    if use_wandb:
        wandb.log({
            "best_cv_auc": best_score,
            "final_oof_auc": final_auc,
            "final_oof_ap": final_ap,
            "gini": summary["gini"],
            "ks_stat": summary["ks"]
        })
        wandb.finish()
    
    print(f"\nAll results saved to: {args.out_dir}")
    print(f"Best parameters saved to: {args.out_dir}/best_params.json")

if __name__ == "__main__":
    main()
