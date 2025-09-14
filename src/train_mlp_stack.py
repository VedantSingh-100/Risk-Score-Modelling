# train_mlp_stack.py
import json, numpy as np, pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from .metrics import summarize_all
from .io_utils import load_xy, dump_json

def cv_oof(model, X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, y):
        m = model()
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[va])[:,1]
        oof[va] = p
    return oof

def make_mlp():
    return Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128,64),
            activation="relu",
            alpha=1e-4, batch_size=256,
            learning_rate_init=1e-3,
            max_iter=80, random_state=42, verbose=False))
    ])

def make_xgb(best_params=None):
    import xgboost as xgb
    params = dict(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        n_estimators=1200,
        eta=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.5,
        random_state=42, early_stopping_rounds=200
    )
    if best_params:
        params.update(best_params)
        params.pop("scale_pos_weight", None)  # we’ll re-fit without early-stop meta params
    def _factory():
        return xgb.XGBClassifier(**params)
    return _factory

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default="model_outputs/stack")
    ap.add_argument("--best-xgb-params", default="model_outputs/gbdt_sweep/best_params.json")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    X, y, target = load_xy(args.data_root)

    best_params = None
    p = Path(args.best_xgb_params)
    if p.exists():
        best_params = json.loads(p.read_text()).get("params", None)

    # 1) OOF from XGB and MLP
    oof_xgb = cv_oof(make_xgb(best_params), X, y, n_splits=args.n_splits, seed=args.seed)
    oof_mlp = cv_oof(make_mlp,             X, y, n_splits=args.n_splits, seed=args.seed)

    # 2) Calibrated stacker (Platt)
    Z = np.column_stack([oof_xgb, oof_mlp])
    cal = LogisticRegression(max_iter=1000).fit(Z, y)
    oof_stack = cal.predict_proba(Z)[:,1]

    # 3) Metrics & artifacts
    rows = []
    for name, vec in [("xgb", oof_xgb), ("mlp", oof_mlp), ("stack", oof_stack)]:
        smry, dec = summarize_all(y, vec, label=name)
        dec.to_csv(Path(args.out_dir)/f"deciles_{name}.csv", index=False)
        rows.append({"model": name, **smry})

    pd.DataFrame({"oof_xgb": oof_xgb, "oof_mlp": oof_mlp, "oof_stack": oof_stack, "y": y})\
      .to_csv(Path(args.out_dir)/"oof_predictions.csv", index=False)

    pd.DataFrame(rows).to_csv(Path(args.out_dir)/"summary.csv", index=False)
    dump_json({"coef": cal.coef_.ravel().tolist(), "intercept": float(cal.intercept_[0])},
              Path(args.out_dir)/"stacker_logit.json")

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nArtifacts → {args.out_dir}")

if __name__ == "__main__":
    main()
