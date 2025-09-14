# io_utils.py
import json, numpy as np, pandas as pd
from pathlib import Path

def load_xy(data_root: str):
    X = pd.read_parquet(Path(data_root)/"X_features.parquet")
    y_df = pd.read_csv(Path(data_root)/"y_label.csv")
    # first non-id column is the target (your file contains label_union)
    target_col = [c for c in y_df.columns if c.lower() not in {"id","row_id","user_id"}][0]
    y = y_df[target_col].astype(int).values
    return X, y, target_col

def maybe_load_monotone(data_root: str, feature_names):
    cfg_path = Path(data_root)/"monotone_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    vec = [int(cfg.get(f, 0)) for f in feature_names]  # 1, -1, 0
    return vec

def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(obj, indent=2))

def save_df(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
