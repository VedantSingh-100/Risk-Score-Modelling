# io_utils.py
import json, numpy as np, pandas as pd
from pathlib import Path

def load_xy(data_root: str, target_name: str | None = None):
    X = pd.read_parquet(Path(data_root)/"X_features.parquet")
    y_df = pd.read_csv(Path(data_root)/"y_label.csv")

    # Try explicit target first
    if target_name is None:
        # Try recommended pipeline
        rec_json = Path(data_root)/"recommended_pipeline.json"
        if rec_json.exists():
            try:
                rec = json.loads(rec_json.read_text())
                target_name = rec.get("best_label", None)
            except Exception:
                target_name = None

    if target_name is None:
        # Fallback: first non-id column (warn)
        non_ids = [c for c in y_df.columns if c.lower() not in {"id","row_id","user_id"}]
        assert len(non_ids) > 0, "No target column found in y_label.csv"
        target_name = non_ids[0]
        print(f"[WARN] Using fallback target={target_name}")

    assert target_name in y_df.columns, f"Target '{target_name}' not in y_label.csv"
    y = y_df[target_name].astype(int).values
    return X, y, target_name


def maybe_load_monotone(data_root: str, feature_names):
    cfg_path = Path(data_root)/"monotone_config.json"
    if not cfg_path.exists():
        return None
    cfg = json.loads(cfg_path.read_text())
    vec = [int(cfg.get(f, 0)) for f in feature_names]  # 1, -1, 0
    return vec

def dump_json(obj, path):
    """Save object to JSON file with numpy type conversion."""
    import numpy as np
    
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    converted_obj = convert_numpy(obj)
    with open(path, "w") as f:
        f.write(json.dumps(converted_obj, indent=2))

def save_df(df, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
