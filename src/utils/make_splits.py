#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Folder containing X_features.parquet and y_label.csv")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_root)
    X = pd.read_parquet(root/"X_features.parquet") # loaded only to get n_rows
    y = pd.read_csv(root/"y_label.csv")["label"].astype(int).values
    n = len(y)
    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    # First: train vs (val+test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(args.val+args.test), random_state=args.seed)
    tr_idx, vt_idx = next(sss1.split(np.zeros(n), y))
    # Second: val vs test within vt_idx
    y_vt = y[vt_idx]
    test_frac = args.test / (args.val + args.test)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=args.seed)
    va_rel, te_rel = next(sss2.split(np.zeros(len(vt_idx)), y_vt))
    va_idx = vt_idx[va_rel]; te_idx = vt_idx[te_rel]

    splits = pd.DataFrame({
        "is_train": np.isin(np.arange(n), tr_idx).astype(int),
        "is_val": np.isin(np.arange(n), va_idx).astype(int),
        "is_test": np.isin(np.arange(n), te_idx).astype(int),
    })
    splits.to_csv(root/"splits.csv", index=False)
    print(f"Saved {root/'splits.csv'} | train={splits.is_train.sum()} val={splits.is_val.sum()} test={splits.is_test.sum()}")

if __name__ == "__main__":
    main()