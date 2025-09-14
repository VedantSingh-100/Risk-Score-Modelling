# qc_suite.py
import json, math, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, ks_2samp

warnings.filterwarnings("ignore")

ROOT = Path("/home/vhsingh/Parshvi_project/deterministic_fe_outputs")
OUTPUT_BASE_DIR = Path("/home/vhsingh/qc_outputs")
if not OUTPUT_BASE_DIR.exists():
    OUTPUT_BASE_DIR.mkdir(exist_ok=True, parents=True)
X_PATH = ROOT / "X_features.parquet"
Y_PATH = ROOT / "y_label.csv"
DROP_PATH = ROOT / "dropped_features.json"
GUARD_TXT = ROOT / "guard_Set.txt"  # optional
TF_CFG = ROOT / "transforms_config.json"

OUT_QC_NULLS = OUTPUT_BASE_DIR / "qc_nulls_report.csv"
OUT_QC_GUARD = OUTPUT_BASE_DIR / "qc_guard_violations.csv"
OUT_QC_REDUND = OUTPUT_BASE_DIR / "qc_redundancy_pairs.csv"
OUT_QC_SINGLE = OUTPUT_BASE_DIR / "qc_single_feature_metrics.csv"
OUT_SUMMARY = OUTPUT_BASE_DIR / "qc_summary.md"

def _read_target(y_path: Path) -> pd.Series:
    ydf = pd.read_csv(y_path)
    # Preference order for target column
    for c in ["label_union", "label_weighted", "label_hierarchical", "label_clustered", "target", "y"]:
        if c in ydf.columns:
            y = ydf[c]
            break
    else:
        # Fallback: first binary-looking column
        bin_cols = [c for c in ydf.columns if set(pd.Series(ydf[c]).dropna().unique()) <= {0,1}]
        if not bin_cols:
            raise ValueError("Could not infer target column from y_label.csv")
        y = ydf[bin_cols[0]]
    return y.astype(int).reset_index(drop=True)

def _jaccard_binary(a: np.ndarray, b: np.ndarray) -> float:
    A = a.astype(bool); B = b.astype(bool)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return float(inter/union) if union > 0 else 0.0

def main():
    assert X_PATH.exists(), f"Missing {X_PATH}"
    assert Y_PATH.exists(), f"Missing {Y_PATH}"

    X = pd.read_parquet(X_PATH)
    y = _read_target(Y_PATH)
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X={len(X)} rows vs y={len(y)}")

    # ---------- 1) No NaNs/Infs ----------
    nulls = X.isna().sum().rename("missing_count")
    infs = np.isinf(X.select_dtypes(include=[np.number])).sum().rename("inf_count")
    nulls_df = pd.concat([nulls, infs], axis=1).fillna(0).astype(int).reset_index().rename(columns={"index":"feature"})
    nulls_df.to_csv(OUT_QC_NULLS, index=False)

    # ---------- 2) Guard compliance ----------
    guard = set()
    if DROP_PATH.exists():
        dropped = json.loads(DROP_PATH.read_text())
        for bucket in ["guard","outcome_guard","identifier"]:
            guard |= set(dropped.get(bucket, []))
    if GUARD_TXT.exists():
        for ln in GUARD_TXT.read_text().splitlines():
            ln = ln.strip()
            if ln:
                guard.add(ln)
    guard_viol = [c for c in X.columns if c in guard]
    pd.DataFrame({"feature": guard_viol}).to_csv(OUT_QC_GUARD, index=False)

    # ---------- 3) Redundancy (|ρ| >= 0.97) ----------
    # Use a numerically-stable, memory-friendly pass
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    corr_pairs = []
    # Standardize columns to avoid scale dominance
    Z = (X[numeric_cols] - X[numeric_cols].mean()) / (X[numeric_cols].std(ddof=0) + 1e-12)
    for i in range(len(numeric_cols)):
        xi = Z.iloc[:, i].values
        for j in range(i+1, len(numeric_cols)):
            xj = Z.iloc[:, j].values
            r = np.nan_to_num(np.corrcoef(xi, xj)[0,1])
            if abs(r) >= 0.97:
                corr_pairs.append((numeric_cols[i], numeric_cols[j], float(r)))
    pd.DataFrame(corr_pairs, columns=["feature_i","feature_j","pearson_r"]).to_csv(OUT_QC_REDUND, index=False)

    # ---------- 4) Single-feature leakage-ish metrics ----------
    rows = []
    y_arr = y.values
    for c in numeric_cols:
        x = X[c].values
        # AUC (guard against constant)
        try:
            if np.nanstd(x) > 0 and len(np.unique(x)) > 1 and y_arr.sum() not in (0, len(y_arr)):
                auc = roc_auc_score(y_arr, x)
            else:
                auc = np.nan
        except Exception:
            auc = np.nan
        # Spearman
        try:
            rho, _ = spearmanr(x, y_arr)
        except Exception:
            rho = np.nan
        # Quick KS (positive vs negative)
        try:
            ks = ks_2samp(x[y_arr==1], x[y_arr==0]).statistic
        except Exception:
            ks = np.nan
        # Optional “Jaccard via top-half thresholding” (heuristic)
        try:
            thr = np.nanmedian(x)
            jac = _jaccard_binary((x >= thr).astype(int), y_arr)
        except Exception:
            jac = np.nan
        rows.append((c, auc, rho, ks, jac))
    pd.DataFrame(rows, columns=["feature","auc_1d","spearman","ks","jaccard_via_median"]).to_csv(OUT_QC_SINGLE, index=False)

    # ---------- 5) Summary ----------
    with open(OUT_SUMMARY, "w") as f:
        f.write("# Feature Engineering QC Summary\n\n")
        f.write(f"- Rows: {len(X):,} | Features: {X.shape[1]}\n")
        f.write(f"- Target positives: {int(y.sum()):,} (prevalence {y.mean():.4f})\n")
        bad_null = nulls_df.query("missing_count>0 or inf_count>0")
        f.write(f"- Null/Inf violations: {len(bad_null)} (see {OUT_QC_NULLS.name})\n")
        f.write(f"- Guard violations: {len(guard_viol)} (see {OUT_QC_GUARD.name})\n")
        f.write(f"- Redundancy pairs (|r|>=0.97): {len(corr_pairs)} (see {OUT_QC_REDUND.name})\n")
        f.write(f"- Single-feature metrics written to {OUT_QC_SINGLE.name}\n")

if __name__ == "__main__":
    main()
