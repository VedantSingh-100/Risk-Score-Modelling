# build_features_snapshot.py
import argparse, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ----------------------- patterns -----------------------
OUTCOME_PAT = re.compile(
    r"(write[\s-]?off|charge[\s-]?off|npa|default|dpd|over[-\s]?limit|"
    r"over\s?due|overdue|past.?due|arrear|min[_\s-]?due|mindue|miss(?:ed|ing)?|"
    r"declin\w*|reject\w*|insufficient|penalt\w*|bounc\w*|ecs|nach|negative\w*)",
    re.I
)
LIFETIME_PAT = re.compile(r"\b(life\s*time|lifetime|ever|since\s*inception)\b", re.I)
ID_PAT = re.compile(r"(?:^|_)(id|uuid|pan|aadhaar|account|application|lead|mobile|email)(?:_|$)", re.I)

# ----------------------- helpers ------------------------
def read_descriptions(xlsx_path: Path) -> dict:
    if not xlsx_path.exists():
        return {}
    try:
        excel = pd.ExcelFile(xlsx_path)
        best_sheet = None; best_len = -1
        for s in excel.sheet_names:
            df = pd.read_excel(xlsx_path, sheet_name=s)
            if len(df) > best_len:
                best_sheet, best_len = s, len(df)
        df = pd.read_excel(xlsx_path, sheet_name=best_sheet).copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        var_col = next((c for c in df.columns if "variable" in c or "name" in c or "field" in c), None)
        desc_col = next((c for c in df.columns if "explanation" in c or "description" in c or "meaning" in c), None)
        if var_col and desc_col:
            return {str(r[var_col]).strip(): str(r[desc_col]) for _, r in df.iterrows() if pd.notna(r[var_col])}
    except Exception as e:
        print(f"[WARN] Could not parse dictionary: {e}")
    return {}

def is_outcome_like(col: str, desc: str) -> bool:
    s = f"{col} {desc}".lower()
    return bool(OUTCOME_PAT.search(s))

def is_lifetime_window(col: str, desc: str) -> bool:
    s = f"{col} {desc}".lower()
    return bool(LIFETIME_PAT.search(s))

def id_like_columns(cols) -> set:
    return {c for c in cols if ID_PAT.search(str(c))}

def near_unique_columns(df: pd.DataFrame, thresh: float = 0.98) -> set:
    out = set()
    n = len(df)
    for c in df.columns:
        try:
            if df[c].nunique(dropna=False) >= thresh * n:
                out.add(c)
        except Exception:
            pass
    return out

def fill_rate(s: pd.Series) -> float:
    return 1.0 - float(s.isna().mean())

def infer_numeric_kind(col: str, desc: str, s: pd.Series) -> str:
    """Return one of {'count','ratio','amount','other'}"""
    name = f"{col} {desc}".lower()
    if any(k in name for k in ["count","no.","num","times","frequency","txn_cnt","#","occurrence","instances"]):
        return "count"
    if any(k in name for k in ["ratio","rate","pct","percent","utilization","dti","share","proportion"]):
        return "ratio"
    if any(k in name for k in ["amount","amt","balance","outstanding","income","expense","bill","limit","emi","loan"]):
        return "amount"

    # Heuristics from data:
    x = pd.to_numeric(s, errors="coerce")
    if (x.dropna()>=0).mean() > 0.98 and (x.dropna() % 1 == 0).mean() > 0.95:
        return "count"
    if x.dropna().between(-5,5).mean() > 0.98:
        return "ratio"
    return "other"

def robust_scale_df(df_num: pd.DataFrame):
    med = df_num.median()
    iqr = df_num.quantile(0.75) - df_num.quantile(0.25)
    iqr_repl = iqr.replace(0, 1.0)
    scaled = (df_num - med) / iqr_repl
    return scaled, med.to_dict(), iqr.to_dict()

def corr_prune_graph(df_num: pd.DataFrame, thr: float = 0.97, priority: pd.Series | None = None):
    """Graph-based pruning: build edges for |r|>=thr and keep higher-priority node in collisions."""
    if df_num.shape[1] <= 1: 
        return df_num.columns.tolist(), []

    corr = df_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    # Collect edges
    edges = [(i,j) for i in upper.columns for j in upper.index if i!=j and upper.loc[j,i] >= thr]
    drop = set()
    kept = set(df_num.columns)
    # Greedy by priority
    def score(c):
        if priority is None: return 0.0
        return float(priority.get(c, 0.0))
    for i,j in edges:
        if i in drop or j in drop: 
            continue
        # keep higher priority
        wi, wj = score(i), score(j)
        loser = i if wi < wj else j
        drop.add(loser)
        if loser in kept: kept.remove(loser)
    return sorted(list(kept)), sorted(list(drop))

# ----------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", required=True,
                    help="Your big user CSV (same one framework used)")
    ap.add_argument("--dict-xlsx", required=True,
                    help="Internal_Algo360VariableDictionary_WithExplanation.xlsx")
    ap.add_argument("--interim-dir", required=True,
                    help="Folder where framework wrote composite_labels.csv & do_not_use_features.txt")
    ap.add_argument("--out-dir", required=True,
                    help="Folder to write X_features.parquet, y_label.csv, feature_keep_list.txt")
    ap.add_argument("--label-column", default="label_union",
                    help="Which label column from composite_labels.csv to use")
    ap.add_argument("--fill-rate-min", type=float, default=0.85)
    ap.add_argument("--corr-thr", type=float, default=0.97)
    ap.add_argument("--allow-outcome-like", action="store_true",
                    help="If set, do NOT drop outcome-like features (NOT recommended).")
    ap.add_argument("--scale", action="store_true", default=True,
                    help="Apply RobustScaler to numeric features")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    interim = Path(args.interim_dir)

    # Load inputs
    print("• Loading raw data...")
    df = pd.read_csv(args.raw_csv, low_memory=False)
    print(f"  -> {df.shape[0]:,} rows × {df.shape[1]:,} cols")

    print("• Loading variable descriptions...")
    desc_map = read_descriptions(Path(args.dict_xlsx))

    # Label
    comp_path = interim / "composite_labels.csv"
    if not comp_path.exists():
        raise RuntimeError(f"composite_labels.csv not found in {interim}")
    labels_df = pd.read_csv(comp_path)
    if args.label_column not in labels_df.columns:
        raise RuntimeError(f"Label '{args.label_column}' not in composite_labels.csv. "
                           f"Available: {list(labels_df.columns)[:10]} ...")

    # Align indices (framework used the same order as raw CSV)
    if len(labels_df) != len(df):
        raise RuntimeError(f"Row count mismatch: labels {len(labels_df)} vs raw {len(df)}")

    y = labels_df[[args.label_column]].copy().rename(columns={args.label_column: "label"})
    y.to_csv(out / "y_label.csv", index=False)
    print(f"• y_label.csv written. Positives={int(y['label'].sum())}/{len(y)} ({y['label'].mean():.4%})")

    # Build guard set
    guard = set()
    guard_file = interim / "do_not_use_features.txt"
    if guard_file.exists():
        guard |= {ln.strip() for ln in guard_file.read_text().splitlines() if ln.strip()}

    # IDs & near-unique
    guard |= id_like_columns(df.columns)
    guard |= near_unique_columns(df)

    # Outcome-like & lifetime-window features
    if not args.allow_outcome_like:
        for c in df.columns:
            d = desc_map.get(c, "")
            if is_outcome_like(c, d) or is_lifetime_window(c, d):
                guard.add(c)

    # Never use label columns as features
    guard |= set(labels_df.columns)

    # Candidate feature set after guard
    cand = [c for c in df.columns if c not in guard]
    print(f"• Guarded off {len(guard)} columns; {len(cand)} candidates remain.")

    # Fill-rate filter
    fr = df[cand].isna().mean().rsub(1.0)  # fill rate
    keep_fr = fr[fr >= args.fill_rate_min].index.tolist()
    print(f"• Fill-rate ≥ {args.fill_rate_min:.2f}: kept {len(keep_fr)} / {len(cand)}")

    # Split numeric/cat
    Xc = df[keep_fr]
    num_cols = Xc.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in keep_fr if c not in num_cols]

    # ---- Encode categoricals ----
    print("• Encoding categoricals (OHE for low-card, freq for high-card)...")
    X_parts = []

    # Frequency enc function
    def freq_encode(s: pd.Series):
        vc = s.value_counts(dropna=False, normalize=True)
        return s.map(vc).fillna(0.0), vc.to_dict()

    cat_manifest = {}
    for c in cat_cols:
        s = Xc[c].astype("object")
        # treat missing as a category
        s = s.where(~s.isna(), "__MISSING__")
        nunq = s.nunique()
        if nunq <= 8:
            top = s.value_counts().index.tolist()
            for val in top:
                col_new = f"{c}__is_{str(val)[:30]}"
                X_parts.append((col_new, (s == val).astype(int)))
            cat_manifest[c] = {"encoding":"ohe","levels": top}
        else:
            enc, mapping = freq_encode(s)
            col_new = f"{c}__freq"
            X_parts.append((col_new, enc.astype(float)))
            cat_manifest[c] = {"encoding":"freq","mapping_top": dict(list(mapping.items())[:50])}

    # ---- Transform numerics ----
    print("• Transforming numerics (log1p for counts; asinh otherwise)...")
    num_manifest = {}
    X_num = pd.DataFrame(index=df.index)
    for c in num_cols:
        d = desc_map.get(c, "")
        kind = infer_numeric_kind(c, d, Xc[c])
        x = pd.to_numeric(Xc[c], errors="coerce")
        med = x.median()
        x = x.fillna(med)
        if kind == "count":
            # guard against negatives
            shift = 0.0 if (x>=0).all() else (-x.min())
            z = np.log1p(x + shift)
        else:
            z = np.arcsinh(x)  # works with negatives and heavy tails
        X_num[c] = z.astype(float)
        num_manifest[c] = {"kind": kind, "impute": float(med)}

    # ---- Robust scale numerics (median/IQR) ----
    scaler_meta = {}
    if args.scale and len(X_num.columns) > 0:
        med = X_num.median()
        iqr = X_num.quantile(0.75) - X_num.quantile(0.25)
        iqr_safe = iqr.replace(0, 1.0)
        X_num = (X_num - med) / iqr_safe
        scaler_meta = {
            "median": {k: float(v) for k,v in med.items()},
            "iqr": {k: float(v) for k,v in iqr.items()}
        }

    # Assemble matrix before correlation pruning
    X_list = [X_num] + [pd.DataFrame({name: series}) for name, series in X_parts]
    X_all = pd.concat(X_list, axis=1) if X_list else pd.DataFrame(index=df.index)

    # ---- Graph-based correlation pruning on numerics only ----
    print(f"• Correlation pruning at |r| ≥ {args.corr_thr:.2f} (numerics only)...")
    num_priority = X_num.var().fillna(0.0)
    keep_num, drop_num = corr_prune_graph(X_num, thr=args.corr_thr, priority=num_priority)
    X_final = pd.concat([X_num[keep_num], X_all.drop(columns=X_num.columns, errors="ignore")], axis=1)

    # ---- Save outputs ----
    X_final.to_parquet(out / "X_features.parquet", index=False)
    Path(out / "feature_keep_list.txt").write_text("\n".join(X_final.columns.tolist()))
    manifest = {
        "n_rows": len(df),
        "guards_used": len(guard),
        "fill_rate_min": args.fill_rate_min,
        "corr_thr": args.corr_thr,
        "label_column": args.label_column,
        "stats": {
            "candidates": len(cand),
            "kept_fr": len(keep_fr),
            "num_before_prune": len(num_cols),
            "num_after_prune": len(keep_num),
            "cat_encoded_cols": len(X_final.columns) - len(keep_num)
        },
        "scaler_meta": scaler_meta,
        "categorical_manifest": cat_manifest,
        "numeric_manifest": num_manifest,
    }
    (out / "fe_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"• Wrote {out/'X_features.parquet'} with {X_final.shape[1]} columns.")
    print("✅ Feature engineering (snapshot) complete.")

if __name__ == "__main__":
    main()
