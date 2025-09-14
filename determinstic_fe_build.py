import json, re, math
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
np.seterr(all="ignore")

# Ensure output directory exists
from pathlib import Path
OUTPUT_BASE_DIR = "/home/vhsingh/deterministic_fe_outputs"
Path(OUTPUT_BASE_DIR).mkdir(exist_ok=True, parents=True)

# -----------------------------
# CONFIG (deterministic)
# -----------------------------
RAW_DATA_PATHS = [
    "50k_users_merged_data_userfile_updated_shopping.csv",  # main
    "raw.csv",                                              # fallback name
]
SELECTED_FEATURES_FILES = [
    "deterministic_build/selected_features_final.csv",  # from build script
    "engineered/feature_list_final.csv",               # from engineer script  
    "selected_features_finalcsv",                       # fallback
    "feature_list_final.csv"                           # fallback
]
GUARD_FILES = [
    "deterministic_build/guard_set.txt",               # from build script
    "guard_Set.txt", "guard_set.txt", "do_not_use_features.txt"  # fallbacks
]
LABEL_FILES = [
    "engineered/y_label.csv",                          # from engineer script
    "deterministic_build/label_union.csv",             # from build script
    "y_label.csv", "label_union.csv"                   # fallbacks
]
LABEL_SOURCES_FILE = "deterministic_build/selected_label_sources.csv"  # from build script
BUILD_SUMMARY_FILES = [
    "deterministic_build/build_summary.json",          # from build script
    "engineered/best_config_used.json",                # from engineer script
    "build_summary.json"                               # fallback
]

FILL_RATE_THRESHOLD = 0.85     # driven by your plan and build_summary
CLIP_QUANTILES = (0.001, 0.999)
REPORT_PRE = f"{OUTPUT_BASE_DIR}/feature_engineering_report_pre.csv"
REPORT_POST = f"{OUTPUT_BASE_DIR}/feature_engineering_report_post.csv"
DROPPED_JSON = f"{OUTPUT_BASE_DIR}/dropped_features.json"
TRANSFORMS_JSON = f"{OUTPUT_BASE_DIR}/transforms_config.json"
LEAKAGE_CSV = f"{OUTPUT_BASE_DIR}/leakage_check.csv"
X_OUT = f"{OUTPUT_BASE_DIR}/X_features.parquet"
Y_OUT = f"{OUTPUT_BASE_DIR}/y_label.csv"

# Label/outcome guard terms (outcome words MUST NOT be features)
OUTCOME_GUARD_TERMS = (
    r"default", r"dpd", r"overdue", r"arrear", r"write[\s-]?off", r"charge[\s-]?off",
    r"npa", r"settle", r"miss", r"min[_\s-]?due", r"over[-\s]?limit", r"declin", r"reject",
    r"bounced", r"nsf", r"negative"
)

# Identifier patterns (drop from modeling)
IDENTIFIER_PATTERNS = (
    r"\b(account|acc|reference|ref|id|uuid|key|number|acct)\b",
    r"primary.*account.*number",
)

# -----------------------------
# Utilities
# -----------------------------
def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return p
    return None

def load_build_summary():
    cfg = {}
    for f in BUILD_SUMMARY_FILES:
        if Path(f).exists():
            try:
                obj = json.loads(Path(f).read_text())
                # normalize keys
                if "best_config_used" in obj:
                    cfg = obj["best_config_used"]
                    # pass through some top-level for convenience
                    cfg["fill_rate_threshold"] = obj.get("fill_rate_threshold", FILL_RATE_THRESHOLD)
                else:
                    cfg = obj
                break
            except Exception:
                pass
    return cfg

def load_selected_feature_list():
    f = first_existing(SELECTED_FEATURES_FILES)
    if not f:
        raise FileNotFoundError("selected_features._finalcsv or feature_list_final.csv not found.")
    df = pd.read_csv(f)
    # normalize columns
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]
    df.columns = cols
    # expected: variable, description, fill_rate (at least)
    # allow 'new_description' instead of 'description'
    if "description" not in df.columns and "new_description" in df.columns:
        df["description"] = df["new_description"]
    req = ["variable", "fill_rate"]
    for r in req:
        if r not in df.columns:
            raise ValueError(f"Missing column '{r}' in {f}. Found columns: {df.columns.tolist()}")
    return df

def load_guard_set():
    guard = set()
    for f in GUARD_FILES:
        p = Path(f)
        if p.exists():
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    guard.add(ln)
    return guard

def load_raw():
    f = first_existing(RAW_DATA_PATHS)
    if not f:
        raise FileNotFoundError("Raw data CSV not found. Expected one of: " + ", ".join(RAW_DATA_PATHS))
    return pd.read_csv(f, low_memory=False)

def load_label(y_hint_cols, df):
    # 1) y_label.csv present?
    lf = first_existing(LABEL_FILES[:1])  # y_label.csv only
    if lf:
        y = pd.read_csv(lf)
        if y.shape[1] == 1:
            return y.iloc[:,0].astype(int)
        # support a named column 'label'
        if "label" in y.columns:
            return y["label"].astype(int)
        return y.iloc[:,0].astype(int)

    # 2) label_union.csv present?
    lf = first_existing(LABEL_FILES[1:])  # label_union.csv
    if lf:
        y = pd.read_csv(lf)
        col = y.columns[0]
        return y[col].astype(int)

    # 3) Reconstruct union from label sources (binary events): event > 0
    src_file = Path(LABEL_SOURCES_FILE)
    if not src_file.exists():
        raise FileNotFoundError("No y_label.csv or label_union.csv found, and Selected_label_sources.csv is missing.")
    src = pd.read_csv(src_file)
    if "variable" not in src.columns:
        raise ValueError("Selected_label_sources.csv must have a 'variable' column.")
    cand = [c for c in src["variable"].astype(str) if c in df.columns]
    if not cand:
        raise ValueError("None of the label source variables are in the raw dataframe.")
    binmat = pd.DataFrame(index=df.index)
    for c in cand:
        s = pd.to_numeric(df[c], errors="coerce").fillna(0)
        binmat[c] = (s > 0).astype(int)
    return (binmat.sum(axis=1) > 0).astype(int)

def is_identifier(name, desc):
    text = f"{name} {desc}".lower()
    for pat in IDENTIFIER_PATTERNS:
        if re.search(pat, text):
            return True
    return False

def looks_like_ratio_01(name, desc):
    text = f"{name} {desc}".lower()
    return ("ratio" in text and "vintage" in text) or "balance is less than" in text

def ratio_can_exceed_1(name, desc):
    text = f"{name} {desc}".lower()
    return "ratio of debit amount/credit amount" in text

def looks_like_count(name, desc):
    text = f"{name} {desc}".lower()
    return text.startswith("no. of") or "count" in text

def looks_like_amount(name, desc):
    text = f"{name} {desc}".lower()
    return ("amount" in text or "sum" in text or "upi" in text) and "per transaction" not in text

def looks_like_per_txn_avg(name, desc):
    text = f"{name} {desc}".lower()
    return "per transaction" in text

def looks_like_score(name, desc):
    text = f"{name} {desc}".lower()
    return "score" in text or "propensity" in text or "affluence" in text

def looks_like_vintage(name, desc):
    text = f"{name} {desc}".lower()
    return "vintage" in text or "days" in text

def outcome_guard_hit(name, desc):
    text = f"{name} {desc}".lower()
    return re.search("|".join(OUTCOME_GUARD_TERMS), text) is not None

def jaccard_from_numeric(x, y):
    # binarize feature: >0 => 1 else 0 (works for non-negative measurements)
    xb = (pd.to_numeric(x, errors="coerce").fillna(0) > 0).astype(int)
    ya = y.astype(int)
    inter = int(((xb == 1) & (ya == 1)).sum())
    union = int(((xb == 1) | (ya == 1)).sum())
    return inter/union if union > 0 else 0.0

def profile(df, desc_map):
    rows = []
    for c in df.columns:
        s = pd.to_numeric(df[c], errors="coerce")
        missing = s.isna().sum()
        n = len(s)
        fill_rate = 1.0 - missing / max(n,1)
        valid = s.dropna()
        uniq = valid.nunique()
        zfrac = float((valid == 0).mean()) if len(valid) else 0.0
        nfrac = float((valid < 0).mean()) if len(valid) else 0.0
        q = {}
        if len(valid):
            q = valid.quantile([0.01,0.05,0.25,0.5,0.75,0.95,0.99]).to_dict()
        rows.append({
            "variable": c,
            "description": desc_map.get(c, ""),
            "dtype": str(df[c].dtype),
            "missing_count": missing,
            "missing_pct": missing/max(n,1),
            "fill_rate": fill_rate,
            "unique_count": int(uniq),
            "frac_zero": zfrac,
            "frac_negative": nfrac,
            "min": float(valid.min()) if len(valid) else np.nan,
            "p01": float(q.get(0.01, np.nan)),
            "p05": float(q.get(0.05, np.nan)),
            "p25": float(q.get(0.25, np.nan)),
            "p50": float(q.get(0.5,  np.nan)),
            "p75": float(q.get(0.75, np.nan)),
            "p95": float(q.get(0.95, np.nan)),
            "p99": float(q.get(0.99, np.nan)),
            "max": float(valid.max()) if len(valid) else np.nan,
            "mean": float(valid.mean()) if len(valid) else np.nan,
            "std":  float(valid.std()) if len(valid) else np.nan,
            "skew": float(valid.skew()) if len(valid) else np.nan,
            "kurtosis": float(valid.kurtosis()) if len(valid) else np.nan,
        })
    return pd.DataFrame(rows)

def clip_quantiles(s, qlow, qhigh):
    if s.notna().sum() == 0:
        return s
    lo, hi = s.quantile([qlow, qhigh])
    return s.clip(lower=lo, upper=hi)

# -----------------------------
# Main
# -----------------------------
def main():
    print("=== Starting Deterministic Feature Engineering ===")
    print(f"Output directory: {OUTPUT_BASE_DIR}")
    
    # Read configs & inputs
    print("Loading configuration...")
    cfg = load_build_summary()
    fill_thr = float(cfg.get("fill_rate_threshold", FILL_RATE_THRESHOLD))
    print("Loading selected features...")
    selected = load_selected_feature_list()
    print("Loading guard set...")
    guard = load_guard_set()
    print("Loading raw data...")
    raw = load_raw()

    # Filter by fill rate
    selected = selected[selected["fill_rate"] >= fill_thr].copy()

    # Normalize description map
    desc_map = dict(zip(selected["variable"], selected.get("description", [""]*len(selected))))

    # Build label deterministically (union)
    y = load_label([], raw)
    y = y.astype(int)

    # Decide feature set (apply guard & identifier drop)
    dropped = {"guard": [], "identifier": [], "outcome_guard": [], "not_in_raw": []}
    keep = []
    for v in selected["variable"]:
        d = desc_map.get(v, "")
        if v not in raw.columns:
            dropped["not_in_raw"].append(v); continue
        if v in guard:
            dropped["guard"].append(v); continue
        if outcome_guard_hit(v, d):
            dropped["outcome_guard"].append(v); continue
        if is_identifier(v, d):
            dropped["identifier"].append(v); continue
        keep.append(v)

    X = raw[keep].copy()

    # Pre report
    report_pre = profile(X, desc_map)
    report_pre.to_csv(REPORT_PRE, index=False)

    # Transform map (recorded for transparency)
    tmap = {}
    X_tr = pd.DataFrame(index=X.index)

    for v in keep:
        s = pd.to_numeric(X[v], errors="coerce")
        d = desc_map.get(v, "")

        # Type routing
        if looks_like_score(v, d):
            # scores: median impute + robust scale (no clipping)
            s = s.fillna(s.median())
            scaler = RobustScaler()
            s2 = pd.Series(scaler.fit_transform(s.values.reshape(-1,1)).ravel(), index=s.index)
            X_tr[v] = s2
            tmap[v] = {"type":"score","impute":"median","scale":"robust"}
            continue

        if looks_like_vintage(v, d):
            s = s.fillna(s.median())
            # leave as is (you can toggle log1p via config)
            X_tr[v] = s
            tmap[v] = {"type":"vintage","impute":"median","scale":None}
            continue

        if looks_like_ratio_01(v, d):
            # [0,1] ratios
            s = s.fillna(0.0).clip(0.0, 1.0)
            X_tr[v] = clip_quantiles(s, *CLIP_QUANTILES)
            tmap[v] = {"type":"ratio_01","impute":"zero","clip":[CLIP_QUANTILES]}
            continue

        if ratio_can_exceed_1(v, d):
            # >0 but can be >1; can be 0 as well
            s = s.fillna(s.median())
            s = pd.Series(np.arcsinh(s.values), index=s.index)  # signed log
            X_tr[v] = clip_quantiles(s, *CLIP_QUANTILES)
            tmap[v] = {"type":"ratio_signed","impute":"median","transform":"asinh","clip":[CLIP_QUANTILES]}
            continue

        if looks_like_count(v, d):
            # counts are non-negative; zero is valid
            s = s.fillna(0.0)
            s = pd.Series(np.log1p(np.maximum(s, 0.0)), index=s.index)
            X_tr[v] = clip_quantiles(s, *CLIP_QUANTILES)
            tmap[v] = {"type":"count","impute":"zero","transform":"log1p","clip":[CLIP_QUANTILES]}
            continue

        if looks_like_per_txn_avg(v, d) or looks_like_amount(v, d):
            # amounts/per-txn: non-negative; treat zeros as zeros
            s = s.fillna(0.0)
            s = pd.Series(np.log1p(np.maximum(s, 0.0)), index=s.index)
            X_tr[v] = clip_quantiles(s, *CLIP_QUANTILES)
            tmap[v] = {"type":"amount","impute":"zero","transform":"log1p","clip":[CLIP_QUANTILES]}
            continue

        # fallback:
        # if can be negative (like surplus) do asinh; else log1p
        if s.min(skipna=True) < 0:
            s = s.fillna(s.median())
            s = pd.Series(np.arcsinh(s.values), index=s.index)
            X_tr[v] = clip_quantiles(s, *CLIP_QUANTILES)
            tmap[v] = {"type":"real_signed","impute":"median","transform":"asinh","clip":[CLIP_QUANTILES]}
        else:
            s = s.fillna(0.0)
            s = pd.Series(np.log1p(np.maximum(s, 0.0)), index=s.index)
            X_tr[v] = clip_quantiles(s, *CLIP_QUANTILES)
            tmap[v] = {"type":"real_nonneg","impute":"zero","transform":"log1p","clip":[CLIP_QUANTILES]}

    # Post report + null guard
    assert X_tr.isna().sum().sum() == 0, "Nulls remain after FE (unexpected)."
    report_post = profile(X_tr, desc_map)
    report_post.to_csv(REPORT_POST, index=False)

    # Leakage checks (guard + stats)
    rows = []
    for c in X_tr.columns:
        s = X_tr[c]
        # Spearman-like corr via pandas .corr(method="spearman") handles ties
        try:
            spearman = float(pd.Series(s).corr(y, method="spearman"))
        except Exception:
            spearman = np.nan
        # 1D AUC (if at least 2 classes and not constant)
        try:
            auc = roc_auc_score(y, s) if s.nunique() > 1 else np.nan
        except Exception:
            auc = np.nan
        # Jaccard on (>0) binarization
        try:
            jac = jaccard_from_numeric(s, y)
        except Exception:
            jac = np.nan

        flagged = False
        reason = []
        if c in guard:
            flagged, reason = True, ["guard"]
        if outcome_guard_hit(c, desc_map.get(c, "")):
            flagged, reason = True, reason + ["outcome_guard"]
        if not np.isnan(auc) and auc >= 0.92:
            flagged, reason = True, reason + [f"auc>0.92({auc:.3f})"]
        if not np.isnan(spearman) and abs(spearman) >= 0.60:
            flagged, reason = True, reason + [f"|spearman|>0.60({spearman:.3f})"]
        if not np.isnan(jac) and jac >= 0.70:
            flagged, reason = True, reason + [f"jaccard>0.70({jac:.3f})"]

        rows.append({
            "feature": c,
            "spearman_with_target": spearman,
            "auc_1d": auc,
            "jaccard_with_target": jac,
            "flagged": flagged,
            "flag_reasons": ";".join(reason)
        })
    pd.DataFrame(rows).sort_values(["flagged","auc_1d","spearman_with_target"],
                                   ascending=[False, False, False]).to_csv(LEAKAGE_CSV, index=False)

    # Persist artifacts
    X_tr.to_parquet(X_OUT, index=False)
    pd.DataFrame({"label": y}).to_csv(Y_OUT, index=False)

    dropped["fill_rate_lt_threshold"] = [v for v in selected["variable"] if v not in keep and v not in sum(dropped.values(), [])]
    Path(DROPPED_JSON).write_text(json.dumps(dropped, indent=2))
    Path(TRANSFORMS_JSON).write_text(json.dumps({
        "fill_rate_threshold": fill_thr,
        "clip_quantiles": CLIP_QUANTILES,
        "transforms": tmap
    }, indent=2))

    print(f"Features kept: {X_tr.shape[1]}, rows: {X_tr.shape[0]}")
    print(f"Artifacts written:\n- {X_OUT}\n- {Y_OUT}\n- {REPORT_PRE}\n- {REPORT_POST}\n- {LEAKAGE_CSV}\n- {TRANSFORMS_JSON}\n- {DROPPED_JSON}")

if __name__ == "__main__":
    main()
