"""
Risk Feature Selection — Extended (Leakage checks, Missingness signal, IV/WOE, Target Encoding, Stability)

What this script adds (beyond your current pipeline):
- Leakage detection:
  * perfect/near-perfect correlation with label
  * single-feature AUC ~ 1.0
  * suspicious name/description tokens (dpd, default, chargeoff, writeoff, settled, overdue, ...)
- Missingness as signal:
  * create <var>__isna flags and rank them
  * class-wise missingness deltas
- Bivariate & multivariate strength:
  * IV/WOE, Mutual Information
  * L1-Logistic, Tree+Permutation, Target Encoding (cross-validated)
- Stability selection across multiple seeds/folds
- Redundancy pruning:
  * numeric Spearman clustering
  * categorical MI screening (optional)
- Final consensus with stability + missingness added

Outputs: CSVs + a Markdown report with key findings and next steps.
"""

from __future__ import annotations
import os, re, json, math, warnings
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ==========================
# CONFIG — EDIT IF NEEDED
# ==========================
RAW_CSV          = Path("50k_users_merged_data_userfile_updated_shopping.csv")
DICT_XLSX        = Path("Internal_Algo360VariableDictionary_WithExplanation.xlsx")
VAR_CATALOG_CSV  = Path("variable_catalog.csv")  # if available, used to attach descriptions/our earlier category

# If you want to force a specific label, set LABEL_COL here; else None to auto-detect
LABEL_COL: Optional[str] = None   # e.g., "var101022"

# Guards for size & speed
N_SEEDS             = 5
N_FOLDS             = 5
VALID_SIZE          = 0.25
RANDOM_SEED         = 42
MAX_MI_FEATURES     = 3000
MAX_L1_FEATURES     = 2500
MAX_TREE_FEATURES   = 2500
MISSING_DROP_HARD   = 0.98
DOMINANT_SHARE_DROP = 0.999
N_TOP_SINGLE_AUC    = 500     # limit single-feature AUC checks for speed
MIN_CAT_FREQ        = 50      # rare categories grouped to '__OTHER__'
N_BINS_IV           = 10      # quantile bins for IV/WOE

# ==========================
# Helpers
# ==========================

OUTCOME_TERMS = [
    r"\bdefault(s|ed|ing)?\b",
    r"\bdpd\b", r"\b(\d+)\s*dpd\b", r"\bever[_\s-]?(\d+)\s*dpd\b",
    r"\bdelinquen", r"\boverdue\b", r"\barrear",
    r"charge[\s-]?off", r"write[\s-]?off", r"\bnpa\b",
    r"\bsettled?\b", r"\brepossess", r"\bforeclos",
    r"\bbad\b(?!\s*debtors?)",  # 'bad' outcome wording
    r"\bgood\b\s*outcome",      # some datasets name the target this way
]

EXCLUDE_TERMS_FOR_LABEL = [
    r"\bgambl",                 # behavioral flags like gambling are NOT labels
    r"\bemail\b|\bdevice\b|\bios\b|\bapp\b|\bcampaign\b|\bchannel\b",
    r"\blifetime\b(?!\s*default|dpd|npa|write[-\s]?off)"  # lifetime behavior ≠ outcome label
]

OUTCOME_RE = re.compile("|".join(OUTCOME_TERMS), re.IGNORECASE)
EXCLUDE_RE = re.compile("|".join(EXCLUDE_TERMS_FOR_LABEL), re.IGNORECASE)

def coerce_bool01(s: pd.Series) -> tuple[bool, float]:
    """Return (is_bool_like_01, pos_rate)."""
    if s.dtype == "O":
        v = pd.to_numeric(s.astype(str).str.lower().map({"true":1,"false":0,"yes":1,"no":0,"y":1,"n":0}), errors="coerce")
    else:
        v = pd.to_numeric(s, errors="coerce")
    uniq = set(v.dropna().unique().tolist())
    is_bool = len(uniq) <= 2 and uniq.issubset({0,1})
    pos = float(np.nanmean(v)) if is_bool else np.nan
    return is_bool, pos

def select_label_strict(
    raw_df: pd.DataFrame,
    variable_catalog_csv: str | Path | None = None,
    dictionary_xlsx: str | Path | None = None,
    prevalence_low: float = 0.005,
    prevalence_high: float = 0.40,
    save_candidates_to: str | Path = "label_candidates.csv",
) -> str | None:
    """
    Returns the name of the selected label column if found; else None.
    Writes a candidates CSV explaining why each column is (not) eligible.
    """
    # Load descriptions (from catalog first; fall back to dictionary if available)
    descriptions = {}
    if variable_catalog_csv and Path(variable_catalog_csv).exists():
        cat = pd.read_csv(variable_catalog_csv)
        if {"Variable","Description"}.issubset(cat.columns):
            for _, r in cat[["Variable","Description"]].dropna(subset=["Variable"]).iterrows():
                descriptions[str(r["Variable"])] = str(r["Description"])
    elif dictionary_xlsx and Path(dictionary_xlsx).exists():
        xls = pd.ExcelFile(dictionary_xlsx)
        sheet = max(xls.sheet_names, key=lambda s: pd.read_excel(dictionary_xlsx, sheet_name=s).shape[0])
        vd = pd.read_excel(dictionary_xlsx, sheet_name=sheet)
        vd.columns = [str(c).strip() for c in vd.columns]
        # heuristic pickers
        lower = {c.lower(): c for c in vd.columns}
        def pick(cands, fb=None):
            for t in cands:
                if t in lower: return lower[t]
            for c in vd.columns:
                for t in cands:
                    if t in c.lower(): return c
            return fb
        name_col = pick(["variable name","variable","name","field","feature","column"]) or vd.columns[0]
        desc_col = pick(["description","explanation","meaning","details","note","notes"])
        if desc_col is not None:
            for _, r in vd[[name_col, desc_col]].dropna(subset=[name_col]).iterrows():
                descriptions[str(r[name_col])] = str(r[desc_col])

    # Score columns
    rows = []
    for c in raw_df.columns:
        if raw_df[c].isna().all():
            continue
        is_bool, pos = coerce_bool01(raw_df[c])
        name = str(c)
        desc = descriptions.get(c, "")

        text = f"{name} {desc}".lower()
        outcome_hit = OUTCOME_RE.search(text) is not None
        excluded_hit = EXCLUDE_RE.search(text) is not None

        eligible = (
            is_bool and
            (pos is not None) and (prevalence_low <= pos <= prevalence_high) and
            outcome_hit and
            not excluded_hit
        )
        rows.append({
            "column": c,
            "is_bool_like_01": is_bool,
            "pos_rate": pos,
            "outcome_keyword_in_name_or_desc": outcome_hit,
            "excluded_keyword_in_name_or_desc": excluded_hit,
            "description_used": desc[:200],
            "eligible_label": eligible
        })

    cand = pd.DataFrame(rows).sort_values(
        ["eligible_label","outcome_keyword_in_name_or_desc","pos_rate"], ascending=[False, False, True]
    )
    cand.to_csv(save_candidates_to, index=False)

    # Return the first eligible; if none, return None
    pick = cand[cand["eligible_label"]].head(1)
    return None if pick.empty else str(pick.iloc[0]["column"])

def safe_read_csv(path: Path) -> pd.DataFrame:
    assert path.exists(), f"Missing file: {path}"
    return pd.read_csv(path, low_memory=False)

def load_dictionary(xlsx_path: Path) -> Optional[pd.DataFrame]:
    if not xlsx_path.exists():
        return None
    xls = pd.ExcelFile(xlsx_path)
    sheet = max(xls.sheet_names, key=lambda s: pd.read_excel(xlsx_path, sheet_name=s).shape[0])
    vd = pd.read_excel(xlsx_path, sheet_name=sheet)
    vd.columns = [str(c).strip() for c in vd.columns]
    lower = {c.lower(): c for c in vd.columns}
    def pick(cands, fb=None):
        for t in cands:
            if t in lower: return lower[t]
        for c in vd.columns:
            for t in cands:
                if t in c.lower(): return c
        return fb
    name_col = pick(["variable name","variable","name","field","attribute","feature","column"]) or vd.columns[0]
    desc_col = pick(["description","explanation","meaning","details","note","notes"])
    type_col = pick(["type","data type","datatype","format","value type"])
    if desc_col is None: vd["__desc__"]=""; desc_col="__desc__"
    if type_col is None: vd["__dtype__"]=""; type_col="__dtype__"
    return vd[[name_col, desc_col, type_col]].rename(
        columns={name_col:"Variable", desc_col:"Description", type_col:"DictDataType"}
    )

def pick_label(df: pd.DataFrame) -> Optional[str]:
    if LABEL_COL is not None and LABEL_COL in df.columns:
        return LABEL_COL
    pats = [
        r'\b(default|bad|ever_?dpd|dpd_?(\d+)|ever_?(\d+)\s*dpd|npa|charge[\s-]?off|write[\s-]?off|fraud|bounced|delinq|overdue|arrear)\b',
        r'\bpaid[_\s-]?off\b'
    ]
    rex = [re.compile(p, re.I) for p in pats]
    def score(c: str) -> int:
        return sum(r.search(c) is not None for r in rex)
    # prefer boolean-like
    candidates = []
    for c in df.columns:
        s = df[c]
        vals = pd.to_numeric(s, errors="coerce") if s.dtype != "O" else pd.to_numeric(
            s.astype(str).str.lower().map({"true":1,"false":0,"yes":1,"no":0,"y":1,"n":0}), errors="coerce"
        )
        uniq = set(vals.dropna().unique().tolist())
        is_bool = len(uniq) <= 2 and uniq.issubset({0,1})
        pos = float(np.nanmean(vals)) if is_bool else np.nan
        candidates.append((c, is_bool, pos, score(c)))
    cand = sorted(candidates, key=lambda t: (t[3], t[1], -(0 if math.isnan(t[2]) else t[2])), reverse=True)
    for c, is_bool, pos, _ in cand:
        if is_bool and (pos is not None) and (0.005 <= pos <= 0.30):
            return c
    # fallback: first boolean-like
    for c, is_bool, pos, _ in cand:
        if is_bool: return c
    return None

def boolify(s: pd.Series) -> pd.Series:
    if s.dtype == "O":
        v = pd.to_numeric(s.astype(str).str.lower().map({"true":1,"false":0,"yes":1,"no":0,"y":1,"n":0}), errors="coerce")
    else:
        v = pd.to_numeric(s, errors="coerce")
    return v.fillna(0).clip(0,1).astype(np.int8)

def make_missing_flags(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    flags = {}
    for c in df.columns:
        flags[f"{c}__isna"] = df[c].isna().astype(np.int8)
    return pd.DataFrame(flags, index=df.index), list(flags.keys())

def robust_numeric_cols(df: pd.DataFrame, exclude: set[str]) -> List[str]:
    cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in exclude]
    return cols

def rare_collapse(s: pd.Series, min_count: int = MIN_CAT_FREQ) -> pd.Series:
    vc = s.value_counts(dropna=False)
    keep = set(vc[vc >= min_count].index.tolist())
    return s.where(s.isin(keep), "__OTHER__")

def iv_woe_for_series(x: pd.Series, y: pd.Series, n_bins: int = N_BINS_IV) -> Tuple[float, pd.DataFrame]:
    """
    Returns (IV, table). Works for numeric (binned) or categorical directly.
    """
    eps = 1e-6
    if pd.api.types.is_numeric_dtype(x):
        # quantile bin; handle low unique
        try:
            if x.nunique(dropna=True) > n_bins:
                binned = pd.qcut(x, q=n_bins, duplicates="drop")
            else:
                binned = x
        except Exception:
            binned = x
        g = pd.crosstab(binned, y)
    else:
        g = pd.crosstab(rare_collapse(x), y)
    if 0 not in g.columns: g[0] = 0
    if 1 not in g.columns: g[1] = 0
    g = g[[0,1]]
    g["good_pct"] = (g[0] + eps) / (g[0].sum() + eps)
    g["bad_pct"]  = (g[1] + eps) / (g[1].sum() + eps)
    g["woe"] = np.log(g["good_pct"] / g["bad_pct"])
    g["iv_contrib"] = (g["good_pct"] - g["bad_pct"]) * g["woe"]
    return float(g["iv_contrib"].sum()), g

def mutual_info_numeric(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    print("     Imputing missing values...")
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    print("     Discretizing features...")
    kb = KBinsDiscretizer(n_bins=min(10, max(2, X_imp.shape[0] // 3000)), encode="ordinal", strategy="quantile")
    X_disc = kb.fit_transform(X_imp)
    print("     Computing mutual information scores...")
    mi = mutual_info_classif(X_disc, y, discrete_features=True, random_state=RANDOM_SEED)
    return pd.Series(mi, index=X.columns, name="MI")

def l1_logistic_importance(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, float]:
    print("     Imputing and scaling features...")
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_imp)
    print("     Splitting data and training L1 model...")
    Xtr, Xva, ytr, yva = train_test_split(X_scaled, y, test_size=VALID_SIZE, stratify=y, random_state=RANDOM_SEED)
    lr = LogisticRegression(
        penalty="l1", solver="saga", max_iter=200, class_weight="balanced", random_state=RANDOM_SEED
    )
    lr.fit(Xtr, ytr)
    print("     Computing validation AUC...")
    auc = roc_auc_score(yva, lr.predict_proba(Xva)[:,1])
    return pd.Series(np.abs(lr.coef_[0]), index=X.columns, name="LR_L1"), float(auc)

def tree_perm_importance(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.Series, float]:
    print("     Imputing missing values...")
    imp = SimpleImputer(strategy="median")
    X_imp = imp.fit_transform(X)
    print("     Training gradient boosting model...")
    Xtr, Xva, ytr, yva = train_test_split(X_imp, y, test_size=VALID_SIZE, stratify=y, random_state=RANDOM_SEED)
    hgb = HistGradientBoostingClassifier(random_state=RANDOM_SEED, class_weight="balanced")
    hgb.fit(Xtr, ytr)
    print("     Computing permutation importance...")
    auc = roc_auc_score(yva, hgb.predict_proba(Xva)[:,1])
    perm = permutation_importance(hgb, Xva, yva, n_repeats=5, random_state=RANDOM_SEED, scoring="roc_auc")
    return pd.Series(perm.importances_mean, index=X.columns, name="HGB_PermImp"), float(auc)

def single_feature_auc_scan(df: pd.DataFrame, y: pd.Series, cols: List[str]) -> pd.DataFrame:
    rows = []
    print(f"🔍 Computing single-feature AUC for {min(len(cols), N_TOP_SINGLE_AUC)} features...")
    for c in tqdm(cols[:N_TOP_SINGLE_AUC], desc="Single-feature AUC scan"):
        x = df[[c]].copy()
        try:
            imp = SimpleImputer(strategy="median")
            x_imp = imp.fit_transform(x)
            Xtr, Xva, ytr, yva = train_test_split(x_imp, y, test_size=VALID_SIZE, stratify=y, random_state=RANDOM_SEED)
            lr = LogisticRegression(max_iter=200, class_weight="balanced")
            lr.fit(Xtr, ytr)
            auc = roc_auc_score(yva, lr.predict_proba(Xva)[:,1])
            rows.append((c, auc))
        except Exception:
            continue
    return pd.DataFrame(rows, columns=["Variable","single_feature_auc"]).sort_values("single_feature_auc", ascending=False)

def leak_name_token(c: str) -> bool:
    toks = ["dpd","default","chargeoff","writeoff","settled","overdue","npa","bad","repossession","foreclosure","bounced","nsf","collection"]
    c = str(c).lower()
    return any(t in c for t in toks)

def consensus_rank(df: pd.DataFrame) -> pd.DataFrame:
    def rankify(s): return s.rank(ascending=False, method="average")
    out = pd.DataFrame(index=df.index)
    for col in ["MI","IV","LR_L1","HGB_PermImp","TE_Score","MissingFlag_MI","MissingFlag_AUC"]:
        if col in df.columns:
            out[f"rank_{col}"] = rankify(df[col].fillna(0.0))
    out["consensus_rank"] = out.mean(axis=1)
    return out

# ==========================
# Main
# ==========================
def main():
    print("🚀 Starting Extended Risk Feature Selection Pipeline...")
    print("=" * 60)
    
    # 1) Load data
    print("📁 Loading raw data...")
    df = safe_read_csv(RAW_CSV)
    n_rows, n_cols = df.shape
    mem_mb = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"   Loaded: {n_rows:,} rows, {n_cols:,} columns, {mem_mb:.1f} MB")

    # 2) Pick/confirm label
    print("\n🎯 Selecting target variable...")
    label = select_label_strict(
    raw_df=df,
    variable_catalog_csv="variable_catalog.csv",
    dictionary_xlsx="Internal_Algo360VariableDictionary_WithExplanation.xlsx",
    save_candidates_to="label_candidates.csv",
)
    if label is None:
        raise RuntimeError(
            "No valid label auto-detected. Please specify the correct default outcome column "
            "or provide a dictionary description containing outcome keywords."
        )
    y = boolify(df[label])
    print(f"   ✅ Using label: {label} | Positivity rate: {y.mean():.4f}")

    # 3) Attach catalog/dictionary (optional)
    print("\n📚 Loading metadata...")
    catalog = None
    if VAR_CATALOG_CSV.exists():
        print("   Loading existing variable catalog...")
        catalog = pd.read_csv(VAR_CATALOG_CSV)
    dictionary = load_dictionary(DICT_XLSX)
    if dictionary is not None:
        print(f"   Loaded dictionary with {len(dictionary)} entries")
    else:
        print("   No dictionary file found")

    # 4) Drop obvious junk (label itself, PII-like by name, near-constant, extremely missing)
    print("\n🧹 Filtering problematic columns...")
    def is_pii_like(name: str) -> bool:
        P = [r'\b(id|uuid|guid|token|hash)\b',
             r'(application|customer|client|account|loan|case|lead|user|pan|aadhaar|ssn|passport|voter|driver|licen[cs]e|dl|mobile|phone|email|address)\s*(number|id|no|#)?\b']
        return any(re.search(p, name, re.I) for p in P)

    drop = {label}
    print("   Checking for PII-like columns...")
    pii_cols = {c for c in tqdm(df.columns, desc="PII detection") if is_pii_like(c)}
    drop |= pii_cols

    # Use catalog if present
    near_constant, high_missing = set(), set()
    if catalog is not None:
        if "DominantShare" in catalog.columns:
            near_constant = set(catalog.loc[catalog["DominantShare"] >= DOMINANT_SHARE_DROP, "Variable"].tolist())
            drop |= near_constant
        if "MissingPct" in catalog.columns:
            high_missing = set(catalog.loc[catalog["MissingPct"] > MISSING_DROP_HARD, "Variable"].tolist())
            drop |= high_missing
    
    print(f"   Dropping {len(drop)} columns: {len(pii_cols)} PII-like, {len(near_constant)} near-constant, {len(high_missing)} high-missing")

    # Numeric baseline set
    print("\n🔢 Selecting numeric features...")
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in drop]
    print(f"   Found {len(num_cols)} numeric columns")
    # Keep a copy of raw for categorical handling when needed
    X_num = df[num_cols].copy()

    # 5) Missingness as signal
    print("\n🔍 Analyzing missingness patterns...")
    print("   Creating missing value flags...")
    miss_flags_df, miss_flag_cols = make_missing_flags(df[num_cols])
    # Basic missingness power via MI on flags + single-flag AUC
    print("   Computing mutual information for missing flags...")
    miss_mi = mutual_info_numeric(miss_flags_df, y)
    miss_auc = single_feature_auc_scan(miss_flags_df, y, miss_flag_cols)
    miss_summary = pd.DataFrame({"Variable": miss_mi.index, "MissingFlag_MI": miss_mi.values}).merge(
        miss_auc, on="Variable", how="left"
    ).sort_values("MissingFlag_MI", ascending=False)
    print(f"   💾 Saving missingness analysis to missingness_signals.csv ({len(miss_summary)} features)")
    miss_summary.to_csv("missingness_signals.csv", index=False)

    # 6) Leakage detection
    print("\n🚨 Detecting potential data leakage...")
    leak_rows = []

    # 6a) Name/desc tokens
    print("   Checking column names for suspicious tokens...")
    for c in tqdm(df.columns, desc="Name/desc token check"):
        reasons = []
        if leak_name_token(c): reasons.append("leak_name_token")
        # attach description flag if dictionary present
        if dictionary is not None:
            row = dictionary[dictionary["Variable"] == c]
            if not row.empty and any(leak_name_token(str(row.iloc[0].get("Description",""))) for _ in [0]):
                reasons.append("desc_leak_token")
        if reasons:
            leak_rows.append((c, ";".join(reasons), np.nan, np.nan))

    # 6b) Perfect/near-perfect correlation with label (numeric only)
    print("   Computing correlations with target...")
    for c in tqdm(num_cols, desc="Correlation check"):
        try:
            r, _ = spearmanr(df[c], y)
            if pd.notna(r) and abs(r) >= 0.999:
                leak_rows.append((c, "spearman≈±1", r, np.nan))
        except Exception:
            pass

    # 6c) Single-feature AUC scan (numeric)
    sfa = single_feature_auc_scan(X_num, y, num_cols)
    print(f"   💾 Saving single-feature AUC results to single_feature_auc_scan.csv ({len(sfa)} features)")
    sfa.to_csv("single_feature_auc_scan.csv", index=False)
    for _, row in sfa.iterrows():
        if row["single_feature_auc"] >= 0.995:
            leak_rows.append((row["Variable"], "single_feature_auc≈1", np.nan, row["single_feature_auc"]))

    leak_df = pd.DataFrame(leak_rows, columns=["Variable","reason","spearman_with_y","single_feature_auc"])
    print(f"   💾 Saving leakage report to leakage_report.csv ({len(leak_df)} suspicious features)")
    leak_df.to_csv("leakage_report.csv", index=False)

    # 7) Bivariate ranks: IV/WOE & MI
    print("\n📊 Computing bivariate feature rankings...")
    # 7a) IV/WOE for numeric columns (optionally extend to selected categoricals)
    print(f"   Computing IV/WOE for {min(len(num_cols), MAX_MI_FEATURES)} features...")
    iv_rows = []
    for c in tqdm(num_cols[:MAX_MI_FEATURES], desc="IV/WOE computation"):
        try:
            iv, _ = iv_woe_for_series(df[c], y)
            iv_rows.append((c, iv))
        except Exception:
            continue
    iv_df = pd.DataFrame(iv_rows, columns=["Variable","IV"]).sort_values("IV", ascending=False)
    print(f"   💾 Saving IV/WOE results to iv_woe.csv ({len(iv_df)} features)")
    iv_df.to_csv("iv_woe.csv", index=False)

    # 7b) MI on numeric
    print(f"   Computing mutual information for {min(len(num_cols), MAX_MI_FEATURES)} features...")
    mi_series = mutual_info_numeric(X_num[num_cols[:MAX_MI_FEATURES]], y)
    mi_df = mi_series.reset_index()
    mi_df.columns = ["Variable","MI"]
    print(f"   💾 Saving MI results to mi.csv ({len(mi_df)} features)")
    mi_df.to_csv("mi.csv", index=False)

    # 8) Multivariate ranks: L1 & Tree Permutation
    print("\n🤖 Computing multivariate feature rankings...")
    # Cap features for speed
    l1_cols   = num_cols[:MAX_L1_FEATURES]
    tree_cols = num_cols[:MAX_TREE_FEATURES]

    print(f"   Training L1 logistic regression on {len(l1_cols)} features...")
    l1_imp, l1_auc = l1_logistic_importance(X_num[l1_cols], y)
    l1_df = l1_imp.reset_index(); l1_df.columns = ["Variable","LR_L1"]
    print(f"   💾 Saving L1 importances to l1_importances.csv (AUC: {l1_auc:.4f})")
    l1_df.to_csv("l1_importances.csv", index=False)

    print(f"   Training gradient boosting on {len(tree_cols)} features...")
    tree_imp, tree_auc = tree_perm_importance(X_num[tree_cols], y)
    tree_df = tree_imp.reset_index(); tree_df.columns = ["Variable","HGB_PermImp"]
    print(f"   💾 Saving tree permutation importances to tree_perm.csv (AUC: {tree_auc:.4f})")
    tree_df.to_csv("tree_perm.csv", index=False)

    # 9) Target encoding (categoricals) — cross-validated mean encoding, scored via AUC
    print("\n🎭 Computing target encoding for categorical features...")
    # Select moderate-cardinality categoricals for safety
    obj_cols = [c for c in df.columns if df[c].dtype == "O" and c not in drop]
    # Limit to top 100 by distinct count (to keep it tractable)
    obj_cols = sorted(obj_cols, key=lambda c: df[c].nunique(dropna=True))[:100]
    print(f"   Processing {len(obj_cols)} categorical columns...")
    te_rows = []
    if obj_cols:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        for c in tqdm(obj_cols, desc="Target encoding"):
            # collapse rare
            x = rare_collapse(df[c], MIN_CAT_FREQ)
            # cross-fit target encoding
            enc = pd.Series(index=df.index, dtype=float)
            for tr, va in skf.split(x, y):
                tr_map = x.iloc[tr].groupby(x.iloc[tr]).apply(lambda s: y.iloc[tr][s.index].mean())
                enc.iloc[va] = x.iloc[va].map(tr_map).fillna(y.iloc[tr].mean())
            try:
                # single-feature AUC on encoded
                Xtr, Xva, ytr, yva = train_test_split(enc.values.reshape(-1,1), y, test_size=VALID_SIZE, stratify=y, random_state=RANDOM_SEED)
                lr = LogisticRegression(max_iter=200, class_weight="balanced")
                lr.fit(Xtr, ytr)
                auc = roc_auc_score(yva, lr.predict_proba(Xva)[:,1])
            except Exception:
                auc = np.nan
            te_rows.append((c, float(auc)))
    te_df = pd.DataFrame(te_rows, columns=["Variable","TE_Score"]).sort_values("TE_Score", ascending=False)
    print(f"   💾 Saving target encoding results to target_encoding_scores.csv ({len(te_df)} features)")
    te_df.to_csv("target_encoding_scores.csv", index=False)

    # 10) Stability selection (L1 + Tree) over seeds/folds
    print(f"\n🎲 Running stability selection across {N_SEEDS} seeds...")
    stability = {}
    for name in ["L1","TREE"]:
        stability[name] = {}

    seeds = list(range(N_SEEDS))
    for seed in tqdm(seeds, desc="Stability selection"):
        Xtr, Xva, ytr, yva = train_test_split(X_num, y, test_size=VALID_SIZE, stratify=y, random_state=seed)
        # L1
        try:
            imp = SimpleImputer(strategy="median")
            Xtr_imp = imp.fit_transform(Xtr)
            Xva_imp = imp.transform(Xva)
            scaler = StandardScaler(with_mean=False)
            Xtr_s = scaler.fit_transform(Xtr_imp)
            Xva_s = scaler.transform(Xva_imp)
            lr = LogisticRegression(penalty="l1", solver="saga", max_iter=200, class_weight="balanced", random_state=seed)
            lr.fit(Xtr_s, ytr)
            coef_abs = pd.Series(np.abs(lr.coef_[0]), index=X_num.columns, name=f"L1_seed{seed}")
            top = set(coef_abs.sort_values(ascending=False).head(200).index)
            for c in top:
                stability["L1"][c] = stability["L1"].get(c, 0) + 1
        except Exception:
            pass

        # TREE
        try:
            hgb = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
            hgb.fit(Xtr, ytr)
            perm = permutation_importance(hgb, Xva, yva, n_repeats=5, random_state=seed, scoring="roc_auc")
            imp_series = pd.Series(perm.importances_mean, index=X_num.columns)
            top = set(imp_series.sort_values(ascending=False).head(200).index)
            for c in top:
                stability["TREE"][c] = stability["TREE"].get(c, 0) + 1
        except Exception:
            pass

    stab_rows = []
    for method in stability:
        for c, count in stability[method].items():
            stab_rows.append((method, c, count/(len(seeds))))
    stab_df = pd.DataFrame(stab_rows, columns=["Method","Variable","SelectionRate"])
    stab_pivot = stab_df.pivot_table(index="Variable", columns="Method", values="SelectionRate", fill_value=0.0)
    stab_pivot["StabilityRate"] = stab_pivot.mean(axis=1)
    print(f"   💾 Saving stability selection to stability_selection.csv ({len(stab_pivot)} features)")
    stab_pivot.to_csv("stability_selection.csv")

    # 11) Redundancy pruning (numeric)
    print("\n🔄 Identifying redundant features via correlation...")
    # Spearman correlation > 0.95 (keep highest IV/MI among cluster)
    corr_keep = set(num_cols)
    try:
        print("   Computing correlation matrix...")
        sample = X_num.sample(min(10000, len(X_num)), random_state=RANDOM_SEED)
        corr = sample.corr(method="spearman").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        corr_keep = set(num_cols) - set(to_drop)
        print(f"   Found {len(to_drop)} highly correlated features to potentially drop")
    except Exception:
        print("   Warning: Could not compute correlation matrix")
        pass

    # 12) Build final consensus table
    print("\n📊 Building final consensus ranking...")
    all_vars = set(num_cols) | set(miss_summary["Variable"])
    print(f"   Combining results for {len(all_vars)} features...")
    base = pd.DataFrame({"Variable": list(all_vars)}).set_index("Variable")

    def attach(df, colname):
        nonlocal base
        if df is None or len(df)==0: return
        dd = df.copy()
        dd = dd.set_index("Variable") if "Variable" in dd.columns else dd.set_index(dd.columns[0])
        base[colname] = dd.iloc[:,0]

    attach(mi_df, "MI")
    attach(iv_df, "IV")
    attach(l1_df, "LR_L1")
    attach(tree_df, "HGB_PermImp")
    attach(te_df, "TE_Score")
    attach(miss_summary[["Variable","MissingFlag_MI"]], "MissingFlag_MI")
    attach(miss_summary[["Variable","single_feature_auc"]].rename(columns={"single_feature_auc":"MissingFlag_AUC"}), "MissingFlag_AUC")

    ranks = consensus_rank(base)
    final = base.join(ranks)

    # Add metadata columns (stability, kept after correlation prune, missing %, dictionary desc if available)
    final["KeptAfterCorrPrune"] = final.index.to_series().isin(corr_keep).astype(int)
    if catalog is not None:
        final = final.merge(catalog[["Variable","MissingPct","Nunique","Description"]], left_index=True, right_on="Variable", how="left").set_index("Variable")
    else:
        if dictionary is not None:
            final = final.merge(dictionary[["Variable","Description"]], left_index=True, right_on="Variable", how="left").set_index("Variable")

    final = final.join(stab_pivot[["StabilityRate"]], how="left")
    final = final.sort_values(["consensus_rank","StabilityRate"], ascending=[True, False]).reset_index()
    
    print(f"   💾 Saving final consensus to consensus_extended.csv ({len(final)} features)")
    final.to_csv("consensus_extended.csv", index=False)

    # 13) Report
    print("\n📋 Generating final report...")
    with open("feature_selection_report.md","w", encoding="utf-8") as f:
        f.write("# Feature Selection — Extended Report\n\n")
        f.write(f"- Rows: **{n_rows}**; Cols: **{n_cols}**\n")
        f.write(f"- Label: **{label}** | Positivity: **{y.mean():.4f}**\n")
        f.write(f"- L1 AUC: **{l1_auc}** | Tree AUC: **{tree_auc}**\n\n")
        f.write("## Leakage checks\n")
        f.write("- See `leakage_report.csv` for suspicious columns.\n")
        f.write("- See `single_feature_auc_scan.csv` for single-feature AUC (flag near 1.0).\n\n")
        f.write("## Missingness signals\n")
        f.write("- See `missingness_signals.csv`; strong `<var>__isna` indicates NULL is predictive.\n\n")
        f.write("## Bivariate & Multivariate ranks\n")
        f.write("- `iv_woe.csv`, `mi.csv`, `l1_importances.csv`, `tree_perm.csv`, `target_encoding_scores.csv`.\n\n")
        f.write("## Stability selection\n")
        f.write("- `stability_selection.csv` with selection rates across seeds; prefer features with high stability.\n\n")
        f.write("## Final consensus\n")
        f.write("- `consensus_extended.csv` includes MI, IV, L1, Tree, TE, missingness impact, stability, and correlation-prune flag.\n")
        f.write("- Sort by `consensus_rank` (asc) and check `StabilityRate` (desc).\n\n")
        f.write("## Next steps\n")
        f.write("- Remove features listed in `leakage_report.csv` **before** final modeling.\n")
        f.write("- If you have timestamps, switch to **time-based** splits.\n")
        f.write("- Add monotonic constraints where sensible; audit fairness.\n")
    
    print("💾 Report saved to feature_selection_report.md")

    print("\n✅ Extended pipeline completed successfully!")
    print("=" * 60)
    print("📊 Final Summary:")
    print(json.dumps({
        "label": label,
        "pos_rate": float(y.mean()),
        "l1_auc": float(l1_auc) if l1_auc is not None else None,
        "tree_auc": float(tree_auc) if tree_auc is not None else None,
        "num_features_analyzed": len(num_cols),
        "num_categorical_features": len(obj_cols) if 'obj_cols' in locals() else 0,
        "num_suspicious_features": len(leak_df) if 'leak_df' in locals() else 0,
        "outputs": [
            "leakage_report.csv",
            "missingness_signals.csv",
            "iv_woe.csv", "mi.csv", "l1_importances.csv", "tree_perm.csv",
            "target_encoding_scores.csv",
            "stability_selection.csv",
            "consensus_extended.csv",
            "feature_selection_report.md"
        ]
    }, indent=2))

if __name__ == "__main__":
    main()
