"""
Label Policy Wizard
-------------------
Turn multiple default-related fields into ONE training label.

Inputs:
- variable_catalog.csv  (has columns: Variable, Description, MissingPct, Nunique, etc.)
- 50k_users_merged_data_userfile_updated_shopping.csv  (raw rows)

Outputs:
- outcome_candidates.csv                 (candidates + stats)
- outcome_agreement_jaccard.csv         (pairwise agreement among candidates)
- labels_preview.csv                     (row-wise preview for a sample: union/hier/severity-wt)
- label_policy.json                      (documented mapping & chosen policy)
- label_variant_eval.csv                 (AUC/Brier for each label variant)
"""

import json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")

# ========== CONFIG ==========
CAT_PATH  = Path("variable_catalog.csv")
RAW_PATH  = Path("50k_users_merged_data_userfile_updated_shopping.csv")
SAMPLE_N  = 20000   # sample rows for speed; increase if you want
SEED      = 42

# Terms that indicate outcome-like fields in name or description
OUTCOME_TERMS = [
    r"\bdefault(s|ed|ing)?\b",
    r"\bdpd\b", r"\b(\d+)\s*dpd\b", r"\bever[_\s-]?(\d+)\s*dpd\b",
    r"\bdelinquen", r"\boverdue\b", r"\barrear",
    r"charge[\s-]?off", r"write[\s-]?off", r"\bnpa\b",
    r"\bsettled?\b", r"\brepossess", r"\bforeclos"
]
OUTCOME_RE = re.compile("|".join(OUTCOME_TERMS), re.IGNORECASE)

# Map of severity keywords -> score (higher = worse). Adjust if needed.
SEVERITY_WEIGHTS = {
    "writeoff": 1.00, "chargeoff": 1.00, "npa": 1.00,
    "90dpd": 0.90, "90": 0.90,
    "60dpd": 0.75, "60": 0.75,
    "30dpd": 0.60, "30": 0.60,
    "overdue": 0.40, "delinq": 0.40, "arrear": 0.40,
    "default": 0.85  # generic "default" gets high weight
}

def severity_from_name_desc(name: str, desc: str) -> float:
    text = f"{name} {desc}".lower()
    score = 0.0
    for k, w in SEVERITY_WEIGHTS.items():
        if k in text:
            score = max(score, w)
    return score

def coerce_bool01(s: pd.Series) -> tuple[bool, pd.Series]:
    """Return (is_boolean_like, numeric_01_series)"""
    if s.dtype == "O":
        m = s.astype(str).str.lower().map({"true":1,"false":0,"yes":1,"no":0,"y":1,"n":0})
        if m.notna().mean() > 0.8:
            v = m
        else:
            v = pd.to_numeric(s, errors="coerce")
    else:
        v = pd.to_numeric(s, errors="coerce")
    uniq = set(v.dropna().unique().tolist())
    is_bool = len(uniq) <= 2 and uniq.issubset({0,1})
    if is_bool:
        v = v.fillna(0).clip(0,1).astype(np.int8)
    return is_bool, v

def main():
    print("🧙 Starting Label Policy Wizard...")
    print("=" * 50)
    
    # 1) Load catalog and find candidate fields by name/description
    print("📊 Loading variable catalog...")
    cat = pd.read_csv(CAT_PATH, low_memory=False)
    name_col = "Variable" if "Variable" in cat.columns else cat.columns[0]
    desc_col = next((c for c in cat.columns if "description" in c.lower()), None)
    print(f"   Loaded catalog with {len(cat)} variables")

    print("\n🔍 Searching for outcome-related variables...")
    name_match = cat[name_col].astype(str).str.contains(OUTCOME_RE, regex=True, na=False)
    desc_match = cat[desc_col].astype(str).str.contains(OUTCOME_RE, regex=True, na=False) if desc_col else False
    cands = cat[name_match | desc_match].copy()
    print(f"   Found {len(cands)} potential outcome candidates")

    # 2) Keep candidates that exist in the raw file
    print("\n📁 Checking which candidates exist in raw data...")
    hdr = pd.read_csv(RAW_PATH, nrows=0)
    exists = cands[cands[name_col].isin(hdr.columns)].copy()
    print(f"   {len(exists)} candidates found in raw data columns")

    # 3) Load a sample with only those columns
    print(f"\n📥 Loading sample data ({SAMPLE_N:,} rows)...")
    use_cols = exists[name_col].tolist()
    print(f"   Loading {len(use_cols)} candidate columns...")
    raw = pd.read_csv(RAW_PATH, usecols=use_cols, nrows=SAMPLE_N, low_memory=False)
    mem_mb = raw.memory_usage(deep=True).sum() / (1024**2)
    print(f"   Loaded: {len(raw):,} rows, {len(raw.columns)} columns, {mem_mb:.1f} MB")

    # 4) Score each candidate: boolean-likeness, prevalence, missingness, inferred severity
    print("\n⚖️ Scoring candidate variables...")
    rows = []
    for _, r in tqdm(exists.iterrows(), total=len(exists), desc="Analyzing candidates"):
        col = r[name_col]
        desc = str(r[desc_col]) if desc_col else ""
        s = raw[col]
        is_bool, v = coerce_bool01(s)
        pos = float(v.mean()) if is_bool else np.nan
        miss = float(s.isna().mean())
        nun = int(s.nunique(dropna=True))
        sev = severity_from_name_desc(str(col), desc)
        rows.append({
            "Variable": col,
            "Description": desc[:220] if isinstance(desc, str) else "",
            "is_bool_like_01": is_bool,
            "sample_pos_rate": pos,
            "sample_missing_pct": miss,
            "sample_nunique": nun,
            "inferred_severity": sev
        })
    cand_stats = pd.DataFrame(rows).sort_values(
        ["is_bool_like_01","inferred_severity","sample_pos_rate"], ascending=[False, False, True]
    )
    print(f"   💾 Saving candidate analysis to outcome_candidates.csv ({len(cand_stats)} candidates)")
    cand_stats.to_csv("outcome_candidates.csv", index=False)

    # 5) Build binary versions for agreement / policy building
    print("\n🔄 Converting to binary format for policy analysis...")
    bin_df = pd.DataFrame(index=raw.index)
    meta = {}
    for _, r in tqdm(cand_stats.iterrows(), total=len(cand_stats), desc="Converting to binary"):
        c = r["Variable"]
        s = raw[c]
        is_bool, v = coerce_bool01(s)
        if not is_bool:
            # heuristic: consider non-boolean as positive if >0 (safe fallback)
            v = pd.to_numeric(s, errors="coerce")
            v = (v.fillna(0) > 0).astype(np.int8)
        bin_df[c] = v
        meta[c] = {
            "desc": str(r.get("Description","")),
            "severity": float(r.get("inferred_severity",0.0)),
        }
    print(f"   Created binary matrix: {len(bin_df)} rows × {len(bin_df.columns)} outcome variables")

    # 6) Pairwise agreement (Jaccard)
    print("\n🤝 Computing pairwise agreement (Jaccard similarity)...")
    def jaccard(a: pd.Series, b: pd.Series) -> float:
        A = a == 1; B = b == 1
        inter = (A & B).sum()
        union = (A | B).sum()
        return float(inter/union) if union>0 else 0.0

    cols = bin_df.columns.tolist()
    jac = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    n_pairs = len(cols) * (len(cols) - 1) // 2
    print(f"   Computing {n_pairs} pairwise comparisons...")
    
    with tqdm(total=n_pairs, desc="Jaccard similarity") as pbar:
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = jaccard(bin_df[cols[i]], bin_df[cols[j]])
                jac.iloc[i,j] = val; jac.iloc[j,i] = val
                pbar.update(1)
    
    print("   💾 Saving agreement matrix to outcome_agreement_jaccard.csv")
    jac.to_csv("outcome_agreement_jaccard.csv")

    # 7) Build three labels
    print("\n🏷️ Creating label variants...")
    # 7a) UNION: any severe event -> 1 (weight>=0.6 as default threshold for "severe")
    print("   Creating UNION label (any severe event)...")
    severe_cols = [c for c in cols if meta[c]["severity"] >= 0.6 or "dpd" in c.lower() or "write" in c.lower() or "charge" in c.lower() or "npa" in c.lower()]
    print(f"   Found {len(severe_cols)} severe outcome columns")
    label_union = (bin_df[severe_cols].sum(axis=1) > 0).astype(np.int8) if severe_cols else bin_df.any(axis=1).astype(np.int8)

    # 7b) HIERARCHICAL: pick highest-severity event present
    print("   Creating HIERARCHICAL label (highest severity first)...")
    sev_order = sorted(cols, key=lambda c: meta[c]["severity"], reverse=True)
    def hier_bad(row) -> int:
        for c in sev_order:
            if row[c] == 1:
                return 1
        return 0
    label_hier = bin_df.apply(hier_bad, axis=1).astype(np.int8)

    # 7c) SEVERITY-WEIGHTED ∈ [0,1]; recommend a cut at >=0.6 for "bad"
    print("   Creating SEVERITY-WEIGHTED label (weighted by outcome severity)...")
    sev_weights = np.array([meta[c]["severity"] for c in cols])
    sev_weights = np.where(sev_weights>0, sev_weights, 0.5)  # default mid-severity if unknown
    score = (bin_df.values * sev_weights).max(axis=1)  # max severity observed
    label_sev_wt = (score >= 0.6).astype(np.int8)
    
    labels = pd.DataFrame({"label_union": label_union, "label_hier": label_hier, "label_severity_wt": label_sev_wt})
    print(f"   💾 Saving label variants to labels_preview.csv ({len(labels)} rows)")
    labels.to_csv("labels_preview.csv", index=False)
    
    # Print label statistics
    for col in labels.columns:
        pos_rate = labels[col].mean()
        print(f"     {col}: {pos_rate:.4f} positive rate ({labels[col].sum():,} positives)")

    # 8) Quick evaluation of each label variant with a numeric-only baseline
    #    This is NOT the final model—just to compare label definitions.
    print("\n🧮 Evaluating label variants with baseline model...")
    num_cols_all = [c for c in hdr.columns if pd.api.types.is_numeric_dtype(pd.Series(dtype=hdr[c].dtype))]
    # If you want speed, limit to the first N numeric columns
    num_cols = num_cols_all[:250] if len(num_cols_all)>250 else num_cols_all
    print(f"   Using {len(num_cols)} numeric features for evaluation...")
    
    # Load a sample of those numeric features
    if num_cols:
        print("   Loading numeric features for baseline model...")
        X = pd.read_csv(RAW_PATH, usecols=num_cols, nrows=SAMPLE_N)
        print("   Preprocessing features (imputation + scaling)...")
        imp = SimpleImputer(strategy="median")
        X_imp = imp.fit_transform(X)
        scaler = StandardScaler(with_mean=False)
        X_scaled = scaler.fit_transform(X_imp)
    else:
        print("   Warning: No numeric columns found for evaluation")
        X_scaled = None

    eval_rows = []
    print("   Training baseline models for each label variant...")
    for lbl_name in tqdm(labels.columns, desc="Evaluating labels"):
        y = labels[lbl_name]
        # only evaluate if not all zeros
        if y.sum() == 0 or y.sum() == len(y): 
            eval_rows.append({"label_variant": lbl_name, "auc": np.nan, "brier": np.nan, "pos_rate": float(y.mean())})
            continue
        if X_scaled is None:
            eval_rows.append({"label_variant": lbl_name, "auc": np.nan, "brier": np.nan, "pos_rate": float(y.mean())})
            continue
        Xtr, Xva, ytr, yva = train_test_split(X_scaled, y, test_size=0.25, stratify=y, random_state=SEED)
        lr = LogisticRegression(max_iter=300, class_weight="balanced")
        lr.fit(Xtr, ytr)
        p = lr.predict_proba(Xva)[:,1]
        auc = roc_auc_score(yva, p)
        brier = brier_score_loss(yva, p)
        eval_rows.append({"label_variant": lbl_name, "auc": float(auc), "brier": float(brier), "pos_rate": float(y.mean())})

    eval_df = pd.DataFrame(eval_rows).sort_values(["auc","brier"], ascending=[False, True])
    print(f"   💾 Saving evaluation results to label_variant_eval.csv")
    eval_df.to_csv("label_variant_eval.csv", index=False)
    
    # Print evaluation summary
    print("\n📊 Label Variant Performance:")
    for _, row in eval_df.iterrows():
        auc_str = f"{row['auc']:.4f}" if not pd.isna(row['auc']) else "N/A"
        brier_str = f"{row['brier']:.4f}" if not pd.isna(row['brier']) else "N/A"
        print(f"     {row['label_variant']:20s} | AUC: {auc_str:>6s} | Brier: {brier_str:>6s} | Pos Rate: {row['pos_rate']:.4f}")

    # 9) Save policy JSON (you can edit this and "fix" the chosen variant later)
    print("\n📋 Generating policy documentation...")
    policy = {
        "candidates": [{"name": c, "severity": meta[c]["severity"], "description": meta[c]["desc"]} for c in cols],
        "policy_variants": {
            "label_union": {"type": "union", "severe_cols": severe_cols},
            "label_hier": {"type": "hierarchical", "order": sev_order},
            "label_severity_wt": {"type": "severity_weighted", "weights": {c: meta[c]["severity"] for c in cols}, "threshold": 0.6}
        },
        "evaluation": eval_rows
    }
    print("   💾 Saving policy configuration to label_policy.json")
    with open("label_policy.json","w") as f:
        json.dump(policy, f, indent=2)

    print("\n✅ Label Policy Wizard completed successfully!")
    print("=" * 50)
    print("📄 Generated Files:")
    output_files = [
        "outcome_candidates.csv",
        "outcome_agreement_jaccard.csv", 
        "labels_preview.csv",
        "label_variant_eval.csv",
        "label_policy.json"
    ]
    for filename in output_files:
        if Path(filename).exists():
            size = Path(filename).stat().st_size / 1024
            print(f"   ✓ {filename} ({size:.1f} KB)")
        else:
            print(f"   ✗ {filename} (not created)")
    
    print("\n🎯 Next Steps:")
    print("   1. Review outcome_candidates.csv for candidate quality")
    print("   2. Check outcome_agreement_jaccard.csv for overlapping definitions")
    print("   3. Compare label variants in label_variant_eval.csv")
    print("   4. Choose best label variant and update your pipeline accordingly")

if __name__ == "__main__":
    main()
