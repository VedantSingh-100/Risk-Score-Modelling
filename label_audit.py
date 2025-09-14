# label_audit_pack.py
# Audits a provisional UNION label built from your 16 default-ish fields.
# Produces: label_union_provisional.csv, event_contribution_summary.csv,
#           jaccard_matrix.csv, weighted_label_tuning.csv,
#           do_not_use_features.txt, baseline_eval.json, label_audit_report.md

import re, json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

RAW = Path("50k_users_merged_data_userfile_updated_shopping.csv")
CAT = Path("variable_catalog.csv")
assert RAW.exists() and CAT.exists(), "Missing RAW or variable_catalog.csv."

# ---- Load catalog (for descriptions) + header
cat = pd.read_csv(CAT, low_memory=False)
name_col = "Variable" if "Variable" in cat.columns else cat.columns[0]
desc_col = next((c for c in cat.columns if "description" in c.lower()), None)
desc_map = dict(zip(cat[name_col].astype(str), cat[desc_col].fillna("").astype(str))) if desc_col else {}
hdr = pd.read_csv(RAW, nrows=0)

# ---- Your 16 policy severe columns + weights (from your label_policy.json)
POLICY_SEVERE_COLS = [
    "var501102","var501056","var501057","var501058","var501059","var501101",
    "var202077","var501055","var501060","var206063","var206064","var206065",
    "var206066","var501052","var501053","var501054"
]
WEIGHTS = {
    "var501102":0.9,"var501056":0.85,"var501057":0.85,"var501058":0.85,"var501059":0.85,
    "var501101":0.85,"var202077":0.85,"var501055":0.85,"var501060":0.75,"var206063":0.85,
    "var206064":0.85,"var206065":0.85,"var206066":0.85,"var501052":0.85,"var501053":0.85,"var501054":0.85
}

events = [c for c in POLICY_SEVERE_COLS if c in hdr.columns]
if not events:
    raise SystemExit("None of the policy severe columns exist in the raw file header.")

# ---- Load a safe sample (keeps runtime predictable)
N_ROWS = 20000
raw = pd.read_csv(RAW, usecols=events, nrows=N_ROWS, low_memory=False)

# ---- Binary event matrix (>0)
binE = pd.DataFrame(index=raw.index)
for c in events:
    s = raw[c]
    if s.dtype == "O":
        m = s.astype(str).str.lower().map({"true":1,"false":0,"yes":1,"no":0,"y":1,"n":0})
        v = m if m.notna().mean() > 0.8 else pd.to_numeric(s, errors="coerce")
    else:
        v = pd.to_numeric(s, errors="coerce")
    binE[c] = (v.fillna(0) > 0).astype(np.int8)

# ---- UNION label
y = (binE.sum(axis=1) > 0).astype(np.int8)
pd.DataFrame({"label_union": y}).to_csv("label_union_provisional.csv", index=False)

# ---- Contribution among positives
pos = y == 1
contrib = binE[pos].mean().sort_values(ascending=False)
pd.DataFrame({
    "event": contrib.index,
    "share_among_positives": contrib.values,
    "description": [desc_map.get(ev, "") for ev in contrib.index]
}).to_csv("event_contribution_summary.csv", index=False)

# ---- Jaccard overlap
def jaccard(a, b):
    A = (a.values == 1); B = (b.values == 1)
    inter = (A & B).sum(); union = (A | B).sum()
    return float(inter/union) if union>0 else 0.0
jac = pd.DataFrame(index=events, columns=events, dtype=float)
for i in range(len(events)):
    for j in range(i, len(events)):
        val = jaccard(binE.iloc[:, i], binE.iloc[:, j])
        jac.iloc[i, j] = val; jac.iloc[j, i] = val
jac.to_csv("jaccard_matrix.csv")

# ---- Fix & tune weighted label (choose threshold to hit target prevalence ~ union)
w = np.array([WEIGHTS.get(c, 0.6) for c in events])
sev_sum = (binE.values * w).sum(axis=1)
nonzero = sev_sum[sev_sum > 0]
target_prev = float(y.mean())  # match union prevalence
if len(nonzero) > 0:
    thr = float(np.quantile(nonzero, 1 - target_prev))
    y_weighted = (sev_sum >= thr).astype(np.int8)
else:
    thr = 1.0
    y_weighted = (sev_sum >= thr).astype(np.int8)
grid = np.linspace(0.2, max(1.0, w.sum()), 12)
wtune = pd.DataFrame({"threshold": grid, "prevalence": [(sev_sum >= t).mean() for t in grid]})
wtune.to_csv("weighted_label_tuning.csv", index=False)

# ---- Build leakage guard list
TOKENS = [r"default", r"dpd", r"overdue", r"arrear", r"write[\s-]?off", r"charge[\s-]?off",
          r"npa", r"settle", r"miss", r"min[_\s-]?due", r"over[-\s]?limit", r"declin",
          r"reject", r"bounced", r"nsf", r"negative"]
OUTCOME_RE = re.compile("|".join(TOKENS), re.IGNORECASE)
guard = set(events)
for var in hdr.columns:
    txt = (var + " " + desc_map.get(var, "")).lower()
    if OUTCOME_RE.search(txt):
        guard.add(var)
with open("do_not_use_features.txt", "w") as f:
    for g in sorted(guard):
        f.write(g + "\n")

# ---- Baseline model on allowed numeric features (quick, no leakage)
num_probe = pd.read_csv(RAW, nrows=1000).select_dtypes(include=[np.number]).columns.tolist()
allowed = [c for c in num_probe if c not in guard][:150]
baseline = {"n_features_used": 0, "auc_valid": None, "brier_valid": None}
if allowed:
    X = pd.read_csv(RAW, usecols=allowed, nrows=len(y))
    # drop constants
    nun = X.nunique(dropna=True); X = X.loc[:, nun > 1]
    if X.shape[1] >= 10 and y.sum() > 0:
        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
        imp = SimpleImputer(strategy="median")
        Xtr_imp = imp.fit_transform(Xtr); Xva_imp = imp.transform(Xva)
        scaler = StandardScaler(with_mean=False)
        Xtr_s = scaler.fit_transform(Xtr_imp); Xva_s = scaler.transform(Xva_imp)
        lr = LogisticRegression(max_iter=300, class_weight="balanced")
        lr.fit(Xtr_s, ytr)
        p = lr.predict_proba(Xva_s)[:,1]
        baseline = {"n_features_used": int(X.shape[1]),
                    "auc_valid": float(roc_auc_score(yva, p)),
                    "brier_valid": float(brier_score_loss(yva, p))}

# ---- Report
with open("label_audit_report.md","w") as f:
    f.write("# Label Audit & Validation (Provisional)\n\n")
    f.write(f"- Rows sampled: {len(y)}\n")
    f.write(f"- Events used in UNION: {len(events)}\n")
    f.write(f"- UNION prevalence: {float(y.mean()):.4f}\n")
    f.write(f"- Weighted label tuned threshold: {thr:.3f} → prevalence {float(y_weighted.mean()):.4f}\n\n")
    f.write("## Top contributors among label=1 (first 10)\n")
    for ev, share in contrib.head(10).items():
        f.write(f"- {ev}: {share:.3f}  |  {desc_map.get(ev,'')[:80]}\n")
    f.write("\n## Leakage guard\n")
    f.write(f"- Guarded features count: {len(guard)} (see do_not_use_features.txt)\n")
    f.write("\n## Baseline (numeric-only, excluding guarded)\n")
    f.write(json.dumps(baseline, indent=2))
print("Wrote label audit artifacts.")
