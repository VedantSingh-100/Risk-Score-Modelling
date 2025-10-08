# build_features_snapshot.py (Extended + Leakage Audit + Sanitation)
import argparse, json, re, sys, math
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTCOME_PAT = re.compile(
    r"(write[\s-]?off|charge[\s-]?off|npa|default|dpd|over[-\s]?limit|"
    r"over\s?due|overdue|past.?due|arrear|min[_\s-]?due|mindue|miss(?:ed|ing)?|"
    r"declin\w*|reject\w*|insufficient|penalt\w*|bounc\w*|ecs|nach|negative\w*)",
    re.I
)
LIFETIME_PAT = re.compile(r"\b(life\s*time|lifetime|ever|since\s*inception)\b", re.I)
ID_PAT = re.compile(
    r"(?:^|_|(?<![A-Za-z0-9]))(id|uuid|pan|aadhaar|account|application|app(?:lication)?id|lead|mobile|phone|msisdn|email(_?address|_?id)?)(?:_|$|(?!=[A-Za-z0-9]))",
    re.I
)
WINDOW_PATTERNS = {
    "1m": re.compile(r"\b(last\s*1\s*month|1m|30d)\b", re.I),
    "3m": re.compile(r"\b(last\s*3\s*month|3m|90d)\b", re.I),
    "6m": re.compile(r"\b(last\s*6\s*month|6m|180d)\b", re.I),
    "12m": re.compile(r"\b(last\s*12\s*month|12m|360d|1\s*year)\b", re.I),
    "lt": re.compile(r"\b(life\s*time|lifetime|ever|since\s*inception)\b", re.I),
}
TIME_LIKE_DROP_PAT = re.compile(
    r"(?:(refdate|asof|snapshot|pull|report|run)[-_]?(date|dt))|"
    r"\b(date|datetime|timestamp|time|ts)\b|"
    r"(^|_)(dow|weekday|week|woy|month|q(uarter)?|year|yr|day|hour|min(ute)?|sec(ond)?)(_|$)",
    re.I
)
TIME_LAG_KEEP_PAT = re.compile(r"(since|lag|delta|diff|tenure|vintage|age)", re.I)

def read_descriptions(xlsx_path: Path) -> dict:
    if not xlsx_path.exists(): return {}
    try:
        excel = pd.ExcelFile(xlsx_path)
        best_sheet = max(excel.sheet_names, key=lambda s: pd.read_excel(xlsx_path, sheet_name=s).shape[0])
        df = pd.read_excel(xlsx_path, sheet_name=best_sheet).copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        var_col = next((c for c in df.columns if any(k in c for k in ["variable","field","column","name"])), None)
        desc_col = next((c for c in df.columns if any(k in c for k in ["explanation","description","meaning","detail"])), None)
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

def infer_window(col: str, desc: str) -> str:
    s = f"{col} {desc}".lower()
    for tag, pat in WINDOW_PATTERNS.items():
        if pat.search(s): return tag
    return "unknown"

def is_window_aggregate(col: str, desc: str) -> bool:
    return infer_window(col, desc) in {"1m","3m","6m","12m","lt"}

def variable_family(varname: str) -> str:
    v = str(varname).lower()
    m = re.match(r"var(\d{3})", v)
    if m: return f"var{m.group(1)}"
    return v.split("_")[0]

def id_like_columns(cols) -> set:
    return {c for c in cols if ID_PAT.search(str(c))}

def near_unique_columns(df: pd.DataFrame, thresh: float = 0.98) -> set:
    out = set(); n = len(df)
    for c in df.columns:
        try:
            if df[c].nunique(dropna=False) >= thresh * n:
                out.add(c)
        except Exception:
            pass
    return out

def infer_numeric_kind(col: str, desc: str, s: pd.Series) -> str:
    name = f"{col} {desc}".lower()
    if any(k in name for k in ["count","no.","num","times","frequency","txn_cnt","#","occurrence","instances"]): return "count"
    if any(k in name for k in ["ratio","rate","pct","percent","utilization","dti","share","proportion"]): return "ratio"
    if any(k in name for k in ["amount","amt","balance","outstanding","income","expense","bill","limit","emi","loan"]): return "amount"
    x = pd.to_numeric(s, errors="coerce")
    if (x.dropna()>=0).mean() > 0.98 and (x.dropna() % 1 == 0).mean() > 0.95: return "count"
    if x.dropna().between(-5,5).mean() > 0.98: return "ratio"
    return "other"

def corr_prune_graph(df_num: pd.DataFrame, thr: float = 0.97, priority: pd.Series | None = None):
    if df_num.shape[1] <= 1: return df_num.columns.tolist(), []
    corr = df_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    edges = [(i,j) for i in upper.columns for j in upper.index if i!=j and upper.loc[j,i] >= thr]
    drop, kept = set(), set(df_num.columns)
    def score(c): return float(priority.get(c, 0.0)) if priority is not None else 0.0
    for i,j in edges:
        if i in drop or j in drop: continue
        loser = i if score(i) < score(j) else j
        drop.add(loser); kept.discard(loser)
    return sorted(list(kept)), sorted(list(drop))

def roc_auc_fast(y: np.ndarray, x: np.ndarray) -> float:
    y = y.astype(int)
    n1 = int(y.sum()); n0 = int((1 - y).sum())
    if n1 == 0 or n0 == 0: return 0.5
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(x)+1)
    _, inv, counts = np.unique(x[order], return_inverse=True, return_counts=True)
    csum = np.cumsum(counts); avg_ranks = (csum - counts/2.0 + 0.5)
    ranks[order] = avg_ranks[inv]
    sum_ranks_pos = ranks[y==1].sum()
    auc = (sum_ranks_pos - n1*(n1+1)/2.0) / (n1*n0)
    return float(auc)

def mutual_info_discrete(y: np.ndarray, x: np.ndarray, bins: int = 10) -> float:
    try:
        if len(np.unique(x)) > bins and not np.array_equal(x, x.astype(int)):
            q = np.quantile(x, np.linspace(0,1,bins+1)); q[0], q[-1] = -np.inf, np.inf
            x_b = np.digitize(x, q[1:-1])
        else:
            x_b = x
        xy = pd.crosstab(pd.Series(x_b), pd.Series(y), normalize=True)
        px = xy.sum(axis=1).values; py = xy.sum(axis=0).values
        mi = 0.0
        for i in range(xy.shape[0]):
            for j in range(xy.shape[1]):
                pxy = xy.iloc[i, j]
                if pxy > 0 and px[i] > 0 and py[j] > 0:
                    mi += pxy * math.log(pxy / (px[i]*py[j] + 1e-12) + 1e-12)
        return float(mi)
    except Exception:
        return 0.0

def ks_stat(y: np.ndarray, x: np.ndarray, bins: int = 50) -> float:
    try:
        x = x.astype(float); pos = x[y==1]; neg = x[y==0]
        qs = np.quantile(x[np.isfinite(x)], np.linspace(0,1,bins))
        def cdf(v, arr): return (arr[:,None] <= v[None,:]).mean(axis=0)
        Fpos = cdf(qs, pos.reshape(-1,1)) if len(pos) else np.zeros_like(qs)
        Fneg = cdf(qs, neg.reshape(-1,1)) if len(neg) else np.zeros_like(qs)
        return float(np.max(np.abs(Fpos - Fneg)))
    except Exception:
        return 0.0

def phi_with_label(y: np.ndarray, b: np.ndarray) -> float:
    A = (b > 0).astype(bool); B = (y > 0).astype(bool)
    n11 = int(np.sum(A & B)); n10 = int(np.sum(A & ~B))
    n01 = int(np.sum(~A & B)); n00 = int(np.sum(~A & ~B))
    num = n11 * n00 - n10 * n01
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if den == 0: return 0.0
    return float(num / np.sqrt(den))

def jaccard_with_label(y: np.ndarray, b: np.ndarray) -> float:
    A = (b > 0).astype(bool); B = (y > 0).astype(bool)
    inter = np.sum(A & B); union = np.sum(A | B)
    return float(inter/union) if union > 0 else 0.0

def drop_time_like_features(cols: list[str], desc_map: dict[str,str]):
    keep, dropped = [], []
    for c in cols:
        text = (str(c) + " " + desc_map.get(c, "")).lower()
        is_time_like = bool(TIME_LIKE_DROP_PAT.search(text))
        is_lag_like = bool(TIME_LAG_KEEP_PAT.search(text))
        is_window = is_window_aggregate(c, desc_map.get(c, ""))
        if is_time_like and not (is_lag_like or is_window): dropped.append(c)
        else: keep.append(c)
    return keep, dropped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", required=True)
    ap.add_argument("--dict-xlsx", required=True)
    ap.add_argument("--interim-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label-column", default="label_clustered")
    ap.add_argument("--fill-rate-min", type=float, default=0.85)
    ap.add_argument("--corr-thr", type=float, default=0.97)
    ap.add_argument("--forbid-outcome-like", action="store_true", default=True)
    ap.add_argument("--forbid-lifetime-features", action="store_true", default=True)
    ap.add_argument("--forbid-windows", default="lt")
    ap.add_argument("--ban-prefixes", default="refdate_")
    ap.add_argument("--drop-time-like", action="store_true", default=True)
    ap.add_argument("--scale", action="store_true", default=True)
    ap.add_argument("--audit", action="store_true", default=True)
    ap.add_argument("--auc-thr", type=float, default=0.90)
    ap.add_argument("--mi-thr", type=float, default=0.20)
    ap.add_argument("--iv-thr", type=float, default=1.50)
    ap.add_argument("--ks-thr", type=float, default=0.70)
    ap.add_argument("--jaccard-thr", type=float, default=0.95)
    ap.add_argument("--phi-thr", type=float, default=0.95)
    ap.add_argument("--oof-folds", type=int, default=5)
    ap.add_argument("--smell-sample", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto-drop-leaky", action="store_true", default=False)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    interim = Path(args.interim_dir)

    print("• Loading raw data...")
    df = pd.read_csv(args.raw_csv, low_memory=False)
    print(f" -> {df.shape[0]:,} rows × {df.shape[1]:,} cols")

    print("• Loading variable descriptions...")
    desc_map = read_descriptions(Path(args.dict_xlsx))

    comp_path = interim / "composite_labels.csv"
    if not comp_path.exists(): raise RuntimeError(f"composite_labels.csv not found in {interim}")
    labels_df = pd.read_csv(comp_path)
    if args.label_column not in labels_df.columns:
        raise RuntimeError(f"Label '{args.label_column}' not in composite_labels.csv. Available (sample): {list(labels_df.columns)[:10]}")
    if len(labels_df) != len(df): raise RuntimeError(f"Row count mismatch: labels {len(labels_df)} vs raw {len(df)}")

    y = labels_df[[args.label_column]].copy().rename(columns={args.label_column: "label"})
    y.to_csv(out / "y_label.csv", index=False)
    print(f"• y_label.csv written. Positives={int(y['label'].sum())}/{len(y)} ({y['label'].mean():.4%})")

    guard = set(); guard_reason = {}
    def add_guard(cols, reason):
        for c in cols:
            if c not in guard:
                guard.add(c); guard_reason[c] = reason

    # read existing guard file(s)
    guard_files = [interim / "do_not_use_features.txt", Path("do_not_use_features.txt")]
    used_guard_file = None
    for guard_file in guard_files:
        if guard_file.exists():
            ext = [ln.strip() for ln in guard_file.read_text().splitlines() if ln.strip()]
            add_guard(ext, f"external_guard_list:{guard_file.name}")
            used_guard_file = guard_file
    if used_guard_file:
        print(f"• Using guard file: {used_guard_file} ({sum(v.startswith('external_guard_list') for v in guard_reason.values())} entries)")

    add_guard(labels_df.columns.tolist(), "label_columns")

    src_path = interim / "label_sources_used.csv"; label_sources = []
    if src_path.exists():
        try:
            src_df = pd.read_csv(src_path)
            if 'label_source' in src_df.columns:
                label_sources = src_df['label_source'].astype(str).tolist()
            elif 'variable' in src_df.columns:
                label_sources = src_df['variable'].astype(str).tolist()
        except Exception as e:
            print(f"[WARN] Could not read label_sources_used.csv: {e}")

    if label_sources:
        add_guard(label_sources, "label_source_exact")
        # NOTE: family guarding was removed per your choice; not adding family here.

    add_guard(id_like_columns(df.columns), "id_like")
    add_guard(near_unique_columns(df), "near_unique")

    if args.forbid_outcome_like:
        out_like = [c for c in df.columns if is_outcome_like(c, desc_map.get(c, ""))]
        add_guard(out_like, "outcome_like_text")

    if args.forbid_lifetime_features:
        lt_like = [c for c in df.columns if is_lifetime_window(c, desc_map.get(c, ""))]
        add_guard(lt_like, "lifetime_window")

    forbid_windows = set(t.strip().lower() for t in args.forbid_windows.split(",") if t.strip())
    if forbid_windows:
        win_ban = [c for c in df.columns if infer_window(c, desc_map.get(c, "")) in forbid_windows]
        add_guard(win_ban, f"forbid_windows:{','.join(sorted(forbid_windows))}")

    ban_prefixes = [p.strip() for p in args.ban_prefixes.split(",") if p.strip()]
    if ban_prefixes:
        pref_ban = [c for c in df.columns if any(str(c).startswith(p) for p in ban_prefixes)]
        add_guard(pref_ban, f"ban_prefixes:{','.join(ban_prefixes)}")

    dropped_time_like = []
    if args.drop_time_like:
        cand0 = [c for c in df.columns if c not in guard]
        kept0, dropped_time_like = drop_time_like_features(cand0, desc_map)
        add_guard(dropped_time_like, "time_like_literal")

    cand = [c for c in df.columns if c not in guard]
    print(f"• Guard set size (all sources): {len(guard)}")
    inter_count = sum(1 for c in guard if c in df.columns)
    print(f"• Guard intersecting raw header: {inter_count}")
    print(f"• Candidates after guard: {len(cand)} / {df.shape[1]}")

    fr = 1.0 - df[cand].isna().mean()
    keep_fr = fr[fr >= args.fill_rate_min].index.tolist()
    drop_fr = [c for c in cand if c not in keep_fr]
    print(f"• Fill-rate ≥ {args.fill_rate_min:.2f}: kept {len(keep_fr)} / {len(cand)} (dropped {len(drop_fr)})")

    Xc = df[keep_fr]
    num_cols = Xc.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in keep_fr if c not in num_cols]

    print("• Encoding categoricals (OHE for low-card, freq for high-card)...")
    X_parts = []; cat_manifest = {}
    for c in cat_cols:
        s = Xc[c].astype("object").where(~Xc[c].isna(), "__MISSING__")
        nunq = s.nunique()
        if nunq <= 8:
            levels = s.value_counts().index.tolist()
            for val in levels:
                col_new = f"{c}__is_{str(val)[:30]}"
                X_parts.append((col_new, (s == val).astype(int)))
            cat_manifest[c] = {"encoding":"ohe","levels": levels}
        else:
            vc = s.value_counts(dropna=False, normalize=True)
            enc = s.map(vc).astype(float).fillna(0.0) # explicit float + fillna
            col_new = f"{c}__freq"
            X_parts.append((col_new, enc.astype(float)))
            cat_manifest[c] = {"encoding":"freq","mapping_top": dict(list(vc.items())[:50])}

    print("• Transforming numerics (log1p for counts; asinh otherwise) + median impute...")
    num_manifest = {}; X_num = pd.DataFrame(index=df.index)
    for c in num_cols:
        d = desc_map.get(c, ""); kind = infer_numeric_kind(c, d, Xc[c])
        x = pd.to_numeric(Xc[c], errors="coerce")
        med = x.median()
        if not np.isfinite(med): med = 0.0 # safe fallback
        miss = x.isna().mean()
        x = x.fillna(med)
        if kind == "count":
            shift = 0.0 if (x>=0).all() else (-float(x.min()))
            z = np.log1p(x + shift)
        else:
            z = np.arcsinh(x)
        X_num[c] = z.astype(float)
        num_manifest[c] = {"kind": kind, "impute": float(med), "missing_rate": float(miss)}

    scaler_meta = {}
    if args.scale and len(X_num.columns) > 0:
        med = X_num.median()
        iqr = X_num.quantile(0.75) - X_num.quantile(0.25)
        iqr = iqr.fillna(1.0); iqr_safe = iqr.replace(0, 1.0)
        X_num = (X_num - med) / iqr_safe
        scaler_meta = {"median": {k: float(v) for k, v in med.items()},
                       "iqr": {k: float(v) for k, v in iqr.items()}}

    X_list = [X_num] + [pd.DataFrame({name: series}) for name, series in X_parts]
    X_all = pd.concat(X_list, axis=1) if X_list else pd.DataFrame(index=df.index)

    print(f"• Correlation pruning at |r| ≥ {args.corr_thr:.2f} (numerics only)...")
    num_for_prune = list(X_num.columns)
    X_num_prune = X_num[num_for_prune]
    num_priority = X_num_prune.var().fillna(0.0)
    keep_num, drop_num = corr_prune_graph(X_num_prune, thr=args.corr_thr, priority=num_priority)

    X_final = pd.concat([X_num_prune[keep_num], X_all.drop(columns=X_num.columns, errors="ignore")], axis=1)

    # ===== Leakage AUDIT =====
    audit_rows = []
    plots_dir = Path(args.out_dir) / "plots"; plots_dir.mkdir(parents=True, exist_ok=True)
    if args.audit and "label" in y.columns and len(X_final.columns):
        print("• Running leakage audit on final matrix...")
        Y = y["label"].astype(int).to_numpy()
        print(" - Auditing missingness predictiveness...")
        miss_rows = []
        for c in keep_fr:
            s = df[c]
            mi = s.isna().astype(int).to_numpy()
            if mi.sum() == 0: 
                continue
            auc_miss = roc_auc_fast(Y, mi.astype(float))
            miss_rows.append((c, float(auc_miss)))
        miss_df = pd.DataFrame(miss_rows, columns=["column","auc_missing_indicator"])
        if not miss_df.empty:
            miss_df.to_csv(Path(args.out_dir)/"audit_missingness_auc.csv", index=False)

        print(" - Computing single-feature metrics (AUC/MI/IV/KS, Jaccard/Phi for binaries)...")
        auc_vals = []
        for c in X_final.columns:
            x = X_final[c].to_numpy()
            try:
                auc = roc_auc_fast(Y, x.astype(float))
            except Exception:
                auc = 0.5
            mi = mutual_info_discrete(Y, x)
            ks = ks_stat(Y, x)
            if set(np.unique(x)).issubset({0,1}):
                jac = jaccard_with_label(Y, x)
                phi = phi_with_label(Y, x)
            else:
                jac, phi = np.nan, np.nan
            try:
                q = np.quantile(x[np.isfinite(x)], np.linspace(0,1,11))
                q[0], q[-1] = -np.inf, np.inf
                xb = np.digitize(x, q[1:-1])
                tab = pd.crosstab(pd.Series(xb), pd.Series(Y), normalize=False)
                tab = tab.replace(0, 0.5)
                good = tab.get(0, pd.Series(0, index=tab.index)).astype(float)
                bad = tab.get(1, pd.Series(0, index=tab.index)).astype(float)
                good /= good.sum(); bad /= bad.sum()
                woe = np.log((bad / good).clip(1e-6, 1e6))
                iv = float(((bad - good) * woe).sum())
            except Exception:
                iv = 0.0

            auc_vals.append(auc)
            audit_rows.append({
                "feature": c, "auc": float(auc), "mi": float(mi), "ks": float(ks),
                "jaccard_vs_label": float(jac) if not np.isnan(jac) else None,
                "phi_vs_label": float(phi) if not np.isnan(phi) else None,
                "iv": float(iv),
                "is_binary": bool(set(np.unique(x)).issubset({0,1}))
            })

        audit_df = pd.DataFrame(audit_rows).sort_values("auc", ascending=False)
        audit_df.to_csv(Path(args.out_dir)/"leakage_audit_raw.csv", index=False)

        plt.figure(figsize=(8,5))
        plt.hist([v for v in auc_vals if np.isfinite(v)], bins=40)
        plt.title("Single-feature AUC distribution")
        plt.tight_layout(); plt.savefig(plots_dir/"single_feature_auc_hist.png"); plt.close()

        topN = audit_df.head(min(30, len(audit_df)))
        plt.figure(figsize=(max(8, 0.3*len(topN)),6))
        xi = np.arange(len(topN))
        plt.bar(xi, topN["auc"].values)
        plt.xticks(xi, topN["feature"].values, rotation=90)
        plt.title("Top features by AUC vs label")
        plt.tight_layout(); plt.savefig(plots_dir/"top_auc_features_bar.png"); plt.close()

        sus_mask = (
            (audit_df["auc"] >= args.auc_thr) |
            (audit_df["auc"] <= 1 - args.auc_thr) |
            (audit_df["mi"] >= args.mi_thr) |
            (audit_df["iv"] >= args.iv_thr) |
            (audit_df["ks"] >= args.ks_thr) |
            (audit_df["jaccard_vs_label"].fillna(0) >= args.jaccard_thr) |
            (audit_df["phi_vs_label"].abs().fillna(0) >= args.phi_thr)
        )
        suspicious = audit_df[sus_mask].copy()

        print(" - Checking near-exact matches...")
        exact_rows = []
        for _, r in suspicious.iterrows():
            c = r["feature"]; x = X_final[c].to_numpy()
            if set(np.unique(x)).issubset({0,1}):
                eq = float((x == Y).mean())
                exact_rows.append((c, eq))
        exact_df = pd.DataFrame(exact_rows, columns=["feature","equal_to_label_frac"])
        suspicious = suspicious.merge(exact_df, on="feature", how="left")
        suspicious["exactish_match"] = suspicious["equal_to_label_frac"].fillna(0) >= 0.995

        def simple_oof_auc(x_all: np.ndarray, y_all: np.ndarray, k: int, seed: int) -> float:
            n = len(y_all)
            rng = np.random.default_rng(seed)
            idx = np.arange(n); rng.shuffle(idx)
            folds = np.array_split(idx, k); aucs = []
            for i in range(k):
                tst = folds[i]
                auc_fold = roc_auc_fast(y_all[tst], x_all[tst].astype(float))
                aucs.append(auc_fold)
            return float(np.mean(aucs)) if len(aucs) else 0.5

        print(" - Running OOF AUC on suspicious features...")
        oof_rows = []
        for _, r in suspicious.iterrows():
            c = r["feature"]; x = X_final[c].to_numpy()
            try:
                oof = simple_oof_auc(x, Y, k=args.oof_folds, seed=args.seed)
            except Exception:
                oof = np.nan
            oof_rows.append((c, oof))
        oof_df = pd.DataFrame(oof_rows, columns=["feature","oof_auc"])
        suspicious = suspicious.merge(oof_df, on="feature", how="left")

        flags = []
        for _, r in suspicious.iterrows():
            reasons = []
            if r["auc"] >= args.auc_thr or r["auc"] <= 1 - args.auc_thr: reasons.append("single_feature_auc")
            if r["mi"] >= args.mi_thr: reasons.append("high_mi")
            if r["iv"] >= args.iv_thr: reasons.append("high_iv")
            if r["ks"] >= args.ks_thr: reasons.append("high_ks")
            if (pd.notna(r["jaccard_vs_label"]) and r["jaccard_vs_label"] >= args.jaccard_thr): reasons.append("jaccard_high")
            if (pd.notna(r["phi_vs_label"]) and abs(r["phi_vs_label"]) >= args.phi_thr): reasons.append("phi_high")
            if bool(r.get("exactish_match", False)): reasons.append("exactish_match")
            if pd.notna(r.get("oof_auc", np.nan)) and (r["oof_auc"] >= args.auc_thr or r["oof_auc"] <= 1 - args.auc_thr):
                reasons.append("oof_auc_confirms")
            flags.append((r["feature"], ";".join(reasons)))
        flag_df = pd.DataFrame(flags, columns=["feature","flag_reasons"])

        leakage_report = suspicious.merge(flag_df, on="feature", how="left").sort_values(["oof_auc","auc"], ascending=[False, False])
        leakage_report.to_csv(Path(args.out_dir)/"leakage_audit.csv", index=False)
        print(f" → leakage_audit.csv: {len(leakage_report)} suspicious feature(s) flagged")

        if args.auto_drop_leaky and len(leakage_report):
            drop_set = set(leakage_report["feature"].tolist())
            print(f" - Auto-dropping {len(drop_set)} leaky features from matrix")
            X_final = X_final[[c for c in X_final.columns if c not in drop_set]]

    # ===== FINAL SANITY & CLEANUP =====
    def _sanitize_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_cols_final = df.select_dtypes(include=[np.number]).columns
        if len(num_cols_final):
            med = df[num_cols_final].median().fillna(0.0)
            df[num_cols_final] = df[num_cols_final].fillna(med)
        df.fillna(0.0, inplace=True)
        for c in num_cols_final:
            df[c] = df[c].astype(np.float32)
        std = df[num_cols_final].std()
        const_cols = std.index[(std.fillna(0) <= 1e-12)].tolist()
        if const_cols:
            print(f"• Dropping {len(const_cols)} constant column(s).")
            df.drop(columns=const_cols, inplace=True)
            for cc in const_cols:
                drop_rows.append({"column": cc, "reason": "constant_feature"})
        arr = df.to_numpy()
        bad = int(np.isnan(arr).sum()) + int(np.isinf(arr).sum())
        if bad:
            print(f"[WARN] Residual NaN/Inf ({bad}) after sanitation; forcing to finite values.")
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            df.loc[:, :] = arr
        return df

    guard_df = pd.DataFrame([{"column": c, "reason": r} for c, r in sorted(guard_reason.items())])
    guard_df.to_csv(out / "guard_reasons.csv", index=False)

    drop_rows = []
    for c in drop_fr: drop_rows.append({"column": c, "reason": "low_fill_rate"})
    for c in dropped_time_like: drop_rows.append({"column": c, "reason": "time_like_literal"})
    for c in drop_num: drop_rows.append({"column": c, "reason": "corr_prune"})

    X_final = _sanitize_numeric_df(X_final)

    sanity = {"rows": int(X_final.shape[0]),"cols": int(X_final.shape[1]),
              "nan_count": int(np.isnan(X_final.to_numpy()).sum()),
              "inf_count": int(np.isinf(X_final.to_numpy()).sum())}
    (out / "fe_sanity_checks.json").write_text(json.dumps(sanity, indent=2))

    pd.DataFrame(drop_rows).to_csv(out / "drop_reasons.csv", index=False)
    X_final.to_parquet(out / "X_features.parquet", index=False)
    Path(out / "feature_keep_list.txt").write_text("\n".join(X_final.columns.tolist()))

    manifest = {
        "n_rows": int(len(df)),"label_column": args.label_column,"guards_used": int(len(guard)),
        "guard_buckets": {k: int(v) for k, v in pd.Series(list(guard_reason.values())).value_counts().sort_index().items()},
        "params": {
            "fill_rate_min": args.fill_rate_min,"corr_thr": args.corr_thr,
            "forbid_outcome_like": bool(args.forbid_outcome_like),"forbid_lifetime_features": bool(args.forbid_lifetime_features),
            "forbid_windows": sorted(list(forbid_windows)) if args.forbid_windows else [],
            "ban_prefixes": ban_prefixes,"drop_time_like": bool(args.drop_time_like),"scale": bool(args.scale),
            "auc_thr": args.auc_thr,"mi_thr": args.mi_thr,"iv_thr": args.iv_thr,"ks_thr": args.ks_thr,
            "jaccard_thr": args.jaccard_thr,"phi_thr": args.phi_thr,"oof_folds": args.oof_folds,
            "smell_sample": args.smell_sample,"auto_drop_leaky": bool(args.auto_drop_leaky),
        },
        "stats": {
            "raw_cols": int(df.shape[1]),"candidates_after_guard": int(len(cand)),
            "kept_after_fill_rate": int(len(keep_fr)), "num_before_prune": int(len(num_cols)),
            "num_after_prune": int(len(keep_num)), "cat_encoded_cols": int(X_final.shape[1] - len(keep_num)),
            "final_feature_count": int(X_final.shape[1])
        },
        "scaler_meta": scaler_meta,"categorical_manifest": {k: (v if isinstance(v, dict) else str(v)) for k,v in cat_manifest.items()},
        "numeric_manifest": num_manifest,
    }
    (out / "fe_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"• Wrote {out/'X_features.parquet'} with {X_final.shape[1]} columns.")
    if args.audit:
        print("• Leakage audit artifacts:")
        print(f" - {out/'leakage_audit.csv'} (flags + OOF confirmation)")
        print(f" - {out/'leakage_audit_raw.csv'} (AUC/MI/IV/KS per feature)")
        print(f" - {out/'audit_missingness_auc.csv'} (missingness AUC per original col)")
        print(f" - {plots_dir/'single_feature_auc_hist.png'}, {plots_dir/'top_auc_features_bar.png'}")
    print("✅ Feature engineering (snapshot) complete.")

if __name__ == "__main__":
    main()