#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic feature engineering + label build + leakage guard for 56 selected variables.

Inputs (expected in working directory unless you pass args):
- Raw data: 50k_users_merged_data_userfile_updated_shopping.csv
- Selected features (with descriptions & fill rate): selected_features._finalcsv
  columns: variable,description,fill_rate
- Label sources (from sweep): Selected_label_sources.csv
  columns: variable, ...
- Guard set (optional): guard_Set.txt (one variable per line)
- Best config (optional, for documentation): build_summary.json

Outputs (folder: engineered/):
- feature_engineering_report_pre.csv
- feature_engineering_report_post.csv
- leakage_check.csv
- feature_list_final.csv
- X_features.parquet
- y_label.csv
- transforms_config.json
- label_stats.json
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------- Configuration (deterministic defaults) ----------
FILL_RATE_THRESHOLD = 0.85
CLIP_LOWER_Q = 0.01
CLIP_UPPER_Q = 0.99
SKEW_LOG_THRESHOLD = 2.0
ONE_HOT_MAX_UNIQUE = 12
LEAK_JACCARD_THRESHOLD = 0.85  # report-only by default
DROP_IDENTIFIER_LIKE = True    # drop description patterns indicating IDs (account no., identifier, etc.)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
pd.options.mode.copy_on_write = True


def read_guard_set(path: Path) -> set:
    guard = set()
    if path.exists():
        for ln in path.read_text().splitlines():
            ln = ln.strip()
            if ln:
                guard.add(ln)
    return guard


def compute_fill_rate(s: pd.Series) -> float:
    return float(s.notna().mean())


def jaccard_bool(a: np.ndarray, b: np.ndarray) -> float:
    # expects boolean arrays
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def is_nonnegative(series: pd.Series) -> bool:
    try:
        x = pd.to_numeric(series, errors="coerce")
        q1 = x.quantile(0.01)
        return bool((q1 >= -1e-12) or (x.min(skipna=True) >= -1e-12))
    except Exception:
        return False


def looks_like_ratio01(series: pd.Series, desc: str) -> bool:
    # heuristic: if description says 'ratio' OR 98th percentile ≤ 1.0 (and min ≥ 0)
    desc_low = (desc or "").lower()
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() == 0:
        return False
    p98 = x.quantile(0.98)
    mn = x.min(skipna=True)
    return ("ratio" in desc_low) or (mn >= -1e-12 and p98 <= 1.0 + 1e-9)


def is_identifier_like(var: str, desc: str) -> bool:
    if not DROP_IDENTIFIER_LIKE:
        return False
    text = f"{var} {desc}".lower()
    pat = re.compile(r"(account\s*number|acc(?:ount)?\s*no\.?|identifier|id\b|aadhaar|pan\s*no|ifsc|upi\s*id|card\s*number)")
    return bool(pat.search(text))


def summarize(df: pd.DataFrame, desc_map: dict) -> pd.DataFrame:
    rows = []
    print("Computing feature statistics...")
    for col in tqdm(df.columns, desc="Summarizing features"):
        s = df[col]
        is_num = pd.api.types.is_numeric_dtype(s)
        n = len(s)
        miss = int(s.isna().sum())
        fill = 1.0 - (miss / n if n else 0)
        uniq = int(s.nunique(dropna=True))
        zeros = float((s == 0).mean()) if is_num else np.nan
        negs = float((s < 0).mean()) if is_num else np.nan
        if is_num:
            q = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
            mean = float(s.mean())
            std = float(s.std())
            skew = float(s.skew())
            kurt = float(s.kurtosis())
            rows.append(dict(
                variable=col,
                description=desc_map.get(col, ""),
                dtype=str(s.dtype),
                missing_count=miss,
                missing_pct=miss / n if n else np.nan,
                fill_rate=fill,
                unique_count=uniq,
                frac_zero=zeros,
                frac_negative=negs,
                min=float(q.loc[0.00]),
                p01=float(q.loc[0.01]),
                p05=float(q.loc[0.05]),
                p25=float(q.loc[0.25]),
                p50=float(q.loc[0.50]),
                p75=float(q.loc[0.75]),
                p95=float(q.loc[0.95]),
                p99=float(q.loc[0.99]),
                max=float(q.loc[1.00]),
                mean=mean,
                std=std,
                skew=skew,
                kurtosis=kurt,
            ))
        else:
            rows.append(dict(
                variable=col,
                description=desc_map.get(col, ""),
                dtype=str(s.dtype),
                missing_count=miss,
                missing_pct=miss / n if n else np.nan,
                fill_rate=fill,
                unique_count=uniq,
                frac_zero=np.nan,
                frac_negative=np.nan,
                min=np.nan, p01=np.nan, p05=np.nan, p25=np.nan, p50=np.nan, p75=np.nan, p95=np.nan, p99=np.nan, max=np.nan,
                mean=np.nan, std=np.nan, skew=np.nan, kurtosis=np.nan,
            ))
    return pd.DataFrame(rows)


def engineer_features(df: pd.DataFrame, desc_map: dict, outdir: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Returns:
        X_final (DataFrame), transforms_config (dict), post_summary (DataFrame)
    """
    transforms = {}
    X = df.copy()

    # Track original dtypes for reporting
    orig_dtypes = {c: str(X[c].dtype) for c in X.columns}

    # Separate numeric vs non-numeric
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    # --- Numeric: impute, clip, optional log1p ---
    print("Processing numeric features...")
    for c in tqdm(num_cols, desc="Numeric engineering"):
        s = pd.to_numeric(X[c], errors="coerce")
        # impute with median
        med = float(s.median(skipna=True)) if s.notna().any() else 0.0
        s = s.fillna(med)

        # clip
        lo = float(s.quantile(CLIP_LOWER_Q))
        hi = float(s.quantile(CLIP_UPPER_Q))
        s = s.clip(lower=lo, upper=hi)

        # decide log1p
        apply_log = False
        if is_nonnegative(s) and (float(s.skew()) > SKEW_LOG_THRESHOLD) and (not looks_like_ratio01(s, desc_map.get(c, ""))):
            s = np.log1p(s)
            apply_log = True

        X[c] = s

        transforms[c] = dict(
            kind="numeric",
            original_dtype=orig_dtypes.get(c),
            impute="median",
            impute_value=med,
            clip=[CLIP_LOWER_Q, CLIP_UPPER_Q, lo, hi],
            log1p=apply_log,
        )

    # --- Categorical: one-hot or frequency encode ---
    # (In your 56 this is unlikely, but we keep it robust.)
    one_hot_frames = []
    drop_these = []
    if cat_cols:
        print("Processing categorical features...")
        for c in tqdm(cat_cols, desc="Categorical engineering"):
            s = X[c].astype("string")
            # impute most frequent
            if s.notna().any():
                mode_val = s.mode(dropna=True)
                fill_val = None if mode_val.empty else str(mode_val.iloc[0])
            else:
                fill_val = "missing"
            s = s.fillna(fill_val)

            n_unique = int(s.nunique(dropna=False))
            if n_unique <= ONE_HOT_MAX_UNIQUE:
                dummies = pd.get_dummies(s, prefix=c, dtype=np.uint8)
                one_hot_frames.append(dummies)
                drop_these.append(c)
                transforms[c] = dict(
                    kind="categorical_onehot",
                    original_dtype=orig_dtypes.get(c),
                    impute="mode",
                    impute_value=fill_val,
                    categories=list(sorted(s.unique()))
                )
            else:
                # frequency encode (relative freq)
                freq = (s.value_counts(dropna=False) / len(s)).to_dict()
                X[c] = s.map(freq).astype(float)
                transforms[c] = dict(
                    kind="categorical_freq",
                    original_dtype=orig_dtypes.get(c),
                    impute="mode",
                    impute_value=fill_val,
                    mapping_size=len(freq)
                )

    if one_hot_frames:
        X = pd.concat([X.drop(columns=drop_these, errors="ignore")] + one_hot_frames, axis=1)

    # --- Final assertions: no NaNs
    assert not X.isna().any().any(), "Unexpected NaNs remained after engineering."

    # --- Post summary
    post_summary = summarize(X, {c: desc_map.get(c, "") for c in X.columns})

    # Persist transforms for audit
    (outdir / "transforms_config.json").write_text(json.dumps(transforms, indent=2))

    return X, transforms, post_summary


def leakage_guard_report(features_df: pd.DataFrame,
                         label: pd.Series,
                         desc_map: dict,
                         label_sources: set,
                         guard_set: set) -> pd.DataFrame:
    """
    Report possible leakage. Does not drop by default.
    """
    rows = []
    outcome_pat = re.compile(r"(default|dpd|over\s?due|overdue|arrear|write[\s-]?off|charge[\s-]?off|miss(?:ed|ing)?|min[_\s-]?due|over[-\s]?limit|declin|reject|bounced|nsf|negative)",
                             re.IGNORECASE)

    y = (label.astype(int) == 1).values

    print("Computing leakage analysis...")
    for c in tqdm(features_df.columns, desc="Leakage check"):
        s = features_df[c]
        # compute Jaccard on a binarized version of the **original** signal logic (>0)
        if pd.api.types.is_numeric_dtype(s):
            b = (s > 0).astype(bool).values
        else:
            # for any encoded categorical (e.g., one-hot), treat non-zero as True
            b = (s.astype(str) != "0").astype(bool).values

        j = jaccard_bool(b, y)

        reasons = []
        if c in label_sources:
            reasons.append("is_label_source")
        if c in guard_set:
            reasons.append("in_guard_set")
        if outcome_pat.search(desc_map.get(c, "") + " " + c):
            reasons.append("outcome_term_match")
        if j >= LEAK_JACCARD_THRESHOLD:
            reasons.append(f"high_jaccard>={LEAK_JACCARD_THRESHOLD:.2f}")

        rows.append(dict(
            feature=c,
            description=desc_map.get(c, ""),
            jaccard_with_target=j,
            flagged=bool(reasons),
            reasons=";".join(reasons) if reasons else ""
        ))

    return pd.DataFrame(rows).sort_values("jaccard_with_target", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="50k_users_merged_data_userfile_updated_shopping.csv")
    ap.add_argument("--selected-features", default="selected_features._finalcsv")
    ap.add_argument("--label-sources", default="Selected_label_sources.csv")
    ap.add_argument("--guard", default="guard_Set.txt")
    ap.add_argument("--build-summary", default="build_summary.json")
    ap.add_argument("--outdir", default="engineered")
    ap.add_argument("--fill-threshold", type=float, default=FILL_RATE_THRESHOLD)
    ap.add_argument("--auto-drop-high-leakage", action="store_true",
                    help="If set, drops features with Jaccard>=LEAK_JACCARD_THRESHOLD")
    args = ap.parse_args()

    print("=== Deterministic Feature Engineering Pipeline ===")
    
    # Setup progress tracking for main steps
    main_steps = [
        "Loading inputs and configuration",
        "Building target labels",
        "Selecting and filtering features", 
        "Computing pre-engineering statistics",
        "Engineering features",
        "Computing post-engineering statistics",
        "Running leakage analysis",
        "Saving final outputs"
    ]
    
    main_progress = tqdm(main_steps, desc="Overall Progress")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load inputs
    print("Loading raw data...")
    df = pd.read_csv(args.data, low_memory=False)
    print(f"   Data: {df.shape[0]:,} rows x {df.shape[1]:,} columns")

    print("Loading selected features (with descriptions & fill rate)...")
    sel = pd.read_csv(args.selected_features)
    assert {"variable", "description", "fill_rate"}.issubset(sel.columns), "selected_features file must have variable, description, fill_rate"
    # Normalize column names to expected
    sel["variable"] = sel["variable"].astype(str)

    # Filter by fill rate threshold (deterministic)
    sel_filt = sel[sel["fill_rate"].astype(float) >= args.fill_threshold].copy()
    desc_map = dict(zip(sel_filt["variable"], sel_filt["description"]))

    print(f"   Selected by fill rate ≥ {args.fill_threshold:.2f}: {len(sel_filt)} features (from {len(sel)})")

    print("Loading label sources (from sweep)...")
    labsrc_df = pd.read_csv(args.label_sources)
    assert "variable" in labsrc_df.columns, "label sources CSV must have a 'variable' column"
    label_sources = set(labsrc_df["variable"].astype(str).tolist())
    print(f"   Label sources: {len(label_sources)}")

    # Optional: best config (for provenance only)
    if Path(args.build_summary).exists():
        try:
            cfg = json.loads(Path(args.build_summary).read_text())
            (outdir / "best_config_used.json").write_text(json.dumps(cfg, indent=2))
            print("   Saved best_config_used.json for provenance.")
        except Exception:
            pass

    # Guard
    guard_set = read_guard_set(Path(args.guard))
    print(f"Guard set loaded: {len(guard_set)} columns")
    main_progress.update(1)

    # --- Step 2: Build target label = union( label sources )
    print("Building target labels...")
    missing_label_cols = [c for c in label_sources if c not in df.columns]
    if missing_label_cols:
        raise RuntimeError(f"Missing label source columns in data: {missing_label_cols[:10]} ...")
    y = (df[list(label_sources)].fillna(0).astype(float) > 0).any(axis=1).astype(np.uint8)
    label_stats = {
        "positives": int(y.sum()),
        "negatives": int((1 - y).sum()),
        "prevalence": float(y.mean())
    }
    (outdir / "y_label.csv").write_text("label\n" + "\n".join(map(str, y.tolist())))
    (outdir / "label_stats.json").write_text(json.dumps(label_stats, indent=2))
    print(f"Label built (union): prevalence={label_stats['prevalence']:.4f} (pos={label_stats['positives']})")
    main_progress.update(1)

    # --- Step 3: Choose feature set from sel_filt, enforce guards deterministically
    print("Applying feature selection filters...")
    features = []
    dropped = {"missing_in_data": [], "in_label_sources": [], "in_guard_set": [], "identifier_like": []}
    for var, desc in tqdm(zip(sel_filt["variable"], sel_filt["description"]), 
                          desc="Filtering features", total=len(sel_filt)):
        if var not in df.columns:
            dropped["missing_in_data"].append(var); continue
        if var in label_sources:
            dropped["in_label_sources"].append(var); continue
        if var in guard_set:
            dropped["in_guard_set"].append(var); continue
        if is_identifier_like(var, desc):
            dropped["identifier_like"].append(var); continue
        features.append(var)

    if not features:
        raise RuntimeError("No features left after guard/filters. Adjust rules or inputs.")

    # Persist the chosen/dropped sets
    final_list = pd.DataFrame({"variable": features, "description": [desc_map.get(v, "") for v in features]})
    final_list.to_csv(outdir / "feature_list_final.csv", index=False)
    (outdir / "dropped_features.json").write_text(json.dumps(dropped, indent=2))
    print(f"Final features: {len(features)}  (dropped: { {k: len(v) for k,v in dropped.items()} })")
    main_progress.update(1)

    # --- Step 4: Pre summary
    pre_summary = summarize(df[features], desc_map)
    pre_summary.to_csv(outdir / "feature_engineering_report_pre.csv", index=False)
    main_progress.update(1)

    # --- Step 5: Engineer features deterministically
    print("Running feature engineering pipeline...")
    X_final, transforms_config, post_summary = engineer_features(df[features], desc_map, outdir)
    main_progress.update(1)
    
    # --- Step 6: Post summary
    post_summary.to_csv(outdir / "feature_engineering_report_post.csv", index=False)
    main_progress.update(1)

    # --- Step 7: Leakage guard report on the **pre-processed raw features** vs label
    leak_report = leakage_guard_report(df[features], y, desc_map, label_sources, guard_set)
    leak_report.to_csv(outdir / "leakage_check.csv", index=False)

    # Optional auto-drop high-leakage (report-only by default)
    if args.auto_drop_high_leakage:
        high_leak = leak_report.loc[leak_report["jaccard_with_target"] >= LEAK_JACCARD_THRESHOLD, "feature"].tolist()
        if high_leak:
            print(f"Auto-dropping {len(high_leak)} high-leak features:", high_leak[:10], "...")
            X_final = X_final.drop(columns=[c for c in high_leak if c in X_final.columns], errors="ignore")
            # Update post summary after drop
            print("Recomputing post-engineering statistics after leakage filtering...")
            post_summary = summarize(X_final, {c: desc_map.get(c, "") for c in X_final.columns})
            post_summary.to_csv(outdir / "feature_engineering_report_post.csv", index=False)
    main_progress.update(1)

    # --- Step 8: Final outputs
    print("Saving final outputs...")
    # Final asserts: no NaNs
    assert not X_final.isna().any().any(), "Unexpected NaNs remained after engineering."

    # Persist final feature matrix (numeric only)
    print("Writing feature matrix to parquet...")
    X_final.to_parquet(outdir / "X_features.parquet", index=False)
    main_progress.update(1)
    main_progress.close()

    # Final console summary
    print("\n=== DONE ===")
    print(f"Engineered features saved: {outdir/'X_features.parquet'}  (cols={X_final.shape[1]})")
    print(f"Label vector saved:        {outdir/'y_label.csv'}")
    print(f"Pre report:                {outdir/'feature_engineering_report_pre.csv'}")
    print(f"Post report:               {outdir/'feature_engineering_report_post.csv'}")
    print(f"Leakage check:             {outdir/'leakage_check.csv'}")
    print(f"Final feature list:        {outdir/'feature_list_final.csv'}")
    print(f"Transforms config:         {outdir/'transforms_config.json'}")
    print(f"Label stats:               {outdir/'label_stats.json'}")


if __name__ == "__main__":
    main()
