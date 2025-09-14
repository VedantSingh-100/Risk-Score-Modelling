#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Deterministic build of:
1) label sources (from sweep artifacts + best_config)
2) target label (union)
3) feature list (filtered by Fill_Rate and guard)
4) leakage checks and guard report

Inputs expected in CWD:
- sweep/best_config.json
- smart_label_candidates.csv
- negative_pattern_variables.csv
- Internal_Algo360VariableDictionary_WithExplanation.xlsx
- 50k_users_merged_data_userfile_updated_shopping.csv
- (optional) do_not_use_features.txt or "do not use features.txt"

Outputs written to deterministic_build/
"""

import json, re, sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tqdm import tqdm


# ----------------------------
# Paths & constants
# ----------------------------
DATA_CSV = "50k_users_merged_data_userfile_updated_shopping.csv"
DICT_XLSX = "Internal_Algo360VariableDictionary_WithExplanation.xlsx"
CANDIDATES_CSV = "smart_label_candidates.csv"
NEGATIVE_CSV = "negative_pattern_variables.csv"
BEST_CONFIG_JSON = "sweep/best_config.json"
OPTIONAL_GUARDS = ["do_not_use_features.txt", "do not use features.txt"]

OUT_DIR = Path("deterministic_build")
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Fill-rate threshold (0.85 means 85%). Will auto-handle 0–1 or 0–100 scales.
FILL_RATE_THRESHOLD = 0.85

# Leakage-guard outcome words (same spirit as framework)
OUTCOME_GUARD_TERMS = (
    r"default", r"dpd", r"overdue", r"arrear", r"write[\s-]?off", r"charge[\s-]?off",
    r"npa", r"settle", r"miss", r"min[_\s-]?due", r"over[-\s]?limit", r"declin", r"reject",
    r"bounced", r"nsf", r"negative"
)

# Near-duplicate threshold to drop label sources (best_config)
# and for leakage check vs target
DEFAULT_DEDUP_JACCARD = 0.85
LEAKAGE_JACCARD_VS_TARGET = 0.98

# If candidates lack max prevalence in best_config, use a conservative cap.
DEFAULT_MAX_LABEL_PREVALENCE = 0.30


# ----------------------------
# Utils
# ----------------------------
def read_best_config(path: str) -> Dict:
    cfg = json.loads(Path(path).read_text())
    # set sensible fallbacks if missing
    cfg.setdefault("max_label_prevalence", DEFAULT_MAX_LABEL_PREVALENCE)
    return cfg

def find_dict_sheet_and_columns(xlsx_path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load the 'Internal_Algo...' dictionary and return a frame with normalized columns:
       variable, description, fill_rate (float in 0–1), plus pass-through columns."""
    xl = pd.ExcelFile(xlsx_path)
    # choose the biggest sheet by rows
    sizes = {s: pd.read_excel(xlsx_path, sheet_name=s).shape[0] for s in xl.sheet_names}
    best = max(sizes, key=sizes.get)
    df = pd.read_excel(xlsx_path, sheet_name=best)

    # Flexible column finding
    cols_lower = {c.lower().strip(): c for c in df.columns}
    # common names in your message: Variables, New_Description, Explaination, Category, Metric Type, Data Type, Importance_Criteria, Fill_Rate, Confidence_Rate
    name_col = None
    for k, orig in cols_lower.items():
        if k in ("variables", "variable", "field", "column", "name"):
            name_col = orig; break
    if name_col is None:
        raise ValueError("Could not find 'Variables' column in dictionary.")

    desc_col = None
    for key in ("new_description", "explaination", "description", "explanation", "meaning", "details"):
        if key in cols_lower:
            desc_col = cols_lower[key]; break

    fill_col = None
    for key in ("fill_rate", "fill rate", "fill_rate%", "fill rate %", "fill%"):
        if key in cols_lower:
            fill_col = cols_lower[key]; break
    if fill_col is None:
        raise ValueError("Could not find a Fill Rate column in dictionary.")

    # Normalize to a simple schema
    out = pd.DataFrame()
    out["variable"] = df[name_col].astype(str)
    if desc_col:
        out["description"] = df[desc_col].astype(str)
    else:
        out["description"] = ""

    # Convert Fill Rate to 0–1
    fr = pd.to_numeric(df[fill_col], errors="coerce")
    # If values look like percentages (e.g., > 1), convert
    if fr.dropna().gt(1).mean() > 0.5:
        fr = fr / 100.0
    out["fill_rate"] = fr.clip(lower=0, upper=1)

    # Keep the rest for reference
    for c in df.columns:
        if c not in (name_col, desc_col, fill_col):
            out[c] = df[c]
    return out, {"name_col": name_col, "desc_col": desc_col, "fill_col": fill_col, "sheet": best}

def load_candidates(candidates_csv: str) -> pd.DataFrame:
    cand = pd.read_csv(candidates_csv)
    # expected columns from framework: variable, description, outcome_score,
    # quality_score, positive_rate, eligible_for_label (maybe)
    # Normalize typical variations
    rename_map = {}
    for c in cand.columns:
        cl = c.lower().strip()
        if cl == "variable": rename_map[c] = "variable"
        if cl == "description": rename_map[c] = "description"
        if cl == "outcome_score": rename_map[c] = "outcome_score"
        if cl == "quality_score": rename_map[c] = "quality_score"
        if cl == "positive_rate": rename_map[c] = "positive_rate"
        if cl == "eligible_for_label": rename_map[c] = "eligible_for_label"
    cand = cand.rename(columns=rename_map)
    # ensure basics exist
    for must in ["variable", "description", "outcome_score", "quality_score", "positive_rate"]:
        if must not in cand.columns:
            raise ValueError(f"Missing '{must}' in smart_label_candidates.csv")
    return cand

def load_negative(neg_csv: str) -> pd.DataFrame:
    neg = pd.read_csv(neg_csv)
    # Normalize
    rename_map = {}
    for c in neg.columns:
        cl = c.lower().strip()
        if cl == "variable": rename_map[c] = "variable"
        if cl == "description": rename_map[c] = "description"
        if cl == "recommended_for_label": rename_map[c] = "recommended_for_label"
        if cl == "positive_rate": rename_map[c] = "positive_rate"
        if cl == "priority_score": rename_map[c] = "priority_score"
    neg = neg.rename(columns=rename_map)
    # best effort
    for must in ["variable", "description"]:
        if must not in neg.columns:
            raise ValueError(f"Missing '{must}' in negative_pattern_variables.csv")
    if "recommended_for_label" not in neg.columns:
        neg["recommended_for_label"] = False
    return neg

def choose_label_sources(cand: pd.DataFrame,
                         neg: pd.DataFrame,
                         best_cfg: Dict) -> pd.DataFrame:
    """Deterministically select label sources using best_config thresholds
       and prioritizing negative_pattern recommendations."""
    out_imp = best_cfg.get("outcome_importance_threshold", 0.6)
    q_thresh = best_cfg.get("quality_threshold_for_label_eligibility", 0.6)
    min_prev = best_cfg.get("min_label_prevalence", 0.01)
    max_prev = best_cfg.get("max_label_prevalence", DEFAULT_MAX_LABEL_PREVALENCE)

    # Eligible by thresholds
    cand["positive_rate"] = pd.to_numeric(cand["positive_rate"], errors="coerce")
    eligible = cand[
        (cand["outcome_score"] >= out_imp) &
        (cand["quality_score"] >= q_thresh) &
        (cand["positive_rate"].between(min_prev, max_prev, inclusive="both"))
    ].copy()

    # Merge in negative pattern vars first (recommended=True)
    neg_rec = neg[neg["recommended_for_label"]].copy()
    neg_rec["source"] = "negative_pattern"
    eligible["source"] = "general"

    # Avoid doubles
    combined = pd.concat([neg_rec[eligible.columns.intersection(["variable","description","positive_rate","source"])],
                          eligible], ignore_index=True)

    # Drop duplicates keeping first occurrence (negative_pattern prioritized by concat order)
    combined = combined.drop_duplicates(subset=["variable"], keep="first")
    return combined

def read_columns(df_path: str, cols: List[str]) -> pd.DataFrame:
    # Read only needed columns; keep order if possible
    usecols = [c for c in cols if c]  # guard
    return pd.read_csv(df_path, usecols=lambda c: c in usecols, low_memory=False)

def to_binary(series: pd.Series) -> pd.Series:
    s = series
    if not pd.api.types.is_numeric_dtype(s):
        s = pd.to_numeric(s, errors="coerce")
    return (s.fillna(0) > 0).astype(np.uint8)

def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    A = a.astype(bool); B = b.astype(bool)
    inter = (A & B).sum()
    union = (A | B).sum()
    return float(inter/union) if union > 0 else 0.0

def dedup_by_jaccard(bin_df: pd.DataFrame, threshold: float) -> List[str]:
    """Keep columns greedily, dropping near-duplicates by Jaccard≥threshold.
       Order of preference is current column order."""
    kept = []
    print("Checking for near-duplicate label sources...")
    for c in tqdm(bin_df.columns, desc="Deduplicating"):
        drop = False
        for k in kept:
            if jaccard(bin_df[c].values, bin_df[k].values) >= threshold:
                drop = True; break
        if not drop:
            kept.append(c)
    return kept

def dominance_prune(bin_df: pd.DataFrame, union_vec: pd.Series, cutoff: float) -> List[str]:
    """Iteratively remove any column whose share among positives ≥ cutoff."""
    removed = []
    pos_mask = union_vec.values == 1
    work = bin_df.copy()
    while True:
        if work.shape[1] == 0:
            break
        shares = [(c, float(work[c].values[pos_mask].mean()) if pos_mask.any() else 0.0) for c in work.columns]
        dom = [c for c, s in shares if s >= cutoff]
        if not dom: break
        removed.extend(dom)
        work = work.drop(columns=dom)
        union_vec = (work.sum(axis=1) > 0).astype(np.uint8)
        pos_mask = union_vec.values == 1
    return removed

def build_union_label(data_csv: str,
                      label_sources: List[str],
                      include_lifetime: bool,
                      dedup_thr: float,
                      dominance_cutoff: float) -> Tuple[pd.Series, List[str], Dict[str, float], List[str], List[str]]:
    """Return (label_union, kept_sources, shares, dedup_dropped, dominance_dropped)."""
    if len(label_sources) == 0:
        raise ValueError("No candidate label sources provided.")

    # Read only label source columns that exist
    # We must check existence first
    header = pd.read_csv(data_csv, nrows=0).columns.tolist()
    present = [c for c in label_sources if c in header]

    if not include_lifetime:
        present = [c for c in present if "lifetime" not in c.lower()]
    if len(present) == 0:
        raise ValueError("No label sources exist in data after lifetime filter.")

    df = read_columns(data_csv, present)
    # Convert to binary
    B = pd.DataFrame({c: to_binary(df[c]) for c in df.columns})

    # Drop zero-signal
    nonzero = [c for c in B.columns if B[c].sum() > 0]
    B = B[nonzero]

    if B.shape[1] == 0:
        raise ValueError("All label sources have zero signal.")

    # Deduplicate near-duplicates
    kept = dedup_by_jaccard(B, dedup_thr)
    dedup_dropped = sorted(set(B.columns) - set(kept))
    B = B[kept]

    # Initial union (all kept sources)
    union = (B.sum(axis=1) > 0).astype(np.uint8)

    # Dominance prune
    dominance_dropped = dominance_prune(B, union, dominance_cutoff)
    if dominance_dropped:
        B = B.drop(columns=dominance_dropped)
        if B.shape[1] == 0:
            raise RuntimeError("Dominance prune removed all sources. Revisit thresholds.")
        union = (B.sum(axis=1) > 0).astype(np.uint8)

    # Contribution shares among positives
    pos = union.values == 1
    shares = {c: float(B[c].values[pos].mean()) if pos.any() else 0.0 for c in B.columns}

    return union, list(B.columns), shares, dedup_dropped, dominance_dropped

def build_guard(header: List[str],
                desc_map: Dict[str, str],
                label_sources: List[str]) -> Set[str]:
    pat = re.compile("|".join(OUTCOME_GUARD_TERMS), re.IGNORECASE)
    guard = set(label_sources)
    for c in header:
        text = (c + " " + desc_map.get(c, "")).lower()
        if pat.search(text):
            guard.add(c)
    # Merge optional, user-maintained guard files
    for fn in OPTIONAL_GUARDS:
        p = Path(fn)
        if p.exists():
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln:
                    guard.add(ln)
    return guard

def leakage_check_vs_target(data_csv: str,
                            features: List[str],
                            target: pd.Series,
                            jaccard_thr: float) -> pd.DataFrame:
    """Compute Jaccard(feature>0, target) for booleanizable features; flag ≥ threshold."""
    if len(features) == 0:
        return pd.DataFrame(columns=["feature", "jaccard_with_target", "flagged"])

    # Read features that exist
    header = pd.read_csv(data_csv, nrows=0).columns.tolist()
    present = [c for c in features if c in header]
    if not present:
        return pd.DataFrame(columns=["feature", "jaccard_with_target", "flagged"])

    print("Loading features for leakage check...")
    X = read_columns(data_csv, present)
    # Align index to target length
    if len(X) != len(target):
        # assume row order matches original CSV in both
        # if it does not, the project should add a stable key join
        pass

    t = target.values.astype(bool)
    rows = []
    print("Computing Jaccard similarity for leakage detection...")
    for c in tqdm(X.columns, desc="Leakage check"):
        b = to_binary(X[c]).values.astype(bool)
        j = jaccard(b, t)
        rows.append((c, j, j >= jaccard_thr))
    rep = pd.DataFrame(rows, columns=["feature", "jaccard_with_target", "flagged"])
    return rep.sort_values("jaccard_with_target", ascending=False)

def safe_write_df(df: pd.DataFrame, path: Path):
    df.to_csv(path, index=False)
    print(f"  -> {path} ({len(df):,} rows)")

def main():
    print("=== Deterministic label & feature build ===")
    
    # Setup progress tracking for main steps
    main_steps = [
        "Loading configuration and dictionary",
        "Selecting label sources",
        "Building union label",
        "Creating guard set",
        "Filtering features",
        "Running leakage checks",
        "Saving final outputs"
    ]
    
    main_progress = tqdm(main_steps, desc="Overall Progress")
    
    # Step 1: Load configuration
    best_cfg = read_best_config(BEST_CONFIG_JSON)
    print("Using best_config:\n", json.dumps(best_cfg, indent=2))

    include_lifetime = bool(best_cfg.get("include_lifetime_in_label", True))
    dedup_thr = float(best_cfg.get("dedup_jaccard_threshold", DEFAULT_DEDUP_JACCARD))
    dominance_cutoff = float(best_cfg.get("dominance_cutoff", 0.6))

    # Dictionary & Fill Rate filter
    print("Loading variable dictionary...")
    dict_df, meta = find_dict_sheet_and_columns(DICT_XLSX)
    print(f"Loaded dictionary sheet='{meta['sheet']}', name_col='{meta['name_col']}', fill_col='{meta['fill_col']}'")
    dict_df["fill_rate"] = pd.to_numeric(dict_df["fill_rate"], errors="coerce").fillna(0).clip(0,1)

    fill_filtered = dict_df[dict_df["fill_rate"] >= FILL_RATE_THRESHOLD].copy()
    print(f"Fill-rate filter @ {FILL_RATE_THRESHOLD:.2%}: kept {len(fill_filtered)}/{len(dict_df)} variables")

    # Build a {variable: description} map for guard checks
    desc_map = dict(zip(dict_df["variable"].astype(str), dict_df["description"].astype(str)))
    main_progress.update(1)

    # Step 2: Select label sources from sweep artifacts
    print("Loading label candidates...")
    cand = load_candidates(CANDIDATES_CSV)
    neg = load_negative(NEGATIVE_CSV)
    label_pool = choose_label_sources(cand, neg, best_cfg)
    print(f"Initial label pool size: {len(label_pool)} (negatives prioritized where recommended=True)")

    # Use only variables that exist in the dataset header
    print("Checking variable presence in dataset...")
    header = pd.read_csv(DATA_CSV, nrows=0).columns.tolist()
    pool_present = label_pool[label_pool["variable"].isin(header)].copy()
    if len(pool_present) == 0:
        raise RuntimeError("No label source variables from pool are present in the dataset.")
    pool_present = pool_present.drop_duplicates(subset=["variable"])
    main_progress.update(1)

    # Step 3: Build the union label deterministically
    print("Building union label...")
    union, kept_sources, shares, dedup_dropped, dom_dropped = build_union_label(
        DATA_CSV,
        label_sources=pool_present["variable"].tolist(),
        include_lifetime=include_lifetime,
        dedup_thr=dedup_thr,
        dominance_cutoff=dominance_cutoff
    )

    # Save label sources detail
    label_sources_df = pool_present.set_index("variable").loc[kept_sources].reset_index()
    label_sources_df["share_among_positives"] = label_sources_df["variable"].map(shares)
    label_sources_df["dropped_by_dedup"] = label_sources_df["variable"].isin(dedup_dropped)
    label_sources_df["dropped_by_dominance"] = label_sources_df["variable"].isin(dom_dropped)
    label_sources_df = label_sources_df.sort_values("share_among_positives", ascending=False).reset_index(drop=True)
    safe_write_df(label_sources_df, OUT_DIR / "selected_label_sources.csv")

    # Persist the target
    target_path = OUT_DIR / "label_union.csv"
    pd.DataFrame({"label_union": union}).to_csv(target_path, index=False)
    print(f"  -> {target_path} (positives={int(union.sum()):,}, prevalence={union.mean():.4f})")
    main_progress.update(1)

    # Step 4: Guard build
    print("Building guard set...")
    guard_set = build_guard(header, desc_map, kept_sources)
    print(f"Guard size (including label sources & outcome words & external files): {len(guard_set)}")
    Path(OUT_DIR / "guard_set.txt").write_text("\n".join(sorted(guard_set)))
    main_progress.update(1)

    # Step 5: Feature list: fill-rate filtered ∩ present in data – guard – labels
    print("Filtering features...")
    present_fill = fill_filtered[fill_filtered["variable"].isin(header)].copy()
    present_fill = present_fill[~present_fill["variable"].isin(guard_set)].copy()
    present_fill = present_fill.drop_duplicates(subset=["variable"])
    safe_write_df(present_fill[["variable", "description", "fill_rate"]], OUT_DIR / "selected_features_initial.csv")
    main_progress.update(1)

    # Step 6: Leakage check: Jaccard(feature>0, target) ≥ 0.98
    leak_report = leakage_check_vs_target(DATA_CSV,
                                          features=present_fill["variable"].tolist(),
                                          target=union,
                                          jaccard_thr=LEAKAGE_JACCARD_VS_TARGET)
    safe_write_df(leak_report, OUT_DIR / "guard_leakage_report.csv")

    # Remove flagged features
    flagged = set(leak_report.loc[leak_report["flagged"], "feature"])
    if flagged:
        print(f"Leakage check flagged {len(flagged)} features near-identical to the target; removing.")
        present_fill = present_fill[~present_fill["variable"].isin(flagged)].copy()
    main_progress.update(1)

    # Step 7: Final outputs
    print("Saving final outputs...")
    safe_write_df(present_fill[["variable", "description", "fill_rate"]],
                  OUT_DIR / "selected_features_final.csv")

    # Emit a compact summary JSON for reproducibility
    summary = {
        "best_config_used": best_cfg,
        "include_lifetime_in_label": include_lifetime,
        "dedup_jaccard_threshold": dedup_thr,
        "dominance_cutoff": dominance_cutoff,
        "fill_rate_threshold": FILL_RATE_THRESHOLD,
        "num_label_sources_final": len(kept_sources),
        "num_features_initial": int((OUT_DIR / "selected_features_initial.csv").exists() and
                                    len(pd.read_csv(OUT_DIR / "selected_features_initial.csv")) or 0),
        "num_features_final": len(present_fill),
        "target_prevalence": float(union.mean()),
        "target_positives": int(union.sum()),
        "dedup_dropped": dedup_dropped,
        "dominance_dropped": dom_dropped,
    }
    (OUT_DIR / "build_summary.json").write_text(json.dumps(summary, indent=2))
    main_progress.update(1)
    main_progress.close()
    print("=== Done. See deterministic_build/ for outputs. ===")


if __name__ == "__main__":
    pd.options.mode.copy_on_write = True
    main()
