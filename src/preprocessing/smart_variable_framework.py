"""
Smart Variable Framework for Risk Modeling
==========================================

This framework automatically:
1. Identifies potential label variables from multiple data sources
2. Creates composite labels using weighted combinations
3. Selects the most predictive features
4. Handles missing data and quality issues
5. Provides interpretable results and recommendations

Inputs:
- Internal_Algo360VariableDictionary_WithExplanation.xlsx (variable descriptions)
- 50k_users_merged_data_userfile_updated_shopping.csv (user data)
- variable_catalog.csv (optional: pre-existing variable analysis)

Outputs:
- smart_label_candidates.csv (ranked potential label variables)
- composite_labels.csv (different label combination strategies)
- feature_importance_matrix.csv (predictive power of all features)
- variable_quality_report.csv (data quality assessment)
- recommended_pipeline.json (best configuration found)
- smart_framework_report.md (comprehensive analysis)
"""

import os, time, argparse, hashlib, random, json, re, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")

# =============================================
# CONFIGURATION
# =============================================

# ==== [NEW HELPERS FOR WINDOWS, GUARDS, PRUNING] ==========================
import re
from datetime import datetime
from sklearn.metrics import roc_auc_score

WINDOW_PATTERNS = {
    "1m": re.compile(r"\b(last\s*1\s*month|1m|30d)\b", re.I),
    "3m": re.compile(r"\b(last\s*3\s*month|3m|90d)\b", re.I),
    "6m": re.compile(r"\b(last\s*6\s*month|6m|180d)\b", re.I),
    "12m": re.compile(r"\b(last\s*12\s*month|12m|360d|1\s*year)\b", re.I),
    "lt": re.compile(r"\b(life\s*time|lifetime|ever|since\s*inception)\b", re.I),
}

OUTCOME_PAT = re.compile(
    r"(write[\s-]?off|charge[\s-]?off|npa|default|dpd|over[-\s]?limit|"
    r"over\s?due|overdue|past.?due|arrear|min[_\s-]?due|mindue|miss(?:ed|ing)?|"
    r"declin\w*|reject\w*|insufficient|penalt\w*|bounc\w*|ecs|nach|negative\w*)",
    re.I
)

ID_PAT = re.compile(r"(?:^|_)(id|uuid|pan|aadhaar|account|application|lead|mobile|email)(?:_|$)", re.I)

def infer_window(name: str, desc: str) -> str:
    s = f"{name} {desc}".lower()
    for tag, pat in WINDOW_PATTERNS.items():
        if pat.search(s):
            return tag
    return "unknown"

def is_outcome_like(name: str, desc: str) -> bool:
    s = f"{name} {desc}".lower()
    return bool(OUTCOME_PAT.search(s))

def variable_family(varname: str) -> str:
    """Group variables by a coarse 'family' key to guard whole families."""
    v = str(varname).lower()
    m = re.match(r"var(\d{3})", v)  # var201xxx -> var201
    if m:
        return f"var{m.group(1)}"
    # fallback: prefix until first underscore
    return v.split("_")[0]

def id_like_columns(cols):
    return {c for c in cols if ID_PAT.search(str(c))}

def near_unique_columns(df: pd.DataFrame, thresh: float = 0.98):
    out = set()
    n = len(df)
    for c in df.columns:
        try:
            u = df[c].nunique(dropna=False)
            if u >= thresh * n:
                out.add(c)
        except Exception:
            pass
    return out

def too_predictive_guard(dfX: pd.DataFrame, y: pd.Series, max_auc: float = 0.88, sample: int = 12000, seed: int = 42):
    """Smell test: any single raw feature with >max_auc AUC is suspicious."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    flags = set()
    yv = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    for c in dfX.columns:
        try:
            xv = pd.to_numeric(dfX[c], errors="coerce").fillna(0).values
            auc = roc_auc_score(yv, xv[idx]) if hasattr(xv, "__getitem__") else roc_auc_score(yv, xv)
            if auc > max_auc or auc < (1 - max_auc):
                flags.add(c)
        except Exception:
            continue
    return flags

def prune_correlated(X: pd.DataFrame, corr_thr: float = 0.95, keep_priority: pd.Series | None = None):
    """
    Drop one feature from each highly correlated pair (> corr_thr).
    If keep_priority is provided (higher is more important), we keep the higher-priority feature.
    """
    if X.shape[1] <= 1:
        return X.copy(), []

    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = set()
    for col in upper.columns:
        high = upper.index[upper[col] > corr_thr].tolist()
        for h in high:
            if h in to_drop or col in to_drop:
                continue
            if keep_priority is not None:
                ka = float(keep_priority.get(col, 0.0))
                kb = float(keep_priority.get(h, 0.0))
                drop = col if ka < kb else h
            else:
                # default: drop the later column alphabetically to be deterministic
                drop = max(col, h)
            to_drop.add(drop)

    kept = [c for c in X.columns if c not in to_drop]
    return X[kept].copy(), sorted(list(to_drop))

def time_safe_feature_mask(feature_names: list[str], descriptions: dict[str, str], forbidden_windows: set[str]):
    """Mask out features whose window collides with forbidden windows (e.g., {'lt'} to remove lifetime features)."""
    mask_keep = []
    for f in feature_names:
        w = infer_window(f, descriptions.get(f, ""))
        mask_keep.append(w not in forbidden_windows)
    return pd.Series(mask_keep, index=feature_names)
# ========================================================================

@dataclass
class FrameworkConfig:
    # File paths
    raw_data_path: str = "/home/vhsingh/Parshvi_project/data/raw/50k_users_merged_data_userfile_updated_shopping.csv"
    dictionary_path: str = "/home/vhsingh/Parshvi_project/data/raw/Internal_Algo360VariableDictionary_WithExplanation.xlsx"
    variable_catalog_path: str = "variable_catalog.csv"

    # Sampling
    max_rows_analysis: int = 50000
    max_features_analysis: int = 2000

    # Label identification
    outcome_importance_threshold: float = 0.7
    min_label_prevalence: float = 0.005
    max_label_prevalence: float = 0.30

    # Label composition & guarding
    include_lifetime_in_label: bool = False             # exclude lifetime flags from label by default
    target_weighted_label_prevalence: Optional[float] = None  # if None, match union prevalence
    outcome_guard_terms: Tuple[str, ...] = (
        r"default", r"dpd", r"overdue", r"arrear", r"write[\s-]?off", r"charge[\s-]?off",
        r"npa", r"settle", r"miss", r"min[_\s-]?due", r"over[-\s]?limit", r"declin", r"reject",
        r"bounced", r"nsf", r"negative"
    )
    dominance_cutoff: float = 0.60           # drop any label source dominating union positives
    dedup_jaccard_threshold: float = 0.90    # drop near-duplicates before union/weights

    # Rescue policy to avoid "no labels"
    rescue_min_prevalence: float = 0.002       # 0.2%
    rescue_max_prevalence: float = 0.40        # 40% upper cap
    rescue_top_k: int = 12                     # max sources to seed a composite label

    # Feature selection
    max_correlation_threshold: float = 0.95
    min_feature_importance: float = 0.001
    stability_threshold: float = 0.7
    quality_threshold_for_label_eligibility: float = 0.50
    severe_weight_threshold_for_union: float = 0.80  # severity cutoff for which sources go into union

    # Model params
    n_cv_folds: int = 5
    n_stability_runs: int = 10
    random_seed: int = 42

# =============================================
# OUTCOME/LABEL IDENTIFICATION
# =============================================

class LabelIdentifier:
    """Identifies potential outcome variables for labeling"""

    def __init__(self, config: FrameworkConfig):
        self.config = config

        self.keyword_weights = {
            r"(writeoff|charge.?off|npa|bankruptcy|foreclosure|repossession)": 1.00,
            r"\bdefault(s|ed|ing)?\b": 0.95,
            r"\b90.?dpd\b": 0.90, r"\b60.?dpd\b": 0.75, r"\b30.?dpd\b": 0.60,
            r"(over\s?due|overdue|past.?due|arrear|miss(?:ed|ing)?|min[_\s-]?due|mindue)": 0.75,
            r"over[-\s]?limit": 0.60,
            r"(negative\w*|noofnegativeevents)": 0.60,
            r"(declin\w*|reject\w*|insufficient|penalt\w*|bounced|nsf|chargeback)": 0.50,
            # fraud/compliance are *different* outcomes; keep low or exclude from default label discovery
            r"(fraud|suspicious|aml|kyc.?fail|block|freeze)": 0.10
        }

        # Exclusion patterns (behavioral, not outcomes)
        self.exclusion_patterns = [
            'email', 'phone', 'address', 'demographic', 'age', 'income',
            'campaign', 'channel', 'device', 'browser', 'ip', 'session',
            'click', 'view', 'visit', 'engagement', 'preference'
        ]

    def score_variable_as_outcome(self, name: str, description: str) -> Dict[str, Any]:
        text = f"{name} {description}".lower()
        if any(re.search(p, text, re.I) for p in self.exclusion_patterns):
            return {'outcome_score': 0.0, 'outcome_type': 'excluded',
                    'matching_keywords': [], 'reason': 'Behavioral/demographic excluded'}

        max_score, best_pat, matches = 0.0, None, []
        for pat, wt in self.keyword_weights.items():
            if re.search(pat, text, re.I):
                matches.append(pat)
                if wt > max_score:
                    max_score, best_pat = wt, pat

        return {
            'outcome_score': max_score,
            'outcome_type': 'keyword_weighted' if best_pat else 'none',
            'matching_keywords': matches,
            'reason': f"Matched {best_pat}" if best_pat else "No outcome pattern found",
        }

    def find_negative_pattern_variables(self, df: pd.DataFrame, descriptions: Dict[str, str]) -> pd.DataFrame:
        """Specifically find variables matching the discovered negative patterns"""
        print("🔍 Scanning for variables with discovered negative patterns...")

        negative_patterns = [
            r'over[-\s]?limit',
            r'default|defaults',
            r'declin\w*',
            r'reject\w*',
            r'insufficient',
            r'penalt\w*',
            r'miss(?:ed|ing)?',
            r'over\s?due|overdue',
            r'min[_\s-]?due|mindue',
            r'\bdue\b|\bdues\b',
            r'negative\w*|noofnegativeevents'
        ]
        combined_pattern = '|'.join(f'({pattern})' for pattern in negative_patterns)
        compiled_regex = re.compile(combined_pattern, re.IGNORECASE)

        negative_matches = []
        for col in tqdm(df.columns, desc="Scanning for negative patterns"):
            desc = descriptions.get(col, "")
            text = f"{col} {desc}".lower()

            match = compiled_regex.search(text)
            if match:
                matched_patterns = []
                for pattern in negative_patterns:
                    if re.search(pattern, text, re.IGNORECASE):
                        matched_patterns.append(pattern)

                series = df[col]
                missing_pct = series.isna().mean()
                unique_count = series.nunique(dropna=True)

                if series.dtype in ['object', 'string']:
                    numeric_series = pd.to_numeric(series.astype(str).str.lower().map({
                        'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0
                    }), errors='coerce')
                    if numeric_series.notna().mean() > 0.8:
                        series = numeric_series

                if not pd.api.types.is_numeric_dtype(series):
                    series = pd.to_numeric(series, errors='coerce')

                unique_vals = set(series.dropna().unique())
                is_binary = len(unique_vals) <= 2 and unique_vals.issubset({0, 1})

                if is_binary and len(series) > 0:
                    pos_rate = float(series.mean())
                    pos_count = int(series.sum())
                else:
                    pos_rate = np.nan
                    pos_count = 0

                priority_score = 0.0
                if missing_pct < 0.1:
                    priority_score += 0.4
                elif missing_pct < 0.3:
                    priority_score += 0.2

                if is_binary:
                    priority_score += 0.3
                    if not pd.isna(pos_rate) and 0.005 <= pos_rate <= 0.3:
                        priority_score += 0.3

                negative_matches.append({
                    'variable': col,
                    'description': desc[:200],
                    'matched_patterns': '; '.join(matched_patterns),
                    'pattern_count': len(matched_patterns),
                    'is_binary': is_binary,
                    'positive_rate': pos_rate,
                    'positive_count': pos_count,
                    'missing_pct': missing_pct,
                    'unique_count': unique_count,
                    'priority_score': priority_score,
                    'recommended_for_label': (
                        is_binary and
                        not pd.isna(pos_rate) and
                        0.005 <= pos_rate <= 0.3 and
                        missing_pct < 0.5 and
                        priority_score >= 0.6
                    )
                })

        if negative_matches:
            result_df = pd.DataFrame(negative_matches).sort_values(
                ['recommended_for_label', 'priority_score', 'pattern_count'],
                ascending=[False, False, False]
            )
        else:
            # Create empty DataFrame with expected columns
            result_df = pd.DataFrame(columns=[
                'variable', 'description', 'matched_patterns', 'pattern_count',
                'is_binary', 'positive_rate', 'positive_count', 'missing_pct',
                'unique_count', 'priority_score', 'recommended_for_label'
            ])

        print(f"   Found {len(result_df)} variables matching negative patterns")
        if len(result_df) > 0:
            recommended = result_df[result_df['recommended_for_label']]
            print(f"   {len(recommended)} variables recommended for label consideration")

        return result_df

    def analyze_candidates(self, df: pd.DataFrame, descriptions: Dict[str, str]) -> pd.DataFrame:
        """Analyze all variables for outcome potential"""
        print("🎯 Analyzing variables for label potential...")

        candidates = []
        for col in tqdm(df.columns, desc="Scoring variables"):
            desc = descriptions.get(col, "")

            series = df[col]
            missing_pct = series.isna().mean()
            unique_count = series.nunique(dropna=True)

            if series.dtype in ['object', 'string']:
                numeric_series = pd.to_numeric(series.astype(str).str.lower().map({
                    'true': 1, 'false': 0, 'yes': 1, 'no': 0, 'y': 1, 'n': 0
                }), errors='coerce')
                if numeric_series.notna().mean() > 0.8:
                    series = numeric_series

            if not pd.api.types.is_numeric_dtype(series):
                series = pd.to_numeric(series, errors='coerce')

            unique_vals = set(series.dropna().unique())
            is_binary = len(unique_vals) <= 2 and unique_vals.issubset({0, 1})

            pos_rate = float(series.mean()) if is_binary and len(series) > 0 else np.nan

            outcome_info = self.score_variable_as_outcome(col, desc)
            window_tag = infer_window(col, desc)
            lifetime_flag = (window_tag == "lt")

            quality_score = 1.0
            if missing_pct > 0.8:
                quality_score *= 0.3
            elif missing_pct > 0.5:
                quality_score *= 0.7
            if unique_count <= 1:
                quality_score = 0.0

            candidates.append({
                'variable': col,
                'description': desc[:200],
                'window': window_tag,
                'is_lifetime_window': lifetime_flag,
                'outcome_score': outcome_info['outcome_score'],
                'outcome_type': outcome_info['outcome_type'],
                'matching_keywords': '; '.join(outcome_info['matching_keywords']),
                'is_binary': is_binary,
                'positive_rate': pos_rate,
                'missing_pct': missing_pct,
                'unique_count': unique_count,
                'quality_score': quality_score,
                'combined_score': outcome_info['outcome_score'] * quality_score,
                'eligible_for_label': (
                    is_binary and
                    outcome_info['outcome_score'] >= self.config.outcome_importance_threshold and
                    quality_score > self.config.quality_threshold_for_label_eligibility and
                    (not pd.isna(pos_rate)) and
                    (self.config.min_label_prevalence <= pos_rate <= self.config.max_label_prevalence)
                )
            })

        result_df = pd.DataFrame(candidates).sort_values(
            ['eligible_for_label', 'combined_score', 'outcome_score'],
            ascending=[False, False, False]
        )
        return result_df


# =============================================
# COMPOSITE LABEL BUILDER
# =============================================

class CompositeLabelBuilder:
    """Creates composite labels from multiple outcome variables"""

    def __init__(self, config: FrameworkConfig):
        self.config = config

    def create_label_variants(self, df: pd.DataFrame, candidates: pd.DataFrame,
                              negative_pattern_vars: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict]:
        """Create different composite label strategies"""
        print("🏗️ Building composite label variants...")

        # Prioritize negative pattern variables if available
        if negative_pattern_vars is not None and len(negative_pattern_vars) > 0:
            priority_vars = negative_pattern_vars[negative_pattern_vars['recommended_for_label']].copy()
            if len(priority_vars) > 0:
                print(f"   ⭐ Prioritizing {len(priority_vars)} negative pattern variables")
                priority_vars['source'] = 'negative_pattern'
                general_eligible = candidates[candidates['eligible_for_label']].copy()
                general_eligible = general_eligible[~general_eligible['variable'].isin(priority_vars['variable'])]
                general_eligible['source'] = 'general'
                eligible = pd.concat([priority_vars, general_eligible], ignore_index=True)
            else:
                eligible = candidates[candidates['eligible_for_label']].copy()
                eligible['source'] = 'general'
        else:
            eligible = candidates[candidates['eligible_for_label']].copy()
            eligible['source'] = 'general'

        if len(eligible) == 0:
            print("⚠️ No eligible label variables found!")
            return pd.DataFrame(), {}

        negative_count = len(eligible[eligible.get('source', '') == 'negative_pattern'])
        general_count = len(eligible[eligible.get('source', '') == 'general'])
        print(f"   Using {len(eligible)} eligible variables ({negative_count} from negative patterns, {general_count} general)")

        # Build binary matrix from eligible vars
        label_data = pd.DataFrame(index=df.index)
        meta = {}
        skipped_lifetime, skipped_zero, kept = [], [], []

        for _, row in eligible.iterrows():
            col, desc = row['variable'], row['description']
            if col not in df.columns:
                continue
            if (not self.config.include_lifetime_in_label) and (
                "lifetime" in str(col).lower() or "lifetime" in str(desc).lower()
            ):
                skipped_lifetime.append(col); continue

            s = df[col]
            if not pd.api.types.is_numeric_dtype(s):
                s = pd.to_numeric(s, errors='coerce')
            b = (s.fillna(0) > 0).astype(int)
            if b.sum() == 0:
                skipped_zero.append(col); continue

            label_data[col] = b
            meta[col] = {'weight': row['outcome_score'], 'type': row['outcome_type'], 'description': desc}
            kept.append(col)

        print(f"   Label sources after filters → kept={len(kept)} | lifetime_skipped={len(skipped_lifetime)} | zero_signal={len(skipped_zero)}")

        # RESCUE: if none survived, expand pool without exposures; prefer recent windows
        if label_data.shape[1] == 0:
            print("   ⚠️ No sources survived. RESCUE #1: expand to non-lifetime default/DPD/overdue/mindue/missed")
            pool1 = self._select_rescue_pool(candidates, allow_lifetime=False)
            for _, row in pool1.iterrows():
                col = row['variable']
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors='coerce')
                    b = (s.fillna(0) > 0).astype(int)
                    if b.sum() == 0: continue
                    label_data[col] = b
                    meta[col] = {'weight': row['outcome_score'], 'type': row['outcome_type'], 'description': row['description']}

        # If still empty, allow lifetime but only real outcomes (default/dpd/missed), NOT exposures
        if label_data.shape[1] == 0:
            print("   ⚠️ Still empty. RESCUE #2: allow lifetime for default/DPD/missed only (exposures still banned)")
            pool2 = self._select_rescue_pool(candidates, allow_lifetime=True)
            txt2 = (pool2['variable'].astype(str) + " " + pool2['description'].astype(str)).str.lower()
            must_pat = re.compile(r"(default|dpd|miss(?:ed|ing)?)", re.I)
            ban_pat  = re.compile(r"(over[-\s]?limit|negative\w*|declin\w*|reject\w*|insufficient|penalt\w*)", re.I)
            pool2 = pool2[txt2.apply(lambda s: bool(must_pat.search(s)) and not ban_pat.search(s))]
            for _, row in pool2.iterrows():
                col = row['variable']
                if col in df.columns:
                    s = pd.to_numeric(df[col], errors='coerce')
                    b = (s.fillna(0) > 0).astype(int)
                    if b.sum() == 0: continue
                    label_data[col] = b
                    meta[col] = {'weight': row['outcome_score'], 'type': row['outcome_type'], 'description': row['description']}

        if label_data.shape[1] == 0:
            print("   ❌ Rescue failed: no viable label sources. Tip: relax rescue_min_prevalence or review variable catalog.")
            return pd.DataFrame(), {}

        # --------------------------------------
        # De-duplicate near-identical sources by Jaccard BEFORE dominance prune
        if label_data.shape[1] >= 2:
            kept_cols = []
            for c in sorted(label_data.columns, key=lambda c: meta[c]['weight'], reverse=True):
                if not kept_cols:
                    kept_cols.append(c); continue
                is_dup = any(
                    (label_data[c].astype(bool) & label_data[k].astype(bool)).sum() /
                    max(1, (label_data[c].astype(bool) | label_data[k].astype(bool)).sum())
                    >= self.config.dedup_jaccard_threshold
                    for k in kept_cols
                )
                if not is_dup:
                    kept_cols.append(c)
            dropped = sorted(set(label_data.columns) - set(kept_cols))
            if dropped:
                print(f"   🔁 De-duplicated {len(dropped)} label sources (Jaccard≥{self.config.dedup_jaccard_threshold})")
            label_data = label_data[kept_cols]
            meta = {k: meta[k] for k in kept_cols}

        if label_data.shape[1] == 0:
            return pd.DataFrame(), {}

        # Cache backup in case dominance prune wipes all
        _label_data_backup = label_data.copy()
        _meta_backup = meta.copy()

        # Strategy 1: UNION on severe cols (configurable), else all
        sev_thr = getattr(self.config, 'severe_weight_threshold_for_union', 0.80)
        severe_cols = [c for c in label_data.columns if meta[c]['weight'] >= sev_thr]
        union_base = label_data[severe_cols] if len(severe_cols) > 0 else label_data
        label_union = (union_base.sum(axis=1) > 0).astype(int)

        def compute_union(ld): return (ld.sum(axis=1) > 0).astype(int)
        def contribution(ld, u):
            shares = []
            pos = (u == 1)
            for c in ld.columns:
                shares.append((c, float(ld[c][pos].mean())))
            return sorted(shares, key=lambda x: x[1], reverse=True)

        # Dominance prune (NON-FATAL) with safety
        def compute_union(ld): return (ld.sum(axis=1) > 0).astype(int)
        union_now = compute_union(label_data)

        def quick_prev(ld):  # simple proxy; we keep prevalence stable within a tiny tolerance
            u = compute_union(ld)
            return float(u.mean())

        best_prev = float(union_now.mean())
        best_ld = label_data.copy(); best_meta = meta.copy()
        pruned = []
        while True:
            contrib = contribution(label_data, union_now)
            dom = [c for c, s in contrib if s >= self.config.dominance_cutoff]
            if not dom:
                break
            trial = label_data.drop(columns=dom)
            if trial.shape[1] == 0:
                break
            new_prev = quick_prev(trial)
            # only accept prune if prevalence doesn't drop materially
            if new_prev >= best_prev - 0.002:
                for c in dom:
                    pruned.append(c)
                    meta.pop(c, None)
                label_data = trial
                union_now = compute_union(label_data)
                best_prev = new_prev
            else:
                # stop pruning if it starts harming union coverage
                break

        if pruned:
            print(f"   🪓 Dominance prune removed {len(pruned)} source(s) with share≥{self.config.dominance_cutoff}")


        # Recompute variants from the pruned (or restored) set
        severe_cols = [c for c in label_data.columns if meta[c]['weight'] >= sev_thr]
        union_base = label_data[severe_cols] if len(severe_cols) > 0 else label_data
        label_union = (union_base.sum(axis=1) > 0).astype(int)

        # Strategy 2: severity-weighted with TARGET prevalence
        weights = np.array([meta[c]['weight'] for c in label_data.columns])
        weighted_score = (label_data.values * weights).sum(axis=1)
        positives = weighted_score[weighted_score > 0]
        if len(positives) > 0:
            target_prev = (self.config.target_weighted_label_prevalence
                           if self.config.target_weighted_label_prevalence is not None
                           else float(label_union.mean()))
            target_prev = max(0.005, min(0.5, target_prev))
            thr = float(np.quantile(positives, 1 - target_prev))
            label_weighted = (weighted_score >= thr).astype(int)
        else:
            thr = 1.0
            label_weighted = (weighted_score >= thr).astype(int)

        # Strategy 3: Hierarchical (worst-first)
        severity_order = sorted(label_data.columns, key=lambda c: meta[c]['weight'], reverse=True)
        def get_hierarchical_label(row):
            for col in severity_order:
                if row[col] == 1:
                    return 1
            return 0
        label_hierarchical = label_data.apply(get_hierarchical_label, axis=1).astype(int)

        # Strategy 4: Clustered
        if len(label_data.columns) >= 3:
            try:
                corr_matrix = label_data.corr().abs()
                kmeans = KMeans(n_clusters=min(3, max(1, len(label_data.columns)//2)),
                                random_state=self.config.random_seed)
                clusters = kmeans.fit_predict(corr_matrix.values)
                reps = []
                for cl in range(kmeans.n_clusters):
                    vars_in = [label_data.columns[i] for i, c in enumerate(clusters) if c == cl]
                    best = max(vars_in, key=lambda v: meta[v]['weight'])
                    reps.append(best)
                label_clustered = (label_data[reps].sum(axis=1) > 0).astype(int)
            except Exception:
                label_clustered = label_union.copy()
        else:
            label_clustered = label_union.copy()

        labels_df = pd.DataFrame({
            'label_union': label_union,
            'label_weighted': label_weighted,
            'label_hierarchical': label_hierarchical,
            'label_clustered': label_clustered
        })

        # Individual very-severe indicators (optional)
        for col in label_data.columns:
            if meta[col]['weight'] >= 0.9:
                labels_df[f'label_{col}'] = label_data[col]

        # store threshold used by weighted strategy
        labels_df.attrs['weighted_threshold'] = thr
        return labels_df, meta

    def _select_rescue_pool(self, candidates: pd.DataFrame, allow_lifetime: bool) -> pd.DataFrame:
        """Broaden pool: prefer default/DPD/overdue/mindue/missed; exclude exposures; relax prevalence."""
        if candidates is None or len(candidates) == 0:
            return pd.DataFrame()

        txt = (candidates['variable'].astype(str) + " " + candidates['description'].astype(str)).str.lower()

        pos_pat = re.compile(r"(default|dpd|overdue|min[_\s-]?due|miss(?:ed|ing)?|arrear)", re.I)
        exp_pat = re.compile(r"(over[-\s]?limit|negative\w*|declin\w*|reject\w*|insufficient|penalt\w*)", re.I)

        pool = candidates.copy()
        pool = pool[txt.apply(lambda s: bool(pos_pat.search(s)))]
        pool = pool[~txt.apply(lambda s: bool(exp_pat.search(s)))]

        if not allow_lifetime:
            mask_lt = txt.str.contains("lifetime")
            pool = pool[~mask_lt]

        pr = pool['positive_rate'].astype(float)
        pool = pool[(pr >= self.config.rescue_min_prevalence) & (pr <= self.config.rescue_max_prevalence)]

        pool = pool.sort_values(
            ['outcome_score', 'quality_score', 'positive_rate'],
            ascending=[False, False, False]
        ).head(self.config.rescue_top_k)

        return pool

# =============================================
# SMART FEATURE SELECTOR
# =============================================

class SmartFeatureSelector:
    """Intelligent feature selection with multiple criteria"""

    def __init__(self, config: FrameworkConfig):
        self.config = config

    def analyze_feature_quality(self, df: pd.DataFrame, descriptions: Dict[str, str]) -> pd.DataFrame:
        """Comprehensive feature quality analysis"""
        print("🔍 Analyzing feature quality and characteristics...")

        features = []
        for col in tqdm(df.columns, desc="Feature analysis"):
            series = df[col]
            desc = descriptions.get(col, "")

            missing_pct = series.isna().mean()
            unique_count = series.nunique(dropna=True)
            total_count = len(series)

            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_categorical = series.dtype == 'object' or unique_count < 20
            is_binary = unique_count == 2 and is_numeric

            if is_numeric:
                try:
                    variance = float(series.var())
                    skewness = float(series.skew())
                    kurtosis = float(series.kurtosis())
                    if unique_count > 0:
                        dominant_pct = series.value_counts().iloc[0] / total_count
                    else:
                        dominant_pct = 1.0
                except:
                    variance = skewness = kurtosis = dominant_pct = np.nan
            else:
                variance = skewness = kurtosis = np.nan
                try:
                    dominant_pct = series.value_counts().iloc[0] / total_count
                except:
                    dominant_pct = 1.0

            feature_type = self._classify_feature_type(col, desc, series)

            quality = self._calculate_quality_score(
                missing_pct, unique_count, total_count, dominant_pct, is_numeric
            )

            features.append({
                'variable': col,
                'description': desc[:150],
                'feature_type': feature_type,
                'is_numeric': is_numeric,
                'is_categorical': is_categorical,
                'is_binary': is_binary,
                'missing_pct': missing_pct,
                'unique_count': unique_count,
                'dominant_value_pct': dominant_pct,
                'variance': variance,
                'skewness': skewness,
                'kurtosis': kurtosis,
                'quality_score': quality,
                'usable_for_modeling': quality > 0.3 and missing_pct < 0.9
            })

        return pd.DataFrame(features).sort_values('quality_score', ascending=False)

    def _classify_feature_type(self, name: str, description: str, series: pd.Series) -> str:
        """Classify feature into business categories"""
        text = f"{name} {description}".lower()

        patterns = {
            'demographic': ['age', 'gender', 'income', 'education', 'occupation', 'marital'],
            'financial': ['balance', 'amount', 'limit', 'credit', 'debt', 'payment', 'transaction'],
            'behavioral': ['frequency', 'count', 'number', 'times', 'usage', 'activity'],
            'temporal': ['date', 'time', 'duration', 'days', 'months', 'years', 'period'],
            'identifier': ['id', 'code', 'number', 'reference', 'key'],
            'geographic': ['address', 'city', 'state', 'country', 'zip', 'location'],
            'product': ['product', 'service', 'account', 'loan', 'card'],
            'risk': ['score', 'rating', 'grade', 'risk', 'probability']
        }

        for category, keywords in patterns.items():
            if any(keyword in text for keyword in keywords):
                return category

        return 'other'

    def _calculate_quality_score(self, missing_pct: float, unique_count: int,
                                 total_count: int, dominant_pct: float, is_numeric: bool) -> float:
        """Calculate overall feature quality score"""
        score = 1.0

        if missing_pct > 0.8:
            score *= 0.1
        elif missing_pct > 0.5:
            score *= 0.5
        elif missing_pct > 0.2:
            score *= 0.8

        if unique_count <= 1:
            score *= 0.0
        elif dominant_pct > 0.99:
            score *= 0.2
        elif dominant_pct > 0.95:
            score *= 0.5

        if unique_count > 1:
            uniqueness_ratio = unique_count / total_count
            if is_numeric:
                if 0.01 <= uniqueness_ratio <= 1.0:
                    score *= 1.0
                else:
                    score *= 0.8
            else:
                if 0.001 <= uniqueness_ratio <= 0.1:
                    score *= 1.0
                else:
                    score *= 0.7

        return max(0.0, min(1.0, score))

    def select_features_for_labels(self, df: pd.DataFrame, labels: pd.DataFrame,
                                   features_df: pd.DataFrame,
                                   guard_set: Optional[set] = None) -> Dict[str, pd.DataFrame]:
        """Select best features for each label using multiple methods"""
        print("🎯 Selecting optimal features for each label...")

        usable_features = features_df[features_df['usable_for_modeling']].copy()
        feature_cols = [col for col in usable_features['variable'] if col in df.columns]

        if guard_set:
            before = len(feature_cols)
            feature_cols = [c for c in feature_cols if c not in guard_set]
            print(f"   Guarded features removed: {before - len(feature_cols)}")

        if len(feature_cols) > self.config.max_features_analysis:
            top_features = usable_features.head(self.config.max_features_analysis)
            feature_cols = [col for col in top_features['variable']
                            if col in df.columns and (not guard_set or col not in guard_set)]

        print(f"   Analyzing {len(feature_cols)} features against {len(labels.columns)} labels")

        results = {}
        for label_col in tqdm(labels.columns, desc="Feature selection per label"):
            y = labels[label_col]
            if y.sum() == 0:
                continue

            X = df[feature_cols].copy()
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_processed = pd.DataFrame(index=X.index)

            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median')
                X_numeric = imputer.fit_transform(X[numeric_cols])
                X_processed[numeric_cols] = X_numeric

            cat_cols = X.select_dtypes(include=['object']).columns
            for col in cat_cols:
                top_cats = X[col].value_counts().head(10).index
                for cat in top_cats:
                    X_processed[f'{col}_{cat}'] = (X[col] == cat).astype(int)

            if len(X_processed.columns) == 0:
                continue

            try:
                var_series = X_processed.var(numeric_only=True)
                X_processed_pruned, dropped_local = prune_correlated(
                    X_processed, corr_thr=self.config.max_correlation_threshold,
                    keep_priority=var_series
                )
                if len(dropped_local) > 0:
                    print(f"   [corr-prune] Dropped {len(dropped_local)} encoded features for label={label_col}")
                X_use = X_processed_pruned
            except Exception:
                X_use = X_processed

            feature_scores = self._multi_method_selection(X_use, y)
            results[label_col] = feature_scores

        return results

    def _multi_method_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Apply multiple feature selection methods"""
        scores = pd.DataFrame({'feature': X.columns})

        try:
            mi_scores = mutual_info_classif(X, y, random_state=self.config.random_seed)
            scores['mutual_info'] = mi_scores
        except:
            scores['mutual_info'] = 0.0

        try:
            f_scores, _ = f_classif(X, y)
            scores['f_statistic'] = pd.Series(f_scores).fillna(0).values
        except:
            scores['f_statistic'] = 0.0

        try:
            corr_scores = [abs(spearmanr(X[col], y)[0]) for col in X.columns]
            scores['correlation'] = [x if not np.isnan(x) else 0 for x in corr_scores]
        except:
            scores['correlation'] = 0.0

        try:
            if y.sum() > 10 and len(X) > 50:
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
                lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1,
                                        random_state=self.config.random_seed)
                lr.fit(X_scaled, y)
                scores['l1_coef'] = np.abs(lr.coef_[0])
            else:
                scores['l1_coef'] = 0.0
        except:
            scores['l1_coef'] = 0.0

        for col in ['mutual_info', 'f_statistic', 'correlation', 'l1_coef']:
            if np.max(scores[col]) > 0:
                scores[col] = scores[col] / np.max(scores[col])

        scores['combined_score'] = (
            scores['mutual_info'] * 0.3 +
            scores['f_statistic'] * 0.25 +
            scores['correlation'] * 0.25 +
            scores['l1_coef'] * 0.2
        )

        return scores.sort_values('combined_score', ascending=False)

# =============================================
# MAIN FRAMEWORK
# =============================================

def build_leakage_guard(header_cols: List[str], descriptions: Dict[str, str],
                        label_sources: List[str], outcome_guard_terms: Tuple[str, ...]) -> set:
    """
    Build a guard set with:
      1) the explicit label sources
      2) any column whose text matches outcome terms
      3) full 'families' of label sources (e.g., var201xxx -> all var201***)
      4) ID-like & near-unique columns (added later once df is available)
    """
    pat = re.compile("|".join(outcome_guard_terms), re.IGNORECASE)
    guard = set(label_sources)

    # outcome-like by text
    for c in header_cols:
        txt = (c + " " + descriptions.get(c, "")).lower()
        if pat.search(txt) or is_outcome_like(c, descriptions.get(c, "")):
            guard.add(c)

    # guard whole families of label sources
    label_fams = {variable_family(c) for c in label_sources}
    for c in header_cols:
        fam = variable_family(c)
        if fam in label_fams:
            guard.add(c)

    return guard


def jaccard_series(a: pd.Series, b: pd.Series) -> float:
    A = a.astype(bool).values; B = b.astype(bool).values
    inter = (A & B).sum(); union = (A | B).sum()
    return float(inter/union) if union > 0 else 0.0

class SmartVariableFramework:
    """Main framework orchestrating the entire analysis"""

    def __init__(self, config: FrameworkConfig = None):
        self.config = config or FrameworkConfig()
        self.label_identifier = LabelIdentifier(self.config)
        self.label_builder = CompositeLabelBuilder(self.config)
        self.feature_selector = SmartFeatureSelector(self.config)

    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """Load and prepare all data sources"""
        print("📁 Loading data sources...")

        print(f"   Loading raw data from {self.config.raw_data_path}...")
        df = pd.read_csv(self.config.raw_data_path, low_memory=False)

        if len(df) > self.config.max_rows_analysis:
            df = df.sample(n=self.config.max_rows_analysis, random_state=self.config.random_seed)
            print(f"   Sampled to {len(df):,} rows for analysis")

        print(f"   Loaded: {len(df):,} rows × {len(df.columns)} columns")

        descriptions = {}

        if Path(self.config.dictionary_path).exists():
            print(f"   Loading variable dictionary from {self.config.dictionary_path}...")
            try:
                excel_file = pd.ExcelFile(self.config.dictionary_path)
                sheet_sizes = {sheet: pd.read_excel(self.config.dictionary_path, sheet_name=sheet).shape[0]
                               for sheet in excel_file.sheet_names}
                best_sheet = max(sheet_sizes, key=sheet_sizes.get)
                dict_df = pd.read_excel(self.config.dictionary_path, sheet_name=best_sheet)

                var_col = None
                exp_col = None
                for col in dict_df.columns:
                    col_lower = str(col).lower()
                    if any(term in col_lower for term in ['variable', 'field', 'column', 'name']):
                        var_col = col
                    elif any(term in col_lower for term in ['explanation', 'description', 'meaning', 'detail']):
                        exp_col = col

                if var_col and exp_col:
                    for _, row in dict_df.iterrows():
                        if pd.notna(row[var_col]) and pd.notna(row[exp_col]):
                            descriptions[str(row[var_col])] = str(row[exp_col])
                    print(f"   Loaded {len(descriptions)} variable descriptions")
                else:
                    print("   ⚠️ Could not find variable/explanation columns in dictionary")
            except Exception as e:
                print(f"   ⚠️ Could not load dictionary: {e}")

        if Path(self.config.variable_catalog_path).exists():
            print(f"   Loading additional descriptions from {self.config.variable_catalog_path}...")
            try:
                catalog_df = pd.read_csv(self.config.variable_catalog_path)
                if 'Variable' in catalog_df.columns and 'Description' in catalog_df.columns:
                    for _, row in catalog_df.iterrows():
                        if pd.notna(row['Variable']) and pd.notna(row['Description']):
                            if str(row['Variable']) not in descriptions:
                                descriptions[str(row['Variable'])] = str(row['Description'])
                    print(f"   Total descriptions available: {len(descriptions)}")
            except Exception as e:
                print(f"   ⚠️ Could not load catalog: {e}")

        return df, descriptions

    def run_analysis(self) -> Dict[str, Any]:
        """Run the complete smart variable analysis"""
        print("🚀 Starting Smart Variable Framework Analysis...")
        print("=" * 60)

        # Load data
        df, descriptions = self.load_data()

        # Step 1A: Targeted scan for negative pattern variables
        print("\n" + "="*50)
        print("STEP 1A: NEGATIVE PATTERN VARIABLE DISCOVERY")
        print("="*50)
        negative_pattern_vars = self.label_identifier.find_negative_pattern_variables(df, descriptions)
        
        # Create interim directory for outputs
        import os
        interim_dir = "/home/vhsingh/Parshvi_project/data/interim"
        os.makedirs(interim_dir, exist_ok=True)
        
        negative_pattern_vars.to_csv(f"{interim_dir}/negative_pattern_variables.csv", index=False)
        print(f"💾 Saved negative pattern analysis to {interim_dir}/negative_pattern_variables.csv")

        high_priority_negatives = negative_pattern_vars[negative_pattern_vars['recommended_for_label']]
        print(f"📊 Found {len(high_priority_negatives)} high-priority negative pattern variables")

        # Step 1B: General label identification
        print("\n" + "="*50)
        print("STEP 1B: COMPREHENSIVE LABEL IDENTIFICATION")
        print("="*50)
        label_candidates = self.label_identifier.analyze_candidates(df, descriptions)
        label_candidates.to_csv(f"{interim_dir}/smart_label_candidates.csv", index=False)
        print(f"💾 Saved general label analysis to {interim_dir}/smart_label_candidates.csv")

        eligible_labels = label_candidates[label_candidates['eligible_for_label']]
        print(f"📊 Found {len(eligible_labels)} eligible label variables")

        # Step 2: Build composite labels
        print("\n" + "="*50)
        print("STEP 2: COMPOSITE LABEL CREATION")
        print("="*50)
        labels_df, label_meta = self.label_builder.create_label_variants(df, label_candidates, negative_pattern_vars)

        header_cols = df.columns.tolist()
        label_sources = [c for c in label_meta.keys()]
        guard_set = build_leakage_guard(header_cols, descriptions, label_sources, self.config.outcome_guard_terms)

        external_guard = set()
        for fname in ["do not use features.txt", "do_not_use_features.txt"]:
            p = Path(fname)
            if p.exists():
                for ln in p.read_text().splitlines():
                    ln = ln.strip()
                    if ln:
                        external_guard.add(ln)
        if external_guard:
            guard_set |= external_guard
            print(f"🛡️  Merged external guard entries: {len(external_guard)} (total guard={len(guard_set)})")

        if 'label_union' in labels_df.columns:
            lu = labels_df['label_union']
            pos_mask = lu == 1
            contrib_rows = []
            for ev in label_sources:
                if ev in df.columns:
                    s = df[ev]
                    s = pd.to_numeric(s, errors='coerce') if not pd.api.types.is_numeric_dtype(s) else s
                    b = (s.fillna(0) > 0).astype(int)
                    share = float(b[pos_mask].mean()) if pos_mask.any() else 0.0
                    contrib_rows.append((ev, share, descriptions.get(ev, "")[:120]))
            contrib_df = pd.DataFrame(contrib_rows, columns=["event","share_among_positives","description"]).sort_values("share_among_positives", ascending=False)
            contrib_df.to_csv(f"{interim_dir}/event_contribution_summary.csv", index=False)

            # Overlap matrix
            jvars = [ev for ev in label_sources if ev in df.columns]
            J = pd.DataFrame(index=jvars, columns=jvars, dtype=float)
            for i in range(len(jvars)):
                si = (pd.to_numeric(df[jvars[i]], errors='coerce').fillna(0) > 0).astype(int)
                for j in range(i, len(jvars)):
                    sj = (pd.to_numeric(df[jvars[j]], errors='coerce').fillna(0) > 0).astype(int)
                    J.iloc[i, j] = J.iloc[j, i] = jaccard_series(si, sj)
            J.to_csv(f"{interim_dir}/jaccard_matrix.csv")

        if 'label_weighted' in labels_df.columns:
            thr = labels_df.attrs.get('weighted_threshold', None)
            with open(f"{interim_dir}/weighted_label_meta.json","w") as f:
                json.dump({"weighted_threshold": thr,
                           "target_prev": float(labels_df['label_union'].mean() if 'label_union' in labels_df.columns else np.nan)}, f, indent=2)

        with open(f"{interim_dir}/do_not_use_features.txt","w") as f:
            for g in sorted(guard_set):
                f.write(g + "\n")
        print(f"🛡️  Leakage guard size: {len(guard_set)}")

        if len(labels_df) > 0:
            labels_df.to_csv(f"{interim_dir}/composite_labels.csv", index=False)
            print(f"💾 Saved {len(labels_df.columns)} label variants to {interim_dir}/composite_labels.csv")

            print("\n📈 Label Variant Statistics:")
            for col in labels_df.columns:
                pos_rate = labels_df[col].mean()
                pos_count = labels_df[col].sum()
                print(f"   {col:25s}: {pos_rate:6.4f} ({pos_count:,} positives)")
        else:
            print("⚠️ No composite labels could be created")
            return {'status': 'failed', 'reason': 'No eligible label variables found'}

        # === [NEW BLOCK] Time-safety & additional guards ====================================
        # 1) For Mode A (snapshot): forbid lifetime features from feature set if labels use lifetime
        label_windows_used = []
        for ev, m in label_meta.items():
            label_windows_used.append(infer_window(ev, m.get('description', '')))
        label_windows_used = set(w for w in label_windows_used if w != "unknown")

        forbidden_windows = set()
        if "lt" in label_windows_used or self.config.include_lifetime_in_label:
            # forbid lifetime (lt) features to reduce leakage
            forbidden_windows.add("lt")

        # Also forbid outcome-like features by text (this duplicates text guard, but keeps mask explicit)
        feature_cols_all = [c for c in df.columns if c not in labels_df.columns]  # exclude label columns
        feature_cols_all = [c for c in feature_cols_all if c not in guard_set]    # first-pass guard

        mask = time_safe_feature_mask(feature_cols_all, descriptions, forbidden_windows)
        feature_cols_time_safe = [feature_cols_all[i] for i, keep in enumerate(mask) if keep]

        # 2) Add ID-like & near-unique guards
        guard_set |= id_like_columns(df.columns)
        guard_set |= near_unique_columns(df)

        # 3) Optional "too-predictive" smell-test guard (uses label_union if present)
        try:
            y_for_guard = None
            for preferred in ["label_union", "label_hierarchical", "label_weighted", "label_clustered"]:
                if preferred in labels_df.columns:
                    y_for_guard = labels_df[preferred]
                    break
            if y_for_guard is not None:
                numX_guard = df[feature_cols_time_safe].select_dtypes(include=['number'])
                guard_set |= too_predictive_guard(numX_guard, y_for_guard, max_auc=0.90)
        except Exception:
            pass

        # Make the final candidate feature list now (after guards)
        feature_cols_time_safe = [c for c in feature_cols_time_safe if c not in guard_set]
        Path(f"{interim_dir}/do_not_use_features.txt").write_text("\n".join(sorted(guard_set)))
        print(f"🛡️  Leakage guard size (post time/ID/unique/predictive): {len(guard_set)}")
        print(f"✅ Time-safe feature candidates: {len(feature_cols_time_safe)}")
        # ===============================================================================

        # Step 3: Feature quality
        print("\n" + "="*50)
        print("STEP 3: FEATURE QUALITY ANALYSIS")
        print("="*50)
        feature_quality = self.feature_selector.analyze_feature_quality(df, descriptions)
        feature_quality.to_csv(f"{interim_dir}/variable_quality_report.csv", index=False)
        print(f"💾 Saved feature analysis to {interim_dir}/variable_quality_report.csv")

        usable_features = feature_quality[feature_quality['usable_for_modeling']]
        print(f"📊 {len(usable_features)} features suitable for modeling (out of {len(feature_quality)})")

        # === [NEW BLOCK] Correlation pruning on numeric features ===================
        print("\n" + "="*50)
        print("STEP 3B: CORRELATION PRUNING")
        print("="*50)

        # Keep only features that passed time/guard filters and are "usable_for_modeling"
        feature_quality = feature_quality.copy()
        usable_mask = feature_quality['usable_for_modeling'] & feature_quality['variable'].isin(feature_cols_time_safe)
        usable = feature_quality[usable_mask].copy()

        # Apply your fill-rate rule: Fill rate ≥ 0.85 (i.e., missing ≤ 0.15)
        usable = usable[usable['missing_pct'] <= (1.0 - 0.85)]

        # Numeric-only for pruning
        num_keep = [c for c in usable['variable'] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

        # Impute quickly to compute corr
        imp_vals = df[num_keep].copy()
        imp_vals = imp_vals.fillna(imp_vals.median(numeric_only=True))

        # Use quality_score as priority to keep the better one in a correlated pair
        priority = usable.set_index('variable')['quality_score']

        X_pruned, dropped_corr = prune_correlated(imp_vals, corr_thr=self.config.max_correlation_threshold, keep_priority=priority)

        kept_features = X_pruned.columns.tolist()
        dropped_features = sorted(set(num_keep) - set(kept_features))

        # Persist keep/drop lists for training scripts
        Path(f"{interim_dir}/feature_keep_list.txt").write_text("\n".join(kept_features))
        Path(f"{interim_dir}/feature_drop_corr.txt").write_text("\n".join(dropped_features))
        print(f"🔗 Correlation pruning: kept={len(kept_features)}, dropped={len(dropped_features)} (> r={self.config.max_correlation_threshold})")
        # ================================================================================


        # Step 4: Feature selection
        print("\n" + "="*50)
        print("STEP 4: FEATURE SELECTION")
        print("="*50)
        feature_selections = self.feature_selector.select_features_for_labels(df, labels_df, feature_quality, guard_set=guard_set)

        if feature_selections:
            all_features = set()
            for label_features in feature_selections.values():
                all_features.update(label_features['feature'])

            feature_matrix = pd.DataFrame({'feature': list(all_features)})

            for label_name, label_features in feature_selections.items():
                feature_matrix = feature_matrix.merge(
                    label_features[['feature', 'combined_score']].rename(columns={'combined_score': f'{label_name}_score'}),
                    on='feature', how='left'
                )

            score_cols = [col for col in feature_matrix.columns if col.endswith('_score')]
            feature_matrix[score_cols] = feature_matrix[score_cols].fillna(0)
            feature_matrix['avg_importance'] = feature_matrix[score_cols].mean(axis=1)

            feature_matrix = feature_matrix.sort_values('avg_importance', ascending=False)
            feature_matrix.to_csv(f"{interim_dir}/feature_importance_matrix.csv", index=False)
            print(f"💾 Saved feature importance matrix to {interim_dir}/feature_importance_matrix.csv")

        # Step 5: Recommendations
        print("\n" + "="*50)
        print("STEP 5: RECOMMENDATIONS")
        print("="*50)
        recommendations = self._generate_recommendations(
            label_candidates, labels_df, feature_quality, feature_selections, negative_pattern_vars
        )

        with open(f"{interim_dir}/recommended_pipeline.json", "w") as f:
            json.dump(recommendations, f, indent=2, default=str)
        print(f"💾 Saved recommendations to {interim_dir}/recommended_pipeline.json")

        self._generate_report(label_candidates, labels_df, feature_quality, recommendations, negative_pattern_vars, interim_dir)

        print("\n✅ Smart Variable Framework Analysis Complete!")
        print("="*60)

        return {
            'status': 'success',
            'label_candidates': len(label_candidates),
            'eligible_labels': len(eligible_labels),
            'label_variants': len(labels_df.columns) if len(labels_df) > 0 else 0,
            'usable_features': len(usable_features),
            'recommendations': recommendations
        }

    def _generate_recommendations(self, label_candidates: pd.DataFrame, labels_df: pd.DataFrame,
                                  feature_quality: pd.DataFrame, feature_selections: Dict,
                                  negative_pattern_vars: pd.DataFrame = None) -> Dict[str, Any]:
        """Generate actionable recommendations"""

        recommendations = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'best_label': None,
            'top_features': [],
            'negative_pattern_findings': {},
            'data_quality_issues': [],
            'next_steps': []
        }

        if negative_pattern_vars is not None and len(negative_pattern_vars) > 0:
            high_priority = negative_pattern_vars[negative_pattern_vars['recommended_for_label']]
            all_patterns = negative_pattern_vars['matched_patterns'].str.split(';').explode().str.strip()
            pattern_counts = all_patterns.value_counts().head(10).to_dict()

            recommendations['negative_pattern_findings'] = {
                'total_variables_found': len(negative_pattern_vars),
                'high_priority_variables': len(high_priority),
                'most_common_patterns': pattern_counts,
                'top_negative_variables': high_priority.head(5)[['variable', 'priority_score', 'positive_rate']].to_dict('records') if len(high_priority) > 0 else []
            }

        if len(labels_df) > 0:
            label_stats = []
            for col in labels_df.columns:
                pos_rate = labels_df[col].mean()
                if 0.01 <= pos_rate <= 0.25:
                    label_stats.append((col, pos_rate))

            if label_stats:
                preferred_order = ['label_union', 'label_hierarchical', 'label_weighted', 'label_clustered']
                for pref in preferred_order:
                    if any(pref in label for label, _ in label_stats):
                        recommendations['best_label'] = next(label for label, _ in label_stats if pref in label)
                        break

                if not recommendations['best_label']:
                    label_stats.sort(key=lambda x: abs(x[1] - 0.1))
                    recommendations['best_label'] = label_stats[0][0]

        if feature_selections and recommendations['best_label'] in feature_selections:
            best_label_features = feature_selections[recommendations['best_label']]
            top_features = best_label_features.head(20)
            recommendations['top_features'] = top_features[['feature', 'combined_score']].to_dict('records')

        high_missing = feature_quality[feature_quality['missing_pct'] > 0.5]
        low_variance = feature_quality[feature_quality['dominant_value_pct'] > 0.95]

        if len(high_missing) > 0:
            recommendations['data_quality_issues'].append(f"{len(high_missing)} features have >50% missing data")
        if len(low_variance) > 0:
            recommendations['data_quality_issues'].append(f"{len(low_variance)} features have very low variance")

        recommendations['next_steps'] = [
            f"Use '{recommendations['best_label']}' as your target variable",
            f"Start with top {min(50, len(recommendations['top_features']))} features for initial modeling",
            "Consider feature engineering for variables with high missing rates",
            "Validate label definition with business stakeholders",
            "Implement time-based validation if temporal data is available"
        ]

        return recommendations

    def _generate_report(self, label_candidates: pd.DataFrame, labels_df: pd.DataFrame,
                         feature_quality: pd.DataFrame, recommendations: Dict,
                         negative_pattern_vars: pd.DataFrame = None, interim_dir: str = None):
        """Generate comprehensive markdown report"""

        report_path = f"{interim_dir}/smart_framework_report.md" if interim_dir else "smart_framework_report.md"
        with open(report_path, "w") as f:
            f.write("# Smart Variable Framework - Analysis Report\n\n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            eligible_labels = label_candidates[label_candidates['eligible_for_label']]
            usable_features = feature_quality[feature_quality['usable_for_modeling']]

            f.write(f"- **Label Variables Found:** {len(eligible_labels)} eligible out of {len(label_candidates)} analyzed\n")
            f.write(f"- **Feature Variables:** {len(usable_features)} usable out of {len(feature_quality)} analyzed\n")
            f.write(f"- **Recommended Label:** `{recommendations.get('best_label', 'None')}`\n")
            f.write(f"- **Top Features:** {len(recommendations.get('top_features', []))}\n")

            neg_findings = recommendations.get('negative_pattern_findings', {})
            if neg_findings:
                f.write(f"- **Negative Pattern Variables:** {neg_findings.get('total_variables_found', 0)} found, {neg_findings.get('high_priority_variables', 0)} high-priority\n")
            f.write("\n")

            if negative_pattern_vars is not None and len(negative_pattern_vars) > 0:
                f.write("## 🎯 Negative Pattern Analysis (Based on Your Research)\n\n")
                f.write("Variables containing patterns like: `over[-\\s]?limit`, `default`, `declin\\w*`, `reject\\w*`, `insufficient`, `penalt\\w*`, `miss(ed|ing)`, `overdue`, `min[_\\s-]?due|mindue`, `\\bdue\\b`, `negative\\w*|noofnegativeevents`.\n\n")

                high_priority = negative_pattern_vars[negative_pattern_vars['recommended_for_label']]
                f.write("### Key Findings\n\n")
                f.write(f"- **Total Variables Matching Patterns:** {len(negative_pattern_vars)}\n")
                f.write(f"- **High-Priority Label Candidates:** {len(high_priority)}\n")
                f.write(f"- **Average Priority Score:** {negative_pattern_vars['priority_score'].mean():.3f}\n\n")

                if len(high_priority) > 0:
                    f.write("### Top Negative Pattern Variables for Labeling\n\n")
                    f.write("| Variable | Matched Patterns | Priority Score | Positive Rate | Description |\n")
                    f.write("|----------|------------------|----------------|---------------|-------------|\n")
                    for _, row in high_priority.head(10).iterrows():
                        desc = row['description'][:50] + "..." if len(row['description']) > 50 else row['description']
                        patterns_matched = row['matched_patterns'][:50] + "..." if len(row['matched_patterns']) > 50 else row['matched_patterns']
                        f.write(f"| {row['variable']} | {patterns_matched} | {row['priority_score']:.3f} | {row['positive_rate']:.4f} | {desc} |\n")
                    f.write("\n")

            f.write("## Label Analysis\n\n")
            if len(labels_df) > 0:
                f.write("### Label Variant Performance\n\n")
                f.write("| Label Variant | Positive Rate | Count | Description |\n")
                f.write("|---------------|---------------|-------|-------------|\n")
                for col in labels_df.columns:
                    pos_rate = labels_df[col].mean()
                    pos_count = labels_df[col].sum()
                    f.write(f"| {col} | {pos_rate:.4f} | {pos_count:,} | Auto-generated composite label |\n")
                f.write("\n")

            f.write("### Top Label Candidates\n\n")
            f.write("| Variable | Outcome Score | Type | Positive Rate | Description |\n")
            f.write("|----------|---------------|------|---------------|-------------|\n")
            eligible_labels = label_candidates[label_candidates['eligible_for_label']]
            for _, row in eligible_labels.head(10).iterrows():
                desc = row['description'][:50] + "..." if len(row['description']) > 50 else row['description']
                f.write(f"| {row['variable']} | {row['outcome_score']:.2f} | {row['outcome_type']} | {row['positive_rate']:.4f} | {desc} |\n")
            f.write("\n")

            f.write("## Feature Analysis\n\n")
            f.write("### Feature Quality Distribution\n\n")
            quality_bins = pd.cut(feature_quality['quality_score'], bins=[0, 0.3, 0.6, 0.8, 1.0],
                                  labels=['Poor', 'Fair', 'Good', 'Excellent'])
            quality_counts = quality_bins.value_counts()

            for quality, count in quality_counts.items():
                f.write(f"- **{quality}:** {count} features\n")
            f.write("\n")

            if recommendations.get('top_features'):
                f.write("### Top Recommended Features\n\n")
                f.write("| Rank | Feature | Importance Score |\n")
                f.write("|------|---------|------------------|\n")
                for i, feature in enumerate(recommendations['top_features'][:15], 1):
                    f.write(f"| {i} | {feature['feature']} | {feature['combined_score']:.4f} |\n")
                f.write("\n")

            f.write("## Data Quality Issues\n\n")
            if recommendations.get('data_quality_issues'):
                for issue in recommendations['data_quality_issues']:
                    f.write(f"- ⚠️ {issue}\n")
            else:
                f.write("- ✅ No major data quality issues detected\n")
            f.write("\n")

            f.write("## Recommendations\n\n")
            f.write("### Immediate Next Steps\n\n")
            for i, step in enumerate(recommendations.get('next_steps', []), 1):
                f.write(f"{i}. {step}\n")
            f.write("\n")

            f.write("### Implementation Guidance\n\n")
            f.write("```python\n")
            f.write("# Use this configuration in your modeling pipeline\n")
            f.write(f"TARGET_LABEL = '{recommendations.get('best_label', 'label_union')}'\n")
            f.write("TOP_FEATURES = [\n")
            for feature in recommendations.get('top_features', [])[:10]:
                f.write(f"    '{feature['feature']}',\n")
            f.write("]\n")
            f.write("```\n\n")

            f.write("## Files Generated\n\n")
            f.write("- `negative_pattern_variables.csv` - Variables matching negative-word research\n")
            f.write("- `smart_label_candidates.csv` - Detailed analysis of all potential label variables\n")
            f.write("- `composite_labels.csv` - Generated label variants for comparison\n")
            f.write("- `variable_quality_report.csv` - Comprehensive feature quality assessment\n")
            f.write("- `feature_importance_matrix.csv` - Feature importance scores for each label\n")
            f.write("- `recommended_pipeline.json` - Machine-readable recommendations\n")
            f.write("- `smart_framework_report.md` - This comprehensive report\n")

        print(f"💾 Saved comprehensive report to {report_path}")

# =============================================
# HYPERPARAMETER SWEEP
# =============================================

@dataclass
class SweepConfig:
    enable: bool = False
    max_configs: int = 120
    random_seed: int = 42
    save_top_k: int = 5
    top_k_baseline_features: int = 400
    n_cv_folds: int = 3
    search_space: Optional[Dict[str, List[Any]]] = None

class HyperparameterSweep:
    def __init__(self, base_config: FrameworkConfig):
        self.base_config = base_config

    def _default_search_space(self) -> Dict[str, List[Any]]:
        return {
            "include_lifetime_in_label": [False, True],
            "dominance_cutoff": [0.80, 0.70, 0.60, 0.50],
            "dedup_jaccard_threshold": [0.95, 0.90, 0.85],
            "outcome_importance_threshold": [0.60, 0.70, 0.80],
            "quality_threshold_for_label_eligibility": [0.40, 0.50, 0.60],
            "min_label_prevalence": [0.005, 0.010],
            "target_weighted_label_prevalence": [None, 0.05, 0.08, 0.10, 0.15],
            "rescue_min_prevalence": [0.001, 0.002, 0.005],
            "rescue_top_k": [8, 12, 16],
            "severe_weight_threshold_for_union": [0.60, 0.80, 0.90],
        }

    def _gen_random_configs(self, sweep: SweepConfig, space: Dict[str, List[Any]]) -> List[FrameworkConfig]:
        rng = random.Random(sweep.random_seed)
        keys = list(space.keys())
        configs = []
        for _ in range(sweep.max_configs):
            cfg = FrameworkConfig(**vars(self.base_config))
            for k in keys:
                setattr(cfg, k, rng.choice(space[k]))
            cfg.random_seed = rng.randint(1, 10_000_000)
            configs.append(cfg)
        return configs

    def _config_id(self, cfg: FrameworkConfig) -> str:
        keys = sorted([
            "include_lifetime_in_label", "dominance_cutoff", "dedup_jaccard_threshold",
            "outcome_importance_threshold", "quality_threshold_for_label_eligibility",
            "min_label_prevalence", "target_weighted_label_prevalence",
            "rescue_min_prevalence", "rescue_top_k", "severe_weight_threshold_for_union"
        ])
        s = "|".join(f"{k}={getattr(cfg,k)}" for k in keys)
        return hashlib.md5(s.encode("utf-8")).hexdigest()[:10]

    def run(self, sweep: SweepConfig):
        os.makedirs("sweep", exist_ok=True)
        results_path = Path("sweep/sweep_results.csv")
        best_dir = Path("sweep/best")
        best_dir.mkdir(parents=True, exist_ok=True)

        # Load once
        fw = SmartVariableFramework(self.base_config)
        df, descriptions = fw.load_data()
        li = LabelIdentifier(self.base_config)
        negative_pattern_vars = li.find_negative_pattern_variables(df, descriptions)

        # Global feature quality once (to speed up)
        fs = SmartFeatureSelector(self.base_config)
        feature_quality = fs.analyze_feature_quality(df, descriptions)
        num_cols = df.select_dtypes(include=[np.number]).columns
        numeric_usable = feature_quality[(feature_quality['usable_for_modeling']) &
                                         (feature_quality['variable'].isin(num_cols))] \
                                         .sort_values('quality_score', ascending=False)

        space = sweep.search_space or self._default_search_space()
        cfgs = self._gen_random_configs(sweep, space)

        all_rows = []
        start = time.time()
        for idx, cfg in enumerate(cfgs, 1):
            t0 = time.time()
            cid = self._config_id(cfg)
            status = "ok"
            try:
                li_cfg = LabelIdentifier(cfg)
                cand = li_cfg.analyze_candidates(df, descriptions)

                lb = CompositeLabelBuilder(cfg)
                labels_df, label_meta = lb.create_label_variants(df, cand, negative_pattern_vars)
                if labels_df is None or len(labels_df) == 0:
                    raise RuntimeError("no_labels_created")

                eval_label = None
                for pref in ["label_union", "label_hierarchical", "label_weighted", "label_clustered"]:
                    if pref in labels_df.columns:
                        eval_label = pref; break
                if not eval_label:
                    raise RuntimeError("no_eval_label")

                y = labels_df[eval_label].astype(int)
                prevalence = float(y.mean())
                pos = int(y.sum())
                n_sources = len(label_meta)

                def share_in_pos(col):
                    s = df[col]
                    s = pd.to_numeric(s, errors='coerce') if not pd.api.types.is_numeric_dtype(s) else s
                    b = (s.fillna(0) > 0).astype(int)
                    return float(b[y==1].mean()) if (y==1).any() else 0.0
                shares = [share_in_pos(c) for c in label_meta.keys()]
                max_share = max(shares) if shares else 0.0

                jvars = [c for c in label_meta.keys() if c in df.columns]
                jmax = 0.0; jmean = 0.0
                if len(jvars) >= 2:
                    vals = []
                    for i in range(len(jvars)):
                        si = (pd.to_numeric(df[jvars[i]], errors='coerce').fillna(0) > 0).astype(int)
                        for j in range(i+1, len(jvars)):
                            sj = (pd.to_numeric(df[jvars[j]], errors='coerce').fillna(0) > 0).astype(int)
                            inter = (si & sj).sum(); union = (si | sj).sum()
                            jac = float(inter/union) if union>0 else 0.0
                            vals.append(jac)
                    if vals:
                        jmax = float(np.max(vals)); jmean = float(np.mean(vals))

                guard = build_leakage_guard(df.columns.tolist(), descriptions,
                                            list(label_meta.keys()), cfg.outcome_guard_terms)

                base_feats = [v for v in numeric_usable['variable'].tolist()
                              if v not in guard][:sweep.top_k_baseline_features]
                if len(base_feats) < 5:
                    raise RuntimeError("too_few_features_after_guard")

                X = df[base_feats].copy()
                imputer = SimpleImputer(strategy="median")
                X = pd.DataFrame(imputer.fit_transform(X), index=df.index, columns=base_feats)

                skf = StratifiedKFold(n_splits=sweep.n_cv_folds, shuffle=True, random_state=cfg.random_seed)
                aucs, aps = [], []
                for tr, te in skf.split(X, y):
                    Xtr, Xte = X.iloc[tr], X.iloc[te]
                    ytr, yte = y.iloc[tr], y.iloc[te]
                    clf = LogisticRegression(penalty="l2", solver="liblinear",
                                             class_weight="balanced", max_iter=1000, random_state=cfg.random_seed)
                    clf.fit(Xtr, ytr)
                    p = clf.decision_function(Xte)
                    aucs.append(roc_auc_score(yte, p))
                    aps.append(average_precision_score(yte, clf.predict_proba(Xte)[:,1]))

                auc = float(np.mean(aucs))
                ap  = float(np.mean(aps))
                ap_norm = max(0.0, min(1.0, (ap - prevalence) / max(1e-6, 1 - prevalence)))
                base_score = 0.6*auc + 0.4*ap_norm

                pen = 0.0
                if n_sources < 2: pen += 0.15
                if max_share > 0.60: pen += (max_share - 0.60) * 1.5
                if prevalence < 0.01:
                    pen += min(0.10, (0.01 - prevalence) * 2.0)
                elif prevalence > 0.25:
                    pen += min(0.10, (prevalence - 0.25) * 0.5)

                score = base_score - pen

                row = {
                    "config_id": cid,
                    "status": status,
                    "score": round(score, 6),
                    "auc": round(auc, 6),
                    "ap": round(ap, 6),
                    "ap_norm": round(ap_norm, 6),
                    "prevalence": round(prevalence, 6),
                    "positives": pos,
                    "num_sources": n_sources,
                    "max_event_share": round(max_share, 6),
                    "jaccard_mean": round(jmean, 6),
                    "jaccard_max": round(jmax, 6),
                    "guard_size": len(guard),
                    "eval_label": eval_label,
                    "runtime_sec": round(time.time()-t0, 2),
                    # swept knobs
                    "include_lifetime_in_label": cfg.include_lifetime_in_label,
                    "dominance_cutoff": cfg.dominance_cutoff,
                    "dedup_jaccard_threshold": cfg.dedup_jaccard_threshold,
                    "outcome_importance_threshold": cfg.outcome_importance_threshold,
                    "quality_threshold_for_label_eligibility": cfg.quality_threshold_for_label_eligibility,
                    "min_label_prevalence": cfg.min_label_prevalence,
                    "target_weighted_label_prevalence": cfg.target_weighted_label_prevalence,
                    "rescue_min_prevalence": cfg.rescue_min_prevalence,
                    "rescue_top_k": cfg.rescue_top_k,
                    "severe_weight_threshold_for_union": cfg.severe_weight_threshold_for_union,
                }
                all_rows.append(row)

                pd.DataFrame(all_rows).sort_values("score", ascending=False).to_csv(results_path, index=False)
                print(f"[{idx}/{len(cfgs)}] {cid} → score={row['score']:.4f} (AUC={row['auc']:.3f}, prev={row['prevalence']:.3%}, sources={row['num_sources']}, maxshare={row['max_event_share']:.3f})")

            except Exception as e:
                row = {"config_id": cid, "status": f"fail:{str(e)}", "runtime_sec": round(time.time()-t0,2)}
                all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(results_path, index=False)
                continue

        res = pd.DataFrame(all_rows)
        res_ok = res[res['status']=="ok"].sort_values("score", ascending=False)
        if len(res_ok)==0:
            print("❌ No successful configs found in sweep.")
            return

        best = res_ok.iloc[0].to_dict()
        Path("sweep/best_config.json").write_text(json.dumps(best, indent=2))
        print(f"🏆 Best config: {best['config_id']}  score={best['score']:.4f}  AUC={best['auc']:.3f}  prev={best['prevalence']:.3%}")

        K = min(len(res_ok), sweep.save_top_k)
        for i in range(K):
            row = res_ok.iloc[i].to_dict()
            cid = row["config_id"]

            cfg = FrameworkConfig(**vars(self.base_config))
            for k in self._default_search_space().keys():
                cfg_val = row.get(k, getattr(cfg, k))
                setattr(cfg, k, cfg_val)

            li = LabelIdentifier(cfg)
            cand = li.analyze_candidates(df, descriptions)
            lb = CompositeLabelBuilder(cfg)
            labels_df, label_meta = lb.create_label_variants(df, cand, negative_pattern_vars)

            guard = build_leakage_guard(df.columns.tolist(), descriptions, list(label_meta.keys()), cfg.outcome_guard_terms)

            if 'label_union' in labels_df.columns:
                lu = labels_df['label_union']
                pos_mask = (lu==1)
                contrib = []
                for ev in label_meta.keys():
                    s = df[ev]
                    s = pd.to_numeric(s, errors='coerce') if not pd.api.types.is_numeric_dtype(s) else s
                    b = (s.fillna(0)>0).astype(int)
                    share = float(b[pos_mask].mean()) if pos_mask.any() else 0.0
                    contrib.append((ev, share, descriptions.get(ev, "")[:140]))
                contrib_df = pd.DataFrame(contrib, columns=["event","share_among_positives","description"]).sort_values("share_among_positives", ascending=False)
            else:
                contrib_df = pd.DataFrame(columns=["event","share_among_positives","description"])

            jvars = [ev for ev in label_meta.keys() if ev in df.columns]
            J = pd.DataFrame(index=jvars, columns=jvars, dtype=float)
            for a in range(len(jvars)):
                sa = (pd.to_numeric(df[jvars[a]], errors='coerce').fillna(0) > 0).astype(int)
                for b in range(a, len(jvars)):
                    sb = (pd.to_numeric(df[jvars[b]], errors='coerce').fillna(0) > 0).astype(int)
                    inter = (sa & sb).sum(); union = (sa | sb).sum()
                    jac = float(inter/union) if union>0 else 0.0
                    J.iloc[a,b] = J.iloc[b,a] = jac

            outdir = best_dir / f"rank_{i+1:02d}_{cid}"
            outdir.mkdir(parents=True, exist_ok=True)
            labels_df.to_csv(outdir / "composite_labels.csv", index=False)
            contrib_df.to_csv(outdir / "event_contribution_summary.csv", index=False)
            J.to_csv(outdir / "jaccard_matrix.csv")
            with open(outdir / "do_not_use_features.txt", "w") as f:
                for g in sorted(guard):
                    f.write(g+"\n")
            with open(outdir / "weighted_label_meta.json", "w") as f:
                json.dump({"weighted_threshold": labels_df.attrs.get("weighted_threshold", None),
                           "target_prev": float(labels_df['label_union'].mean() if 'label_union' in labels_df.columns else np.nan)},
                          f, indent=2)

        elapsed = time.time() - start
        print(f"✅ Sweep complete. Tried {len(cfgs)} configs in {elapsed/60:.1f} min.")
        print(f"   Results: {results_path}")
        print(f"   Best artifacts: {best_dir}")

# =============================================
# EXECUTION
# =============================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
    parser.add_argument("--sweep-max-configs", type=int, default=120)
    parser.add_argument("--sweep-save-top-k", type=int, default=5)
    parser.add_argument("--sweep-seed", type=int, default=42)
    args, _ = parser.parse_known_args()

    run_sweep = args.sweep or os.getenv("SMART_SWEEP", "0") == "1"

    config = FrameworkConfig()
    if run_sweep:
        sweep = SweepConfig(
            enable=True,
            max_configs=int(os.getenv("SWEEP_MAX_CONFIGS", args.sweep_max_configs)),
            save_top_k=int(os.getenv("SWEEP_SAVE_TOP_K", args.sweep_save_top_k)),
            random_seed=int(os.getenv("SWEEP_SEED", args.sweep_seed)),
            n_cv_folds=3,
        )
        HyperparameterSweep(config).run(sweep)
        return

    framework = SmartVariableFramework(config)
    try:
        results = framework.run_analysis()

        if results['status'] == 'success':
            print(f"\n🎉 Analysis completed successfully!")
            print(f"📊 Summary:")
            print(f"   - Label candidates: {results['label_candidates']}")
            print(f"   - Eligible labels: {results['eligible_labels']}")
            print(f"   - Label variants: {results['label_variants']}")
            print(f"   - Usable features: {results['usable_features']}")
            print(f"   - Recommended label: {results['recommendations'].get('best_label', 'None')}")
        else:
            print(f"\n❌ Analysis failed: {results.get('reason', 'Unknown error')}")
    except Exception as e:
        print(f"\n💥 Critical error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
