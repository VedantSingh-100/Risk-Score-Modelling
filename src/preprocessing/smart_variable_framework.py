#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart Variable Framework (Extended)
- Continuous event thresholds
- label_clustered primary + risk_score_0_100
- Lifetime excluded everywhere
- Window aggregates retained; literal timestamps dropped at feature time
- Stage-wise filtering; no forced label-source reductions
- Full diagnostics and plots
"""
import os, argparse, json, re, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mutual_info_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

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
ID_PAT = re.compile(
    r"(?:^|_|(?<![A-Za-z0-9]))(id|uuid|pan|aadhaar|account|acct|application|app(?:lication)?id|lead|mobile(_?no)?|phone|msisdn|email(_?address|_?id)?)(?:_|$|(?!=[A-Za-z0-9]))",
    re.I
)
TIME_LIKE_DROP_PAT = re.compile(
    r"(?:(refdate|asof|snapshot|pull|report|run)[-_]?(date|dt))|"
    r"\b(date|datetime|timestamp|time|ts)\b|"
    r"(^|_)(dow|weekday|week|woy|month|q(uarter)?|year|yr|day|hour|min(ute)?|sec(ond)?)(_|$)",
    re.I
)
TIME_LAG_KEEP_PAT = re.compile(r"(since|lag|delta|diff|tenure|vintage|age)", re.I)

def infer_window(name: str, desc: str) -> str:
    s = f"{name} {desc}".lower()
    for tag, pat in WINDOW_PATTERNS.items():
        if pat.search(s): return tag
    return "unknown"

def is_window_aggregate(name: str, desc: str) -> bool:
    return infer_window(name, desc) in {"1m","3m","6m","12m","lt"}

def is_outcome_like(name: str, desc: str) -> bool:
    s = f"{name} {desc}".lower()
    return bool(OUTCOME_PAT.search(s))

def variable_family(varname: str) -> str:
    v = str(varname).lower()
    m = re.match(r"var(\d{3})", v)
    if m: return f"var{m.group(1)}"
    return v.split("_")[0]

def id_like_columns(cols):
    return {c for c in cols if ID_PAT.search(str(c))}

def near_unique_columns(df: pd.DataFrame, thresh: float = 0.98):
    out = set(); n = len(df)
    for c in df.columns:
        try:
            s = df[c]
            if s.dtype == 'object' or pd.api.types.is_string_dtype(s):
                u = s.nunique(dropna=False)
                if u >= thresh * n: out.add(c); continue
                ratio_hexish = s.astype(str).str.match(r'^[0-9a-fA-F\-]{8,}$', na=False).mean()
                if ratio_hexish > 0.95: out.add(c); continue
            else:
                vals = pd.to_numeric(s, errors='coerce')
                if vals.notna().sum() == 0: continue
                int_like = np.nanmax(np.abs(vals - np.round(vals))) < 1e-8
                if int_like:
                    u = vals.nunique(dropna=True)
                    if u >= thresh * n and np.nanmax(np.abs(vals)) > 1e7:
                        out.add(c)
        except Exception:
            continue
    return out

def to_numeric_bool(series: pd.Series) -> pd.Series:
    if series.dtype in ['object', 'string']:
        s = series.astype(str).str.strip().str.lower()
        mapv = {
            'true':1, 'false':0, 'yes':1, 'no':0, 'y':1, 'n':0,
            't':1, 'f':0, '1':1, '0':0
        }
        s = s.map(mapv); s = pd.to_numeric(s, errors='coerce'); return s
    return pd.to_numeric(series, errors='coerce')

def transform_value(x: pd.Series, mode: str = "log1p", cap: float = 10.0) -> pd.Series:
    s = to_numeric_bool(x).astype(float)
    if mode == "log1p": z = np.clip(s, 0, None); return np.log1p(z.fillna(0))
    if mode == "asinh": z = np.clip(s, 0, None); return np.arcsinh(z.fillna(0))
    if mode == "signed_asinh": return np.arcsinh(s.fillna(0))
    if mode == "cap": z = np.clip(s, 0, None); return np.clip(z.fillna(0), 0, cap)
    if mode == "raw": return s.fillna(0)
    z = s.copy()
    if z.min() >= 0: return z.fillna(0) if z.max() <= 1.0 else np.log1p(z.fillna(0))
    return np.arcsinh(z.fillna(0))

def robust_normalize(v: pd.Series | np.ndarray, q_lo: float = 0.50, q_hi: float = 0.995) -> np.ndarray:
    s = pd.Series(v).astype(float)
    finite = s[np.isfinite(s)]
    if finite.empty: return np.zeros(len(s), dtype=float)
    lo = float(np.quantile(finite, q_lo)); hi = float(np.quantile(finite, q_hi))
    if not np.isfinite(hi - lo) or (hi - lo) <= 0: return np.zeros(len(s), dtype=float)
    z = (s.fillna(lo) - lo) / (hi - lo)
    return np.clip(z.values, 0.0, 1.0)

def prune_correlated(X: pd.DataFrame, corr_thr: float = 0.95, keep_priority: pd.Series | None = None):
    if X.shape[1] <= 1: return X.copy(), []
    corr = X.corr().abs(); upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        high = upper.index[upper[col] > corr_thr].tolist()
        for h in high:
            if h in to_drop or col in to_drop: continue
            if keep_priority is not None:
                ka = float(keep_priority.get(col, 0.0)); kb = float(keep_priority.get(h, 0.0))
                drop = col if ka < kb else h
            else: drop = max(col, h)
            to_drop.add(drop)
    kept = [c for c in X.columns if c not in to_drop]
    return X[kept].copy(), sorted(list(to_drop))

def time_safe_feature_mask(feature_names: List[str], descriptions: Dict[str, str], forbidden_windows: set[str]):
    mask_keep = []
    for f in feature_names:
        w = infer_window(f, descriptions.get(f, ""))
        mask_keep.append(w not in forbidden_windows)
    return pd.Series(mask_keep, index=feature_names)

def drop_time_like_features(cols: List[str], descriptions: Dict[str, str]) -> List[str]:
    keep, dropped = [], []
    for c in cols:
        text = (str(c) + " " + descriptions.get(c, "")).lower()
        is_time_like = bool(TIME_LIKE_DROP_PAT.search(text))
        is_lag_like = bool(TIME_LAG_KEEP_PAT.search(text))
        is_window = is_window_aggregate(c, descriptions.get(c, ""))
        if is_time_like and not (is_lag_like or is_window): dropped.append(c)
        else: keep.append(c)
    if dropped: print(f"🕒 Dropped {len(dropped)} literal time feature(s) (kept window/lag): first few → {dropped[:6]}")
    return keep, dropped

def jaccard_series(a: pd.Series, b: pd.Series) -> float:
    A = a.astype(bool).values; B = b.astype(bool).values
    inter = (A & B).sum(); union = (A | B).sum()
    return float(inter/union) if union > 0 else 0.0

def phi_correlation(a: pd.Series, b: pd.Series) -> float:
    A = a.astype(bool).values; B = b.astype(bool).values
    n11 = int(np.sum(A & B)); n10 = int(np.sum(A & ~B))
    n01 = int(np.sum(~A & B)); n00 = int(np.sum(~A & ~B))
    num = n11 * n00 - n10 * n01
    den = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    if den == 0: return 0.0
    return float(num / np.sqrt(den))

@dataclass
class FrameworkConfig:
    raw_data_path: str = "/home/vhsingh/Parshvi_project/data/raw/50k_users_merged_data_userfile_updated_shopping.csv"
    dictionary_path: str = "/home/vhsingh/Parshvi_project/data/raw/Internal_Algo360VariableDictionary_WithExplanation.xlsx"
    variable_catalog_path: str = "variable_catalog.csv"
    interim_dir: str = "/home/vhsingh/Parshvi_project/data/interim"
    max_rows_analysis: int = 50000
    max_features_analysis: int = 2000
    outcome_importance_threshold: float = 0.60
    min_label_prevalence: float = 0.005
    max_label_prevalence: float = 0.30
    cont_event_quantile_for_eligibility: float = 0.95
    include_lifetime_in_label: bool = False
    exclude_lifetime_everywhere: bool = True
    target_weighted_label_prevalence: Optional[float] = None
    union_severe_only: bool = False
    severe_weight_threshold_for_union: float = 0.50
    outcome_guard_terms: Tuple[str, ...] = (
        r"default", r"dpd", r"overdue", r"arrear", r"write[\s-]?off", r"charge[\s-]?off",
        r"npa", r"settle", r"miss", r"min[_\s-]?due", r"over[-\s]?limit", r"declin", r"reject",
        r"bounced", r"nsf", r"negative"
    )
    dominance_cutoff: float = 0.60
    dedup_jaccard_threshold: float = 0.90
    rescue_min_prevalence: float = 0.002
    rescue_max_prevalence: float = 0.40
    rescue_top_k: int = 12
    max_correlation_threshold: float = 0.95
    quality_threshold_for_label_eligibility: float = 0.50
    weighted_value_transform: str = "log1p"
    weighted_value_cap: float = 10.0
    outcome_as_feature_policy: str = "strict"
    drop_time_like: bool = True
    emit_continuous_risk: bool = True
    risk_window_weights: Dict[str, float] = field(default_factory=lambda: {
        "1m": 1.00, "3m": 0.95, "6m": 0.90, "12m": 0.85, "lt": 0.80, "unknown": 0.90
    })
    risk_normalization: str = "rank"
    risk_cap_percentiles: Tuple[float, float] = (0.50, 0.995)
    use_rarity_weight: bool = True
    rarity_floor: float = 1e-3
    rarity_cap: float = 8.0
    chosen_label_variant: str = "label_clustered"

class LabelIdentifier:
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
            r"(fraud|suspicious|aml|kyc.?fail|block|freeze)": 0.10
        }
        self.exclusion_patterns = [
            'campaign','channel','device','browser','ip','session','click','view','visit','engagement','preference',
            'transaction','transactions','purchase','purchases','booking','bookings',
            'recharge','recharges','order','orders','site','sites',
            'credit transactions','debit transactions','imps','neft','upi','wallet',
            'travel','hotel','food','grocery','market place','entertainment'
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
                if wt > max_score: max_score, best_pat = wt, pat

        return {
            'outcome_score': max_score,
            'outcome_type': 'keyword_weighted' if best_pat else 'none',
            'matching_keywords': matches,
            'reason': f"Matched {best_pat}" if best_pat else "No outcome pattern found",
        }

    def find_negative_pattern_variables(self, df: pd.DataFrame, descriptions: Dict[str, str]) -> pd.DataFrame:
        print("🔍 Scanning for variables with discovered negative patterns...")
        negative_patterns = [
            r'over[-\s]?limit', r'default|defaults', r'declin\w*', r'reject\w*',
            r'insufficient', r'penalt\w*', r'miss(?:ed|ing)?', r'over\s?due|overdue',
            r'min[_\s-]?due|mindue', r'negative\w*|noofnegativeevents'
        ]
        combined = re.compile('|'.join(f'({p})' for p in negative_patterns), re.I)

        rows = []
        for col in tqdm(df.columns, desc="Scanning for negative patterns"):
            desc = descriptions.get(col, ""); text = f"{col} {desc}".lower()
            if not combined.search(text): continue
            if self.config.exclude_lifetime_everywhere and infer_window(col, desc) == "lt": continue

            s = to_numeric_bool(df[col])
            missing_pct = s.isna().mean()
            vals = s.dropna()
            uniq = set(vals.unique())
            is_binary = len(uniq) <= 2 and uniq.issubset({0, 1})
            is_integer_like = pd.api.types.is_integer_dtype(s) or (len(vals) > 0 and np.allclose(vals % 1, 0, atol=1e-8))
            event_pos_rate = float((s.fillna(0) > 0).mean()) if len(s) > 0 else np.nan
            event_pos_count = int((s.fillna(0) > 0).sum()) if len(s) > 0 else 0

            priority = 0.0
            priority += 0.4 if missing_pct < 0.1 else (0.2 if missing_pct < 0.3 else 0.0)
            if is_binary or is_integer_like: priority += 0.3
            if not pd.isna(event_pos_rate) and 0.005 <= event_pos_rate <= 0.30: priority += 0.3

            rows.append({
                'variable': col,'description': desc[:200],
                'matched_patterns': '; '.join([p for p in negative_patterns if re.search(p, text, re.I)]),
                'pattern_count': sum(bool(re.search(p, text, re.I)) for p in negative_patterns),
                'is_binary': is_binary,'is_integer_like': is_integer_like,
                'event_positive_rate': event_pos_rate,'event_positive_count': event_pos_count,
                'missing_pct': missing_pct,'unique_count': s.nunique(dropna=True),
                'priority_score': priority,
                'recommended_for_label': (
                    (is_binary or is_integer_like) and not pd.isna(event_pos_rate) and
                    0.005 <= event_pos_rate <= 0.30 and missing_pct < 0.5 and priority >= 0.6
                )
            })
        cols = ['variable','description','matched_patterns','pattern_count','is_binary','is_integer_like',
                'event_positive_rate','event_positive_count','missing_pct','unique_count','priority_score','recommended_for_label']
        res = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
        if len(res) > 0:
            res = res.sort_values(['recommended_for_label','priority_score','pattern_count'], ascending=[False, False, False])
            print(f" Found {len(res)} variables matching negative patterns")
            print(f" {int(res['recommended_for_label'].sum())} variables recommended for label consideration")
        else:
            print(" Found 0 variables matching negative patterns")
        return res

    def analyze_candidates(self, df: pd.DataFrame, descriptions: Dict[str, str]) -> pd.DataFrame:
        print("🎯 Analyzing variables for label potential...")
        rows = []
        for col in tqdm(df.columns, desc="Scoring variables"):
            desc = descriptions.get(col, "")
            if self.config.exclude_lifetime_everywhere and infer_window(col, desc) == "lt": continue
            s = to_numeric_bool(df[col])
            missing_pct = s.isna().mean()
            unique_count = s.nunique(dropna=True)
            vals = s.dropna()
            uniq = set(vals.unique())
            is_binary = len(uniq) <= 2 and uniq.issubset({0, 1})
            is_integer_like = pd.api.types.is_integer_dtype(s) or (len(vals) > 0 and np.allclose(vals % 1, 0, atol=1e-8))
            is_continuous = not (is_binary or is_integer_like)

            event_threshold = None
            if is_binary or is_integer_like:
                event_pos_rate = float((s.fillna(0) > 0).mean()) if len(s) > 0 else np.nan
                event_threshold = 0.0
            else:
                try:
                    q = self.config.cont_event_quantile_for_eligibility
                    thr = float(np.nanquantile(vals.astype(float), q))
                    event_pos_rate = float((s.astype(float) >= thr).mean())
                    event_threshold = thr
                except Exception:
                    event_pos_rate = np.nan; event_threshold = None

            outcome_info = self.score_variable_as_outcome(col, desc)
            window_tag = infer_window(col, desc)

            quality_score = 1.0
            if missing_pct > 0.8: quality_score *= 0.3
            elif missing_pct > 0.5: quality_score *= 0.7
            if unique_count <= 1: quality_score = 0.0

            outcome_gate_ok = (outcome_info['outcome_score'] >= self.config.outcome_importance_threshold)
            if self.config.outcome_importance_threshold <= 0.0: outcome_gate_ok = True

            prevalence_ok = (not pd.isna(event_pos_rate)) and (self.config.min_label_prevalence <= event_pos_rate <= self.config.max_label_prevalence)

            eligible = (outcome_gate_ok and quality_score > self.config.quality_threshold_for_label_eligibility and prevalence_ok)

            rows.append({
                'variable': col,'description': desc[:200],'window': window_tag,'is_lifetime_window': (window_tag == "lt"),
                'is_binary': is_binary,'is_integer_like': is_integer_like,'is_continuous': is_continuous,
                'event_positive_rate': event_pos_rate,'missing_pct': missing_pct,'unique_count': unique_count,
                'outcome_score': outcome_info['outcome_score'],'outcome_type': outcome_info['outcome_type'],
                'matching_keywords': '; '.join(outcome_info['matching_keywords']),
                'quality_score': quality_score,'combined_score': outcome_info['outcome_score'] * quality_score,
                'event_threshold': event_threshold,'eligible_for_label': eligible
            })
        result_df = pd.DataFrame(rows).sort_values(['eligible_for_label','combined_score','outcome_score'], ascending=[False, False, False])
        return result_df

class CompositeLabelBuilder:
    def __init__(self, config: FrameworkConfig): self.config = config

    def create_label_variants(self, df: pd.DataFrame, candidates: pd.DataFrame,
                              negative_pattern_vars: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("🏗️ Building composite label variants...")
        if negative_pattern_vars is not None and len(negative_pattern_vars) > 0:
            priority_vars = negative_pattern_vars[negative_pattern_vars['recommended_for_label']].copy()
            if len(priority_vars) > 0:
                print(f" ⭐ Prioritizing {len(priority_vars)} negative pattern variables")
                priority_vars['source'] = 'negative_pattern'
                general_eligible = candidates[candidates['eligible_for_label']].copy()
                general_eligible = general_eligible[~general_eligible['variable'].isin(priority_vars['variable'])]
                general_eligible['source'] = 'general'
                eligible = pd.concat([priority_vars, general_eligible], ignore_index=True)
                eligible = eligible.drop(columns=[c for c in ["outcome_score","outcome_type"] if c in eligible], errors="ignore") \
                   .merge(
                       candidates[["variable","outcome_score","outcome_type","description","event_threshold","is_continuous"]],
                       on="variable", how="left"
                   )
            else:
                eligible = candidates[candidates['eligible_for_label']].copy()
                eligible['source'] = 'general'
        else:
            eligible = candidates[candidates['eligible_for_label']].copy()
            eligible['source'] = 'general'

        if self.config.exclude_lifetime_everywhere and len(eligible) > 0:
            wmask = eligible['variable'].apply(lambda c: infer_window(c, eligible.set_index('variable').loc[c, 'description'] if 'description' in eligible.columns else ""))
            before = len(eligible)
            eligible = eligible[wmask != "lt"]
            print(f" ⛔ Lifetime window exclusion in label sources: {before - len(eligible)} removed")

        if len(eligible) == 0:
            print("⚠️ No eligible label variables found!")
            return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        print(f" Using {len(eligible)} eligible variables")

        label_data = pd.DataFrame(index=df.index)
        sev_data = pd.DataFrame(index=df.index)
        meta = {}
        skipped_lifetime, skipped_zero, kept = [], [], []
        cont_used, bin_used = 0, 0
        thr_records = []

        for _, row in eligible.iterrows():
            col = row['variable']; desc = row.get('description', '')
            if col not in df.columns: continue
            win = infer_window(col, desc)
            if self.config.exclude_lifetime_everywhere and win == "lt": skipped_lifetime.append(col); continue
            s = to_numeric_bool(df[col])

            thr = row.get('event_threshold', None)
            is_cont = bool(row.get('is_continuous', False))
            if pd.notna(thr) and is_cont and float(thr) > 0:
                b = (s.astype(float).fillna(-np.inf) >= float(thr)).astype(int); cont_used += 1
            else:
                b = (s.fillna(0) > 0).astype(int); bin_used += 1
            if b.sum() == 0: skipped_zero.append(col); continue

            label_data[col] = b
            sev_data[col] = transform_value(s, self.config.weighted_value_transform, self.config.weighted_value_cap)
            meta[col] = {
                'weight': float(row.get('outcome_score', 0.0)),'type': row.get('outcome_type', ''),
                'description': desc,'event_threshold': thr if pd.notna(thr) else 0.0,'window': win
            }
            kept.append(col); thr_records.append((col, thr if pd.notna(thr) else 0.0, is_cont, win))

        print(f" Label sources after filters → kept={len(kept)} | lifetime_skipped={len(skipped_lifetime)} | zero_signal={len(skipped_zero)}")
        print(f" Composition → continuous-thresholded={cont_used} | binary/integer-like={bin_used}")
        if label_data.shape[1] == 0:
            print("❌ No viable label sources after filtering."); return pd.DataFrame(), {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        thr_df = pd.DataFrame(thr_records, columns=["variable","event_threshold","is_continuous","window"])
        thr_df.to_csv(Path(self.config.interim_dir) / "label_event_thresholds.csv", index=False)
        print(f"💾 Saved per-variable event thresholds → {self.config.interim_dir}/label_event_thresholds.csv")

        if label_data.shape[1] >= 2:
            kept_cols = []
            for c in sorted(label_data.columns, key=lambda c: meta[c]['weight'], reverse=True):
                if not kept_cols: kept_cols.append(c); continue
                is_dup = any(jaccard_series(label_data[c], label_data[k]) >= self.config.dedup_jaccard_threshold for k in kept_cols)
                if not is_dup: kept_cols.append(c)
            dropped = sorted(set(label_data.columns) - set(kept_cols))
            if dropped: print(f" 🔁 De-duplicated {len(dropped)} label sources (Jaccard≥{self.config.dedup_jaccard_threshold})")
            label_data = label_data[kept_cols]; sev_data = sev_data[kept_cols]; meta = {k: meta[k] for k in kept_cols}

        def compute_union(ld): return (ld.sum(axis=1) > 0).astype(int)
        def contribution(ld, u):
            pos = (u == 1)
            return sorted([(c, float(ld[c][pos].mean())) for c in ld.columns], key=lambda x: x[1], reverse=True)

        union_now = compute_union(label_data)
        best_prev = float(union_now.mean())
        pruned = []
        while True:
            contrib = contribution(label_data, union_now)
            dom = [c for c, s in contrib if s >= self.config.dominance_cutoff]
            if not dom: break
            trial = label_data.drop(columns=dom)
            if trial.shape[1] == 0: break
            new_prev = float(compute_union(trial).mean())
            if new_prev >= best_prev - 0.002:
                for c in dom:
                    pruned.append(c); meta.pop(c, None); sev_data.drop(columns=[c], inplace=True, errors='ignore')
                label_data = trial; union_now = compute_union(label_data); best_prev = new_prev
            else: break
        if pruned: print(f" 🪓 Dominance prune removed {len(pruned)} source(s) with share≥{self.config.dominance_cutoff})")

        if self.config.union_severe_only:
            thr = self.config.severe_weight_threshold_for_union
            basis_cols = [c for c in label_data.columns if meta[c]['weight'] >= thr] or list(label_data.columns)
        else:
            basis_cols = list(label_data.columns)

        label_union = (label_data[basis_cols].sum(axis=1) > 0).astype(int)
        weights = np.array([meta[c]['weight'] for c in basis_cols], dtype=float)
        wscore = (sev_data[basis_cols].values * weights).sum(axis=1)
        if not np.any(wscore > 0): wscore = (label_data[basis_cols].values * weights).sum(axis=1)

        positives = wscore[wscore > 0]
        if len(positives) > 0:
            target_prev = (self.config.target_weighted_label_prevalence
                           if self.config.target_weighted_label_prevalence is not None
                           else float(label_union.mean()))
            target_prev = max(0.005, min(0.5, target_prev))
            thr_q = float(np.quantile(positives, 1 - target_prev))
            label_weighted = (wscore >= thr_q).astype(int)
        else:
            thr_q = 0.0; label_weighted = np.zeros_like(wscore, dtype=int)

        sev_order = sorted(basis_cols, key=lambda c: meta[c]['weight'], reverse=True)
        def get_hier(row):
            for c in sev_order:
                if row[c] == 1: return 1
            return 0
        label_hierarchical = label_data[basis_cols].apply(get_hier, axis=1).astype(int)

        if len(basis_cols) >= 3:
            try:
                corr_matrix = label_data[basis_cols].corr().abs()
                kmeans = KMeans(n_clusters=min(3, max(1, len(basis_cols)//2)), random_state=42)
                clusters = kmeans.fit_predict(corr_matrix.values)
                reps = []
                for cl in range(kmeans.n_clusters):
                    vars_in = [basis_cols[i] for i, c in enumerate(clusters) if c == cl]
                    best = max(vars_in, key=lambda v: meta[v]['weight'])
                    reps.append(best)
                label_clustered = (label_data[reps].sum(axis=1) > 0).astype(int)
                print(f" 📦 label_clustered uses {len(reps)} representative sources from {len(basis_cols)} basis")
            except Exception:
                label_clustered = label_union.copy(); print(" ⚠️ KMeans clustering failed; label_clustered = label_union fallback")
        else:
            label_clustered = label_union.copy()

        labels_df = pd.DataFrame({
            'label_union': label_union,
            'label_weighted': label_weighted,
            'label_hierarchical': label_hierarchical,
            'label_clustered': label_clustered
        })
        labels_df.attrs['weighted_threshold'] = thr_q

        for col in label_data.columns:
            if meta[col]['weight'] >= 0.9:
                labels_df[f'label_{col}'] = label_data[col]

        if self.config.emit_continuous_risk:
            risk_cols = basis_cols; contrib_sum = np.zeros(len(df), dtype=float)
            for c in risk_cols:
                q_lo, q_hi = self.config.risk_cap_percentiles
                z = robust_normalize(sev_data[c], q_lo, q_hi)
                w_kw = float(meta[c]['weight'])
                w_win = float(self.config.risk_window_weights.get(meta[c]['window'], 
                                                                  self.config.risk_window_weights.get('unknown', 0.90)))
                pi = float(label_data[c].mean())
                if self.config.use_rarity_weight:
                    w_rare_raw = (1.0 / np.sqrt(max(pi, self.config.rarity_floor)))
                    w_rare = min(w_rare_raw, self.config.rarity_cap)
                else:
                    w_rare = 1.0
                contrib_sum += z * (w_kw * w_win * w_rare)

            risk_pct = pd.Series(contrib_sum).rank(pct=True).values if self.config.risk_normalization == "rank" \
                        else robust_normalize(contrib_sum, 0.01, 0.99)
            labels_df['risk_score_raw'] = contrib_sum
            labels_df['risk_percentile'] = risk_pct
            labels_df['risk_score_0_100'] = np.round(100 * risk_pct, 2)

        used_sources = list(label_data.columns)
        label_sources_df = pd.DataFrame({
            "label_source": used_sources,
            "weight": [meta[c]["weight"] for c in used_sources],
            "desc": [meta[c]["description"] for c in used_sources],
            "event_threshold": [meta[c]["event_threshold"] for c in used_sources],
            "window": [meta[c]["window"] for c in used_sources]
        }).sort_values("weight", ascending=False)

        return labels_df, meta, label_sources_df, label_data, sev_data

class SmartFeatureSelector:
    def __init__(self, config: FrameworkConfig): self.config = config

    def analyze_feature_quality(self, df: pd.DataFrame, descriptions: Dict[str, str]) -> pd.DataFrame:
        print("🔍 Analyzing feature quality and characteristics...")
        features = []
        for col in tqdm(df.columns, desc="Feature analysis"):
            series = df[col]; desc = descriptions.get(col, "")
            if self.config.exclude_lifetime_everywhere and infer_window(col, desc) == "lt": continue
            missing_pct = series.isna().mean(); unique_count = series.nunique(dropna=True); total_count = len(series)
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_categorical = (series.dtype == 'object') or (unique_count < 20)
            is_binary = unique_count == 2 and is_numeric
            if is_numeric:
                try:
                    variance = float(series.var()); dominant_pct = series.value_counts().iloc[0] / total_count if unique_count > 0 else 1.0
                    skewness = float(series.skew()); kurtosis = float(series.kurtosis())
                except Exception:
                    variance = skewness = kurtosis = np.nan; dominant_pct = 1.0
            else:
                variance = skewness = kurtosis = np.nan
                try:
                    dominant_pct = series.value_counts().iloc[0] / total_count
                except Exception:
                    dominant_pct = 1.0

            quality = self._calculate_quality_score(missing_pct, unique_count, total_count, dominant_pct, is_numeric)
            features.append({
                'variable': col,'description': desc[:150],'is_numeric': is_numeric,'is_categorical': is_categorical,
                'is_binary': is_binary,'missing_pct': missing_pct,'unique_count': unique_count,
                'dominant_value_pct': dominant_pct,'variance': variance,'skewness': skewness,'kurtosis': kurtosis,
                'quality_score': quality,'usable_for_modeling': quality > 0.3 and missing_pct < 0.9
            })
        return pd.DataFrame(features).sort_values('quality_score', ascending=False)

    def _calculate_quality_score(self, missing_pct: float, unique_count: int, total_count: int,
                                 dominant_pct: float, is_numeric: bool) -> float:
        score = 1.0
        if missing_pct > 0.8: score *= 0.1
        elif missing_pct > 0.5: score *= 0.5
        elif missing_pct > 0.2: score *= 0.8
        if unique_count <= 1: score *= 0.0
        elif dominant_pct > 0.99: score *= 0.2
        elif dominant_pct > 0.95: score *= 0.5
        if unique_count > 1:
            uniqueness_ratio = unique_count / total_count
            if is_numeric: score *= 1.0 if 0.01 <= uniqueness_ratio <= 1.0 else 0.8
            else: score *= 1.0 if 0.001 <= uniqueness_ratio <= 0.1 else 0.7
        return max(0.0, min(1.0, score))

    def select_features_for_labels(self, df: pd.DataFrame, labels: pd.DataFrame,
                                   features_df: pd.DataFrame, guard_set: Optional[set] = None) -> Dict[str, pd.DataFrame]:
        print("🎯 Selecting optimal features for each label...")
        usable_features = features_df[features_df['usable_for_modeling']].copy()
        feature_cols = [col for col in usable_features['variable'] if col in df.columns]
        if guard_set:
            before = len(feature_cols)
            feature_cols = [c for c in feature_cols if c not in guard_set]
            print(f" Guarded features removed: {before - len(feature_cols)}")
        if len(feature_cols) > self.config.max_features_analysis:
            top_features = usable_features.head(self.config.max_features_analysis)
            feature_cols = [c for c in top_features['variable'] if c in df.columns and (not guard_set or c not in guard_set)]
        print(f" Analyzing {len(feature_cols)} features against {len(labels.columns)} labels")

        results = {}
        for label_col in tqdm(labels.columns, desc="Feature selection per label"):
            y = labels[label_col]; uniq = pd.Series(y).dropna().unique()
            if set(np.unique(uniq)).issubset({0, 1}) is False: continue
            if y.sum() == 0: continue
            X = df[feature_cols].copy(); numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_processed = pd.DataFrame(index=X.index)
            if len(numeric_cols) > 0:
                imputer = SimpleImputer(strategy='median'); X_numeric = imputer.fit_transform(X[numeric_cols])
                X_processed[numeric_cols] = X_numeric
            cat_cols = X.select_dtypes(include=['object']).columns
            for col in cat_cols:
                top_cats = X[col].value_counts().head(10).index
                for cat in top_cats:
                    X_processed[f'{col}_{cat}'] = (X[col] == cat).astype(int)
            if len(X_processed.columns) == 0: continue
            try:
                var_series = X_processed.var(numeric_only=True)
                X_processed_pruned, _ = prune_correlated(X_processed, corr_thr=self.config.max_correlation_threshold, keep_priority=var_series)
                X_use = X_processed_pruned
            except Exception:
                X_use = X_processed
            scores = self._multi_method_selection(X_use, y)
            results[label_col] = scores
        return results

    def _multi_method_selection(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        scores = pd.DataFrame({'feature': X.columns})
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42)
            scores['mutual_info'] = mi_scores
        except Exception:
            scores['mutual_info'] = 0.0
        try:
            f_scores, _ = f_classif(X, y)
            scores['f_statistic'] = pd.Series(f_scores).fillna(0).values
        except Exception:
            scores['f_statistic'] = 0.0
        try:
            corr_scores = [abs(spearmanr(X[col], y)[0]) for col in X.columns]
            scores['correlation'] = [x if not np.isnan(x) else 0 for x in corr_scores]
        except Exception:
            scores['correlation'] = 0.0
        try:
            if y.sum() > 10 and len(X) > 50:
                scaler = RobustScaler(); X_scaled = scaler.fit_transform(X)
                lr = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, random_state=42)
                lr.fit(X_scaled, y); scores['l1_coef'] = np.abs(lr.coef_[0])
            else:
                scores['l1_coef'] = 0.0
        except Exception:
            scores['l1_coef'] = 0.0

        for col in ['mutual_info', 'f_statistic', 'correlation', 'l1_coef']:
            mx = np.max(scores[col]) if len(scores[col]) else 0
            if mx > 0: scores[col] = scores[col] / mx
        scores['combined_score'] = (scores['mutual_info'] * 0.3 + scores['f_statistic'] * 0.25 +
                                    scores['correlation'] * 0.25 + scores['l1_coef'] * 0.2)
        return scores.sort_values('combined_score', ascending=False)

def build_leakage_guard(header_cols: List[str], descriptions: Dict[str, str],
                        label_sources: List[str], outcome_guard_terms: Tuple[str, ...],
                        policy: str = "strict") -> set:
    guard = set(label_sources)
    label_fams = {variable_family(c) for c in label_sources}
    for c in header_cols:
        if variable_family(c) in label_fams: guard.add(c)
    if policy == "strict":
        pat = re.compile("|".join(outcome_guard_terms), re.IGNORECASE)
        for c in header_cols:
            txt = (c + " " + descriptions.get(c, "")).lower()
            if pat.search(txt) or is_outcome_like(c, descriptions.get(c, "")): guard.add(c)
    return guard

def read_external_guard(interim_dir: str) -> set:
    candidates = ["do not use features.txt", "do_not_use_features.txt", str(Path(interim_dir) / "do_not_use_features.txt")]
    out = set()
    for fname in candidates:
        p = Path(fname)
        if p.exists():
            for ln in p.read_text().splitlines():
                ln = ln.strip()
                if ln: out.add(ln)
    return out

def write_external_guard(guard_set: set, interim_dir: str):
    root_file = Path("do_not_use_features.txt")
    interim_file = Path(interim_dir) / "do_not_use_features.txt"
    interim_file.parent.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(sorted(guard_set))
    interim_file.write_text(txt); root_file.write_text(txt)

def ensure_plot_dir(base_dir: str) -> Path:
    p = Path(base_dir) / "plots"; p.mkdir(parents=True, exist_ok=True); return p

def save_heatmap(matrix: np.ndarray, labels: List[str], filename: Path, title: str, vmin: float = 0.0, vmax: float = 1.0):
    plt.figure(figsize=(max(6, len(labels)*0.35), max(5, len(labels)*0.35)))
    plt.imshow(matrix, vmin=vmin, vmax=vmax, aspect='auto'); plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90); plt.yticks(range(len(labels)), labels)
    plt.title(title); plt.tight_layout(); plt.savefig(filename); plt.close()

def save_bar(names: List[str], values: List[float], filename: Path, title: str):
    plt.figure(figsize=(max(6, len(names)*0.4), 6))
    idx = np.arange(len(names)); plt.bar(idx, values)
    plt.xticks(idx, names, rotation=90); plt.title(title); plt.tight_layout(); plt.savefig(filename); plt.close()

def save_hist(values: np.ndarray, bins: int, filename: Path, title: str):
    plt.figure(figsize=(8, 5)); plt.hist(values, bins=bins); plt.title(title); plt.tight_layout(); plt.savefig(filename); plt.close()

def generate_diagnostics_and_plots(df: pd.DataFrame, labels_df: pd.DataFrame,
                                   label_sources_df: pd.DataFrame, label_events_df: pd.DataFrame,
                                   sev_data_df: pd.DataFrame, descriptions: Dict[str, str], config: FrameworkConfig):
    print("\n" + "="*50); print("STEP X: STATISTICAL DIAGNOSTICS & PLOTS"); print("="*50)
    out_dir = ensure_plot_dir(config.interim_dir)
    chosen = config.chosen_label_variant if config.chosen_label_variant in labels_df.columns else list(labels_df.columns)[0]
    y = labels_df[chosen]
    print("\n📈 Label Variant Statistics (prevalence):")
    for col in ['label_union','label_weighted','label_hierarchical','label_clustered']:
        if col in labels_df.columns:
            pos_rate = float(labels_df[col].mean()); pos_count = int(labels_df[col].sum())
            print(f" {col:20s}: prevalence={pos_rate:6.4f} ({pos_count:,} positives)")

    events = [c for c in label_events_df.columns]
    if len(events) >= 2:
        J = np.zeros((len(events), len(events)), dtype=float)
        for i in range(len(events)):
            ai = label_events_df[events[i]]
            for j in range(i, len(events)):
                aj = label_events_df[events[j]]
                J[i, j] = J[j, i] = jaccard_series(ai, aj)
        save_heatmap(J, events, out_dir / "jaccard_heatmap.png", "Jaccard Similarity between Label Sources", 0.0, 1.0)
        pd.DataFrame(J, index=events, columns=events).to_csv(Path(config.interim_dir) / "jaccard_matrix.csv", index=True)
        print(f"💾 Saved Jaccard heatmap and matrix")

    if len(events) >= 2:
        P = np.zeros((len(events), len(events)), dtype=float)
        for i in range(len(events)):
            ai = label_events_df[events[i]]
            for j in range(i, len(events)):
                aj = label_events_df[events[j]]
                P[i, j] = P[j, i] = phi_correlation(ai, aj)
        save_heatmap(P, events, out_dir / "phi_heatmap.png", "Phi Correlation between Label Sources", -1.0, 1.0)
        pd.DataFrame(P, index=events, columns=events).to_csv(Path(config.interim_dir) / "phi_matrix.csv", index=True)
        print(f"💾 Saved Phi correlation heatmap and matrix")

    mi_rows = []
    for ev in events:
        mi = mutual_info_score(label_events_df[ev], y)
        mi_rows.append((ev, mi))
    pd.DataFrame(mi_rows, columns=["event","mutual_info_with_chosen"])\
        .sort_values("mutual_info_with_chosen", ascending=False)\
        .to_csv(Path(config.interim_dir) / "event_mutual_info_vs_chosen.csv", index=False)

    contrib_rows = []; pos_mask = (y == 1)
    for ev in events:
        share = float(label_events_df.loc[pos_mask, ev].mean()) if pos_mask.any() else 0.0
        contrib_rows.append((ev, share))
    contrib_df = pd.DataFrame(contrib_rows, columns=["event","share_among_chosen_positives"]).sort_values("share_among_chosen_positives", ascending=False)
    contrib_df.to_csv(Path(config.interim_dir) / "event_contribution_summary_vs_chosen.csv", index=False)
    topK = contrib_df.head(min(20, len(contrib_df)))
    save_bar(topK['event'].tolist(), topK['share_among_chosen_positives'].tolist(), out_dir / "top_contributors_bar.png",
             f"Top Contributors among {config.chosen_label_variant} positives")
    print(f"💾 Saved top contributors bar chart")

    if 'risk_score_0_100' in labels_df.columns:
        save_hist(labels_df['risk_score_0_100'].values, bins=30, filename=out_dir / "risk_score_hist.png",
                  title="Risk Score (0–100) Distribution")
        print(f"💾 Saved risk score distribution")

    triggers = label_events_df.sum(axis=1).values
    save_hist(triggers, bins=30, filename=out_dir / "event_trigger_count_hist.png",
              title="Number of Label Sources Triggered per User")
    print(f"💾 Saved event trigger count histogram")

    if 'risk_score_0_100' in labels_df.columns:
        rho_rows = []; rscore = labels_df['risk_score_0_100']
        for ev in events:
            try:
                rho = spearmanr(sev_data_df[ev], rscore)[0]; 
                if np.isnan(rho): rho = 0.0
            except Exception:
                rho = 0.0
            rho_rows.append((ev, float(rho)))
        rho_df = pd.DataFrame(rho_rows, columns=["event","spearman_risk_vs_severity"]).sort_values("spearman_risk_vs_severity", ascending=False)
        rho_df.to_csv(Path(config.interim_dir) / "risk_vs_source_severity_spearman.csv", index=False)
        topK_rho = rho_df.head(min(20, len(rho_df)))
        save_bar(topK_rho['event'].tolist(), topK_rho['spearman_risk_vs_severity'].tolist(),
                 out_dir / "risk_vs_severity_corr_bar.png", "Spearman Correlation: Risk vs Source Severity")
        print(f"💾 Saved risk vs severity correlation bar chart")
    print("✅ Diagnostics complete. See data/interim/ and data/interim/plots/ for outputs.\n")

class SmartVariableFramework:
    def __init__(self, config: FrameworkConfig = None):
        self.config = config or FrameworkConfig()
        self.label_identifier = LabelIdentifier(self.config)
        self.label_builder = CompositeLabelBuilder(self.config)
        self.feature_selector = SmartFeatureSelector(self.config)

    def load_data(self) -> Tuple[pd.DataFrame, Dict[str, str]]:
        print("📁 Loading data sources...")
        df = pd.read_csv(self.config.raw_data_path, low_memory=False)
        if len(df) > self.config.max_rows_analysis:
            df = df.sample(n=self.config.max_rows_analysis, random_state=42)
            print(f" Sampled to {len(df):,} rows for analysis")
        print(f" Loaded: {len(df):,} rows × {len(df.columns)} columns")

        descriptions = {}
        if Path(self.config.dictionary_path).exists():
            print(f" Loading variable dictionary from {self.config.dictionary_path} ...")
            try:
                excel_file = pd.ExcelFile(self.config.dictionary_path)
                sheet_sizes = {sheet: pd.read_excel(self.config.dictionary_path, sheet_name=sheet).shape[0]
                               for sheet in excel_file.sheet_names}
                best_sheet = max(sheet_sizes, key=sheet_sizes.get)
                dict_df = pd.read_excel(self.config.dictionary_path, sheet_name=best_sheet)
                var_col = None; exp_col = None
                for col in dict_df.columns:
                    cl = str(col).lower()
                    if any(t in cl for t in ['variable', 'field', 'column', 'name']): var_col = col
                    elif any(t in cl for t in ['explanation', 'description', 'meaning', 'detail']): exp_col = col
                if var_col and exp_col:
                    for _, row in dict_df.iterrows():
                        if pd.notna(row[var_col]) and pd.notna(row[exp_col]):
                            descriptions[str(row[var_col])] = str(row[exp_col])
                    print(f" Loaded {len(descriptions)} variable descriptions")
            except Exception as e:
                print(f" ⚠️ Could not load dictionary: {e}")

        if Path(self.config.variable_catalog_path).exists():
            try:
                catalog_df = pd.read_csv(self.config.variable_catalog_path)
                if 'Variable' in catalog_df.columns and 'Description' in catalog_df.columns:
                    for _, row in catalog_df.iterrows():
                        if pd.notna(row['Variable']) and pd.notna(row['Description']):
                            descriptions.setdefault(str(row['Variable']), str(row['Description']))
                print(f" Total descriptions available: {len(descriptions)}")
            except Exception as e:
                print(f" ⚠️ Could not load catalog: {e}")

        return df, descriptions

    def run_analysis(self) -> Dict[str, Any]:
        print("🚀 Starting Smart Variable Framework (Extended)")
        print("=" * 60)
        Path(self.config.interim_dir).mkdir(parents=True, exist_ok=True)
        df, descriptions = self.load_data()

        print("\n" + "="*50); print("STEP 1A: NEGATIVE PATTERN VARIABLE DISCOVERY"); print("="*50)
        negative_pattern_vars = self.label_identifier.find_negative_pattern_variables(df, descriptions)
        negative_pattern_vars.to_csv(f"{self.config.interim_dir}/negative_pattern_variables.csv", index=False)
        print(f"💾 Saved → {self.config.interim_dir}/negative_pattern_variables.csv")

        print("\n" + "="*50); print("STEP 1B: COMPREHENSIVE LABEL IDENTIFICATION"); print("="*50)
        label_candidates = self.label_identifier.analyze_candidates(df, descriptions)
        label_candidates.to_csv(f"{self.config.interim_dir}/smart_label_candidates.csv", index=False)
        print(f"💾 Saved → {self.config.interim_dir}/smart_label_candidates.csv")
        eligible_labels = label_candidates[label_candidates['eligible_for_label']]
        print(f"📊 Eligible label variables: {len(eligible_labels)}")

        print("\n" + "="*50); print("STEP 2: COMPOSITE LABEL CREATION + RISK SCORE (0–100)"); print("="*50)
        labels_df, label_meta, label_sources_df, label_events_df, sev_data_df = self.label_builder.create_label_variants(
            df, label_candidates, negative_pattern_vars
        )
        if labels_df.empty:
            print("⚠️ No composite labels could be created")
            return {'status': 'failed', 'reason': 'No eligible label variables found'}

        header_cols = df.columns.tolist()
        label_sources = label_sources_df['label_source'].tolist()
        guard_set = build_leakage_guard(header_cols, descriptions, label_sources, self.config.outcome_guard_terms,
                                        policy=self.config.outcome_as_feature_policy)

        external_guard = read_external_guard(self.config.interim_dir)
        if external_guard:
            guard_set |= external_guard
            print(f"🛡️ Merged external guard entries: {len(external_guard)} (total guard={len(guard_set)})")

        label_sources_df.to_csv(f"{self.config.interim_dir}/label_sources_used.csv", index=False)
        with open(f"{self.config.interim_dir}/label_basis.json","w") as f:
            json.dump({"union_severe_only": self.config.union_severe_only,
                       "num_sources_in_basis": int(len(label_sources))}, f, indent=2)
        print(f"💾 Saved label sources → {self.config.interim_dir}/label_sources_used.csv")

        labels_df.to_csv(f"{self.config.interim_dir}/composite_labels.csv", index=False)
        print(f"💾 Saved label variants → {self.config.interim_dir}/composite_labels.csv")

        print("\n📈 Label Variant Statistics:")
        for col in labels_df.columns:
            if labels_df[col].dropna().nunique() <= 2:
                pos_rate = float(labels_df[col].mean()); pos_count = int(labels_df[col].sum())
                print(f" {col:25s}: prevalence={pos_rate:6.4f} ({pos_count:,} positives)")
        if 'risk_score_0_100' in labels_df.columns:
            print(f" risk_score_0_100: min={labels_df['risk_score_0_100'].min():.2f} | "
                  f"p50={labels_df['risk_score_0_100'].median():.2f} | "
                  f"p95={labels_df['risk_score_0_100'].quantile(0.95):.2f} | "
                  f"max={labels_df['risk_score_0_100'].max():.2f}")

        generate_diagnostics_and_plots(df, labels_df, label_sources_df, label_events_df, sev_data_df, descriptions, self.config)

        print("="*50); print("STEP 3: FEATURE QUALITY & FILTERING"); print("="*50)
        feature_quality = self.feature_selector.analyze_feature_quality(df, descriptions)
        feature_quality.to_csv(f"{self.config.interim_dir}/variable_quality_report.csv", index=False)
        print(f"💾 Saved feature analysis → {self.config.interim_dir}/variable_quality_report.csv")

        feature_cols_all = [c for c in df.columns if c not in labels_df.columns and c not in guard_set]
        if self.config.drop_time_like:
            kept, dropped = drop_time_like_features(feature_cols_all, descriptions)
            feature_cols_all = kept; guard_set |= set(dropped)

        forbidden_windows = {"lt"} if self.config.exclude_lifetime_everywhere else set()
        mask = time_safe_feature_mask(feature_cols_all, descriptions, forbidden_windows)
        feature_cols_time_safe = [feature_cols_all[i] for i, keep in enumerate(mask) if keep]

        guard_set |= id_like_columns(df.columns)
        guard_set |= near_unique_columns(df)

        try:
            y_for_guard = None
            preferred_order = [self.config.chosen_label_variant, "label_union", "label_hierarchical", "label_weighted", "label_clustered"]
            for preferred in preferred_order:
                if preferred in labels_df.columns: y_for_guard = labels_df[preferred]; break
            # Optional "too-predictive" guard disabled in framework; kept in snapshot builder instead
        except Exception:
            pass

        feature_cols_time_safe = [c for c in feature_cols_time_safe if c not in guard_set]
        write_external_guard(guard_set, self.config.interim_dir)
        print(f"🛡️ Leakage guard size (post time/ID/unique/predictive): {len(guard_set)}")
        print(f"✅ Time-safe feature candidates: {len(feature_cols_time_safe)}")

        fq = feature_quality.copy()
        usable_mask = fq['usable_for_modeling'] & fq['variable'].isin(feature_cols_time_safe)
        usable = fq[usable_mask & (fq['missing_pct'] <= (1.0 - 0.85))].copy()

        num_keep = [c for c in usable['variable'] if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        if len(num_keep):
            imp_vals = df[num_keep].copy().fillna(df[num_keep].median(numeric_only=True))
            priority = usable.set_index('variable')['quality_score']
            X_pruned, dropped_corr = prune_correlated(imp_vals, corr_thr=self.config.max_correlation_threshold, keep_priority=priority)
            kept_features = X_pruned.columns.tolist(); dropped_features = sorted(set(num_keep) - set(kept_features))
        else:
            kept_features, dropped_features = [], []; X_pruned = pd.DataFrame(index=df.index)

        Path(f"{self.config.interim_dir}/feature_keep_list.txt").write_text("\n".join(kept_features))
        Path(f"{self.config.interim_dir}/feature_drop_corr.txt").write_text("\n".join(dropped_features))
        print(f"🔗 Correlation pruning: kept={len(kept_features)}, dropped={len(dropped_features)} (> r={self.config.max_correlation_threshold})")

        print("="*50); print("STEP 4: FEATURE SELECTION (scoring)"); print("="*50)
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
            score_cols = [c for c in feature_matrix.columns if c.endswith('_score')]
            feature_matrix[score_cols] = feature_matrix[score_cols].fillna(0)
            feature_matrix['avg_importance'] = feature_matrix[score_cols].mean(axis=1)
            feature_matrix = feature_matrix.sort_values('avg_importance', ascending=False)
            feature_matrix.to_csv(f"{self.config.interim_dir}/feature_importance_matrix.csv", index=False)
            print(f"💾 Saved feature importance matrix → {self.config.interim_dir}/feature_importance_matrix.csv")

        if 'risk_score_0_100' in labels_df.columns:
            risk_out = pd.DataFrame({"risk_score_0_100": labels_df['risk_score_0_100'].values})
            risk_out.to_csv(f"{self.config.interim_dir}/final_risk_scores.csv", index=False)
            print(f"💾 Saved final risk-only scores (0–100) → {self.config.interim_dir}/final_risk_scores.csv")
        else:
            print("⚠️ Risk score column not found; check emit_continuous_risk option")

        print("\n✅ Analysis complete."); print("="*60)
        return {'status': 'success','label_candidates': len(label_candidates),
                'eligible_labels': int(eligible_labels['eligible_for_label'].sum()) if len(eligible_labels) else 0,
                'label_variants': len(labels_df.columns) if len(labels_df) > 0 else 0,
                'usable_features': int((feature_quality['usable_for_modeling']).sum()),
                'chosen_label': self.config.chosen_label_variant}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data", default=None)
    parser.add_argument("--dict-xlsx", default=None)
    parser.add_argument("--catalog-csv", default=None)
    parser.add_argument("--outcome-score-thr", type=float)
    parser.add_argument("--outcome-policy", choices=["strict","relaxed"])
    parser.add_argument("--weighted-value-transform", choices=["log1p","asinh","raw","cap","signed_asinh"])
    parser.add_argument("--weighted-value-cap", type=float)
    parser.add_argument("--no-drop-time-like", action="store_true")
    parser.add_argument("--chosen-label", choices=["label_union","label_weighted","label_hierarchical","label_clustered"], default="label_clustered")
    parser.add_argument("--cont-eligibility-quantile", type=float, default=None)
    args, _ = parser.parse_known_args()

    config = FrameworkConfig()
    if args.raw_data: config.raw_data_path = args.raw_data
    if args.dict_xlsx: config.dictionary_path = args.dict_xlsx
    if args.catalog_csv: config.variable_catalog_path = args.catalog_csv
    if args.outcome_score_thr is not None: config.outcome_importance_threshold = args.outcome_score_thr
    if args.outcome_policy is not None: config.outcome_as_feature_policy = args.outcome_policy
    if args.weighted_value_transform is not None: config.weighted_value_transform = args.weighted_value_transform
    if args.weighted_value_cap is not None: config.weighted_value_cap = args.weighted_value_cap
    if args.no_drop_time_like: config.drop_time_like = False
    if args.chosen_label: config.chosen_label_variant = args.chosen_label
    if args.cont_eligibility_quantile is not None: config.cont_event_quantile_for_eligibility = args.cont_eligibility_quantile

    config.include_lifetime_in_label = False
    config.exclude_lifetime_everywhere = True

    framework = SmartVariableFramework(config)
    try:
        results = framework.run_analysis()
        if results['status'] == 'success':
            print("\n🎉 Success")
            print(f" - Label candidates: {results['label_candidates']}")
            print(f" - Eligible labels: {results['eligible_labels']}")
            print(f" - Label variants: {results['label_variants']}")
            print(f" - Usable features: {results['usable_features']}")
            print(f" - Chosen label: {results['chosen_label']}")
        else:
            print(f"\n❌ Failed: {results.get('reason','Unknown error')}")
    except Exception as e:
        print(f"\n💥 Critical error: {e}")
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()
