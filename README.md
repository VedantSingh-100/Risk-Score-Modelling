# Risk Model Pipeline (Snapshot Mode, Leakage-Safe)

This repository builds a production‑grade **credit risk score** using a fully reproducible pipeline:

1) **Label discovery & composition** from raw vendor variables  
2) **Leakage guards** and **time-safety** for features  
3) **Feature engineering** (fill-rate filtering, transforms, encoding, scaling, correlation pruning)  
4) **Model training** (GBDT sweep, MLP stack, and a deep attention model)  
5) **Calibration & reporting**

The pipeline is designed for **snapshot training with lifetime labels** while **removing outcome-like and lifetime features** from the model inputs to prevent leakage.

---

## Data Inputs

- `data/raw/50k_users_merged_data_userfile_updated_shopping.csv`  
  > Wide user-level table (~50k rows x ~1.6k columns). Each column is a raw variable.

- `data/raw/Internal_Algo360VariableDictionary_WithExplanation.xlsx`  
  > Dictionary that maps variable codes to human-readable descriptions. We infer **windows** and **semantics** from these descriptions.

- *(Optional)* `data/raw/variable_catalog.csv`  
  > Additional variable metadata (if available).

---

## Stage 1 — Label Discovery & Composition

**File:** `src/smart_variable_framework.py`

**What it does**
- Crawls all columns and descriptions to detect **negative-outcome patterns** via regex:
  - `default`, `dpd`, `overdue`, `arrear`, `write-off`, `charge-off`, `miss(ed|ing)`, `min_due|mindue`,
  - `overlimit`, `decline`, `reject`, `insufficient`, `penalty`, `bounced|nsf`, `negative events` …
- Scores candidates with keyword weights and **data quality checks** (missing %, variance, binary shape, prevalence).
- Produces **composite labels** from multiple sources:
  - **Union** (with "severe" weighting threshold)
  - **Severity‑weighted** (quantile‑thresholded to target prevalence)
  - **Hierarchical** (worst-first)
  - **Clustered** (KMeans on source correlation; representative pick)
- **Duplicate control:** Drop near-duplicate sources by **Jaccard** similarity.
- **Dominance control:** If one source explains ≥ dominance_cutoff of label positives, it can be down‑weighted or pruned (non‑fatal).
- **Rescue policy:** If no viable labels surface, progressively relaxes rules (but keeps exposures out).

**Outputs (stored under `data/interim/`):**
- `negative_pattern_variables.csv` – All variables that matched negative patterns with stats
- `smart_label_candidates.csv` – All variables scored for outcome‑likeness
- `composite_labels.csv` – Columns: `label_union`, `label_weighted`, `label_hierarchical`, `label_clustered`, …
- `event_contribution_summary.csv` – Share of each source among positives of the union label
- `jaccard_matrix.csv` – Overlap matrix among label sources
- `do_not_use_features.txt` – Baseline **guard** list (label sources, outcome-like columns, families)
- `smart_framework_report.md` – Human-readable report
- `recommended_pipeline.json` – Machine recommendations (best label, top features if available)

**Why lifetime labels?**  
Your data shows best stability and prevalence with lifetime flags. We keep **lifetime labels**, but **do NOT allow lifetime or outcome-like features** into the model to prevent leakage.

---

## Stage 2 — Leakage Guard & Time Safety

We create a **guard set** of columns that must never enter the feature matrix:

- **Label sources** used to build the target  
- Any column whose **name/description matches outcome patterns** (same regex as above)  
- **Whole families** of the label sources (e.g., `var201xxx`)  
- **ID-like** columns (e.g., `*_id`, `email`, `mobile`, `pan`, etc.)  
- **Near-unique** columns (≥ 98% unique values)  
- **Too-predictive smell test** (any single raw feature with suspiciously high AUC on the label)  
- **Lifetime-window features** when labels are lifetime (to remove lookahead leakage)

This guard is written to `data/interim/do_not_use_features.txt` and is respected downstream.

---

## Stage 3 — Feature Engineering (Snapshot)

**File:** `src/build_features_snapshot.py`

**What it does**
1. **Pick a label** from `composite_labels.csv` (default: `label_union`) and write `data/processed/y_label.csv`.
2. Build the **candidate feature set** = all columns minus the **guard** list.
3. **Fill‑rate filter:** keep columns with fill‑rate ≥ **0.85** (i.e., ≤ 15% missing).
4. **Categorical encoding:**  
   - **OHE** if ≤ 8 distinct values  
   - **Frequency encoding** otherwise
5. **Numeric transforms:**  
   - `log1p` for **counts/occurrences**  
   - `asinh` (safe for negatives) for **ratios/amounts/other**  
   - **median imputation** for NaNs
6. **Robust scaling** (median/IQR) for numerics
7. **Graph-based correlation pruning** at **|r| ≥ 0.97** with **quality‑score/variance priority**
8. Write final matrix to `data/processed/X_features.parquet` + manifests:
   - `feature_keep_list.txt` – final kept features  
   - `feature_drop_corr.txt` – correlated drops  
   - `fe_manifest.json` – full FE metadata

**Why these choices**
- High **fill‑rate** ensures stable features.  
- `log1p` and `asinh` tame heavy tails and preserve directionality.  
- **Robust scaling** protects against outliers.  
- **Graph‑pruning** reduces redundancy and overfitting.

---

## Stage 4 — Modeling

### (A) Gradient Boosted Trees (XGBoost / LightGBM)

**File:** `src/train_gbdt.py`

- **Stratified K‑fold OOF** (default 5‑fold)
- **Random hyperparameter sweep**
- **Early stopping** on `auc` or `aucpr` (auto picks `aucpr` if prevalence ≤ 20%)
- Respects `feature_keep_list.txt` (or uses `feature_drop_corr.txt` / `--prune-corr`)
- Optionally **monotone constraints** (`data/processed/monotone_config.json`)

**Artifacts**
- `best_params.json`, `trials_summary.csv`  
- `oof_running_best.csv` (OOF predictions & y)  
- `deciles_running_best.csv`, `feature_importance_running_best.csv`  
- **Platt calibration** summary + `deciles_running_best_calibrated.csv`

---

### (B) Linear Stacking (XGB + MLP)

**File:** `src/train_mlp_stack.py`

- Gets OOF predictions from an **XGB** (using best params) and a **baseline MLP**
- **Stacker** = Logistic regression (Platt), with optional **nested OOF** (`--stack-nested`) for a fully unbiased stacked OOF.
- Artifacts: `summary.csv`, `oof_predictions.csv`, `deciles_xxx.csv`, `stacker_logit.json`

---

### (C) Deep Attention Model (FT‑Transformer for Tabular)

**File:** `src/train_fttransformer.py` *(see below for code)*

- **Feature tokenizer** turns each numeric feature into a token (learned per‑feature projection)
- Several **Transformer blocks** (Multi‑head self‑attention + PreNorm + residual MLP)
- **CLS token** + pooling for classification
- **Cosine LR** with warmup, **AdamW**, **label imbalance** via `pos_weight`
- **Mixed precision** (if CUDA), **early stopping** on AUC
- Produces OOF, deciles, and a saved model state

---

## Stage 5 — Calibration & Reporting

- GBDT: Platt calibration (`oof_cal`)  
- Deciles and KS/Gini in each model output  
- W&B logging optional

---

## Directory Layout

```
data/
  raw/
    50k_users_merged_data_userfile_updated_shopping.csv
    Internal_Algo360VariableDictionary_WithExplanation.xlsx
  interim/
    composite_labels.csv
    do_not_use_features.txt
    negative_pattern_variables.csv
    smart_label_candidates.csv
    smart_framework_report.md
    ...
  processed/
    X_features.parquet
    y_label.csv
    feature_keep_list.txt
    feature_drop_corr.txt
    fe_manifest.json

model_outputs/
  gbdt_sweep_YYYYMMDD_HHMMSS/
  stack_YYYYMMDD_HHMMSS/
  fttr_YYYYMMDD_HHMMSS/
  ...
```

---

## How to Run (Snapshot Mode)

1. **Label discovery & guards**
   ```bash
   python -m src.smart_variable_framework
   # artifacts written to data/interim/
   ```

2. **Feature engineering (leakage-safe)**
   ```bash
   python -m src.build_features_snapshot \
     --raw-csv   data/raw/50k_users_merged_data_userfile_updated_shopping.csv \
     --dict-xlsx data/raw/Internal_Algo360VariableDictionary_WithExplanation.xlsx \
     --interim-dir data/interim \
     --out-dir     data/processed \
     --label-column label_union \
     --fill-rate-min 0.85 --corr-thr 0.97
   ```

3. **GBDT sweep**
   ```bash
   python -m src.train_gbdt \
     --data-root data/processed \
     --out-dir   model_outputs/gbdt_sweep_$(date +%Y%m%d_%H%M%S) \
     --algo xgb --trials 120 --n-splits 5 --eval-metric auto --use-monotone
   ```

4. **Stacking**
   ```bash
   python -m src.train_mlp_stack \
     --data-root data/processed \
     --out-dir   model_outputs/stack_$(date +%Y%m%d_%H%M%S) \
     --best-xgb-params model_outputs/gbdt_sweep_.../best_params.json \
     --stack-nested
   ```

5. **Deep attention model (FT‑Transformer)**
   ```bash
   python -m src.train_fttransformer \
     --data-root data/processed \
     --out-dir   model_outputs/fttr_$(date +%Y%m%d_%H%M%S) \
     --layers 4 --d-model 96 --heads 8 --dropout 0.2
   ```

---

## Sanity Checks

- **Leakage guard applied?** `data/interim/do_not_use_features.txt` exists and training logs show it was loaded/applied.
- **Fill‑rate policy enforced?** FE logs show ≥0.85 fill‑rate filtering counts.
- **Correlation pruning applied?** FE logs report how many features were dropped (r ≥ 0.97).
- **Label prevalence** printed at train start.
- **OOF metrics** reported; use deciles & KS for business sanity.

---

## Extensibility

- **Time-based splits** (e.g., Jan–Sep train, Oct–Dec validate): add a `ref_date` column and filter/roll your windows; the framework already isolates lifetime features out of X to remain leakage‑safe even on snapshot data.
- **Fairness & bias checks**: slice metrics by demographics (if legally permissible).
- **Monitoring**: rebuild label sources and guard set regularly to catch vendor schema drift.