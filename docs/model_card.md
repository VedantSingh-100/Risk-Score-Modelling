# Risk Feature Discovery — Run Report

## Dataset
- Rows: **49389**; Columns: **1638**; Memory: **772.7 MB**

## Outputs
- Variable catalog: `variable_catalog.csv`
- Candidate labels: `candidate_targets_ranked.csv`
- Consensus ranking: `feature_importance_consensus.csv`

## Supervised ranking
- Label used: **var101022**
- Positivity rate: **0.1692**
- L1-Logistic AUC (valid): **0.7404302955051536**
- HGB (tree) AUC (valid): **1.0**

**Recommended next:**
- Add categorical encoders (OneHot with max_categories or FeatureHasher) and re-run.
- Run stability selection: multiple seeds/folds; keep features stable across runs.
