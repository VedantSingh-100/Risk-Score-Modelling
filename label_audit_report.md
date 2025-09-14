# Label Audit & Validation (Provisional)

- Rows sampled: 20000
- Events used in UNION: 16
- UNION prevalence: 0.0588
- Weighted label tuned threshold: 2.450 → prevalence 0.0066

## Top contributors among label=1 (first 10)
- var501060: 0.959  |  Flag - BNPL Overdue Occurrence [Lifetime]
- var501052: 0.173  |  No. of BNPL Defaults [Lifetime]
- var501053: 0.116  |  No. of BNPL Defaults [Last 12 Months]
- var501054: 0.030  |  No. of BNPL Defaults [Last 6 Months]
- var206063: 0.014  |  No. of Loan Defaults [Last 28 Days]
- var206064: 0.011  |  No. of Loan Defaults [Last 21 Days]
- var206065: 0.008  |  No. of Loan Defaults [Last 14 Days]
- var206066: 0.006  |  No. of Loan Defaults [Last 7 Days]
- var501055: 0.003  |  No. of BNPL Defaults [Last 3 Months]
- var501102: 0.003  |  No. of BNPL Defaults [Between 60-90 Days]

## Leakage guard
- Guarded features count: 157 (see do_not_use_features.txt)

## Baseline (numeric-only, excluding guarded)
{
  "n_features_used": 150,
  "auc_valid": 0.8426223868212819,
  "brier_valid": 0.16793080313473596
}