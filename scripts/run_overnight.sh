#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Parshvi Risk Model — Overnight training pipeline
# 1) GBDT sweep (XGB/LGBM) with CV, calibration, deciles, KS, Gini
# 2) Baseline MLP (TabMLP) with comprehensive wandb logging
# 3) MLP stacking (sklearn MLP + calibrated XGB+MLP stack)
# All outputs -> model_outputs/
# ============================================================
export WANDB_API_KEY="3ff6a13421fb5921502235dde3f9a4700f33b5b8"
export WANDB_MODE="online"
# ---------- knobs you may tweak ----------
TRIALS="${TRIALS:-60}"           # # of GBDT trials (increase for longer runs)
FOLDS="${FOLDS:-5}"              # CV folds
SEED="${SEED:-42}"
ALGO="${ALGO:-xgb}"              # xgb | lgb
USE_WANDB="${USE_WANDB:-0}"      # 1 to enable W&B logging
WANDB_PROJECT="${WANDB_PROJECT:-Risk_Score}"
USE_MONOTONE="${USE_MONOTONE:-0}" # 1 to read data/monotone_config.json if present

# ---------- paths ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/processed}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/model_outputs}"
LOG_DIR="${REPO_ROOT}/artifacts/logs"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

STAMP="$(date +'%Y%m%d_%H%M%S')"
LOGFILE="${LOG_DIR}/overnight_${STAMP}.log"

echo "=== Overnight run @ ${STAMP} ===" | tee -a "${LOGFILE}"
echo "Repo: ${REPO_ROOT}" | tee -a "${LOGFILE}"
echo "Data: ${DATA_ROOT}" | tee -a "${LOGFILE}"
echo "Out : ${OUT_ROOT}" | tee -a "${LOGFILE}"
echo "Algo: ${ALGO} | Trials: ${TRIALS} | Folds: ${FOLDS} | Seed: ${SEED}" | tee -a "${LOGFILE}"
echo "WANDB: ${USE_WANDB} (${WANDB_PROJECT}) | Monotone: ${USE_MONOTONE}" | tee -a "${LOGFILE}"
echo "---------------------------------------------" | tee -a "${LOGFILE}"

# ---------- (optional) Weights & Biases ----------
WANDB_FLAG=""
if [[ "${USE_WANDB}" == "1" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "[WARN] USE_WANDB=1 but WANDB_API_KEY is not set. Disabling W&B." | tee -a "${LOGFILE}"
  else
    export WANDB_API_KEY
    export WANDB_PROJECT
    export WANDB_ENTITY="ved100-carnegie-mellon-university"
    WANDB_FLAG="--wandb --wandb-project ${WANDB_PROJECT} --wandb-entity ${WANDB_ENTITY}"
  fi
fi

# ---------- Part A: GBDT sweep ----------
echo "[1/3] GBDT sweep starting..." | tee -a "${LOGFILE}"
GBDT_DIR="${OUT_ROOT}/gbdt_sweep_${ALGO}_${STAMP}"
MONO_FLAG=""
[[ "${USE_MONOTONE}" == "1" ]] && MONO_FLAG="--use-monotone"

python -m src.models.train_gbdt_sweep \
  --data-root "${DATA_ROOT}" \
  --out-dir "${GBDT_DIR}" \
  --algo "${ALGO}" \
  --trials "${TRIALS}" \
  --n-splits "${FOLDS}" \
  --seed "${SEED}" \
  ${MONO_FLAG} ${WANDB_FLAG} 2>&1 | tee -a "${LOGFILE}"

echo "[1/3] GBDT sweep done. Best params at: ${GBDT_DIR}/best_params.json" | tee -a "${LOGFILE}"
echo "      Trials summary at       : ${GBDT_DIR}/trials_summary.csv"      | tee -a "${LOGFILE}"
echo "---------------------------------------------" | tee -a "${LOGFILE}"

# ---------- Part B: Baseline MLP (TabMLP) ----------
echo "[2/3] Baseline MLP (TabMLP) starting..." | tee -a "${LOGFILE}"
TABMLP_DIR="${OUT_ROOT}/tabmlp_baseline_${STAMP}"

python -m src.models.train_baseline_mlp \
  --data-root "${DATA_ROOT}" \
  --out-dir "${TABMLP_DIR}" \
  --epochs 140 \
  --lr 1e-3 \
  --wd 1e-4 \
  --patience 18 \
  --n-splits "${FOLDS}" \
  --seed "${SEED}" \
  ${WANDB_FLAG} 2>&1 | tee -a "${LOGFILE}"

echo "[2/3] Baseline MLP done. Summary: ${TABMLP_DIR}/summary.json" | tee -a "${LOGFILE}"
echo "      OOF predictions: ${TABMLP_DIR}/oof_deep.csv"           | tee -a "${LOGFILE}"
echo "      Model weights: ${TABMLP_DIR}/tabmlp_state.pt"         | tee -a "${LOGFILE}"
echo "---------------------------------------------" | tee -a "${LOGFILE}"

# ---------- Part C: MLP + calibrated stack ----------
echo "[3/3] MLP + Stack starting..." | tee -a "${LOGFILE}"
STACK_DIR="${OUT_ROOT}/stack_${STAMP}"

python -m src.models.train_mlp_stack \
  --data-root "${DATA_ROOT}" \
  --out-dir "${STACK_DIR}" \
  --best-xgb-params "${GBDT_DIR}/best_params.json" \
  --n-splits "${FOLDS}" \
  --seed "${SEED}" 2>&1 | tee -a "${LOGFILE}"

echo "[3/3] MLP + Stack done. Stack summary: ${STACK_DIR}/summary.csv" | tee -a "${LOGFILE}"
echo "OOF stack preds: ${STACK_DIR}/oof_predictions.csv"               | tee -a "${LOGFILE}"
echo "Deciles (xgb/mlp/stack): ${STACK_DIR}/deciles_*.csv"            | tee -a "${LOGFILE}"

echo "=== Overnight training completed. Logs: ${LOGFILE} ==="
