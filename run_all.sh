#!/usr/bin/env bash
# ============================================================================
# run_all.sh — Full sm_attribution pipeline
#
# Usage:
#   ./run_all.sh                     # run all steps (1–4)
#   ./run_all.sh --start-from 3      # resume from step 3
#   ./run_all.sh --dry-run           # print commands without executing
#
# Steps:
#   1  SSI + drought features  (orchestrate_ssi_drought_features.py)
#   2  Temporal correlations   (batch_run_correlations.py)
#   3  Spatial correlations    (orchestrate_drought_feature_spatial_correlations.py)
#   4  AR6 regional metrics    (orchestrate_drought_feature_ar6_metrics.py)
#
# Parallelism (edit below or override via environment):
#   DASK_NUM_WORKERS   — Dask worker processes   (default: nproc - 2)
#   CONCURRENT_MODELS  — outer-loop parallelism   (default: 1)
#
# IMPORTANT: when CONCURRENT_MODELS > 1, set use_distributed: false in
# configs/settings.yml so each thread spawns its own process pool.
# ============================================================================
set -euo pipefail

# ── Tunables ────────────────────────────────────────────────────────────────
START_FROM=${START_FROM:-1}
DRY_RUN=${DRY_RUN:-false}
PERIOD_MODE=${PERIOD_MODE:-both}          # common | maxspan | both
MODEL_SSI_MODE=${MODEL_SSI_MODE:-standalone}
SSI_METHOD=${SSI_METHOD:-deseasonal_ecdf_gpd}
OVERWRITE=${OVERWRITE:-}                  # set to "--overwrite" to force recompute

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-from) START_FROM="$2"; shift 2 ;;
    --dry-run)    DRY_RUN=true;    shift   ;;
    --overwrite)  OVERWRITE="--overwrite"; shift ;;
    --period-mode) PERIOD_MODE="$2"; shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# Export DASK_NUM_WORKERS if not already set
export DASK_NUM_WORKERS="${DASK_NUM_WORKERS:-}"
# Suppress HDF5 C-level error-stack dumps (they bypass Python's stderr)
export HDF5_LOG_LEVEL="${HDF5_LOG_LEVEL:-none}"

SCRIPTS="scripts"
LOGDIR="logs"
mkdir -p "$LOGDIR"

run_step() {
  local step="$1"
  local desc="$2"
  shift 2
  local logfile="$LOGDIR/step${step}_$(date +%Y%m%d_%H%M%S).log"

  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Step $step: $desc"
  echo "  Log: $logfile"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  CMD: python $*"

  if [[ "$DRY_RUN" == "true" ]]; then
    echo "  [dry-run — skipped]"
    return 0
  fi

  if python "$@" 2>&1 | tee "$logfile"; then
    echo "  ✓ Step $step completed."
  else
    echo "  ✗ Step $step FAILED — see $logfile"
    exit 1
  fi
}

# ── Step 1: SSI + drought features ──────────────────────────────────────────
if (( START_FROM <= 1 )); then
  STEP1_ARGS=(
    "$SCRIPTS/orchestrate_ssi_drought_features.py"
    --period-mode "$PERIOD_MODE"
    --model-ssi-mode "$MODEL_SSI_MODE"
    --ssi-method "$SSI_METHOD"
  )
  [[ -n "$OVERWRITE" ]] && STEP1_ARGS+=("$OVERWRITE")
  run_step 1 "SSI + drought features" "${STEP1_ARGS[@]}"
fi

# ── Step 2: temporal correlations ───────────────────────────────────────────
if (( START_FROM <= 2 )); then
  # batch_run_correlations uses --period-mode common|maxspan (not "both"),
  # so we run once per mode when PERIOD_MODE=both.
  _corr_modes=()
  if [[ "$PERIOD_MODE" == "both" ]]; then
    _corr_modes=(common maxspan)
  else
    _corr_modes=("$PERIOD_MODE")
  fi

  for pm in "${_corr_modes[@]}"; do
    STEP2_ARGS=(
      "$SCRIPTS/batch_run_correlations.py"
      --period-mode "$pm"
      --mode "$MODEL_SSI_MODE"
      --ssi-method "$SSI_METHOD"
      --target both
    )
    [[ -n "$OVERWRITE" ]] && STEP2_ARGS+=("$OVERWRITE")
    run_step 2 "Temporal correlations (${pm})" "${STEP2_ARGS[@]}"
  done
fi

# ── Step 3: spatial correlations ────────────────────────────────────────────
if (( START_FROM <= 3 )); then
  _spat_modes=()
  if [[ "$PERIOD_MODE" == "both" ]]; then
    _spat_modes=(common maxspan)
  else
    _spat_modes=("$PERIOD_MODE")
  fi

  for pm in "${_spat_modes[@]}"; do
    STEP3_ARGS=(
      "$SCRIPTS/orchestrate_drought_feature_spatial_correlations.py"
      --period-mode "$pm"
      --model-ssi-mode "$MODEL_SSI_MODE"
      --ssi-method "$SSI_METHOD"
    )
    [[ -n "$OVERWRITE" ]] && STEP3_ARGS+=("$OVERWRITE")
    run_step 3 "Spatial correlations (${pm})" "${STEP3_ARGS[@]}"
  done
fi

# ── Step 4: AR6 regional metrics ────────────────────────────────────────────
if (( START_FROM <= 4 )); then
  _ar6_modes=()
  if [[ "$PERIOD_MODE" == "both" ]]; then
    _ar6_modes=(common maxspan)
  else
    _ar6_modes=("$PERIOD_MODE")
  fi

  for pm in "${_ar6_modes[@]}"; do
    STEP4_ARGS=(
      "$SCRIPTS/orchestrate_drought_feature_ar6_metrics.py"
      --period-mode "$pm"
      --model-ssi-mode "$MODEL_SSI_MODE"
      --ssi-method "$SSI_METHOD"
    )
    [[ -n "$OVERWRITE" ]] && STEP4_ARGS+=("$OVERWRITE")
    run_step 4 "AR6 regional metrics (${pm})" "${STEP4_ARGS[@]}"
  done
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Pipeline complete."
echo "  Logs: $LOGDIR/"
echo "════════════════════════════════════════════════════════════════"
