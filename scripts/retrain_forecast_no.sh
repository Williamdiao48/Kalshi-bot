#!/usr/bin/env bash
#
# Rolling-window retrain for the forecast-NO band models (V2 high/low LightGBM
# + isotonic calibrator).  Chains the existing pipeline end-to-end:
#
#   1. extend band_arb_hist_cache.json  (IEM obs + HRRR/OM forecasts → today)
#   2. refresh kalshi_markets_cache.json (force re-fetch of settled summer bands)
#   3. build_kalshi_no_training_data.py  → base training CSV
#   4. build_training_data_v2.py         → v2 feature CSV
#   5. back up the live model .pkl files
#   6. train_forecast_no_model_v2.py     → overwrites the deployed V2 models
#   7. print validation metrics for a manual promote/rollback decision
#
# Nothing here auto-promotes: the trainer overwrites the live .pkl in place, so
# step 5 keeps a timestamped backup you can restore if step 7 looks worse than
# the incumbent.  The bot only picks up new models on restart.
#
# Requirements: same environment as the bot (Kalshi creds available for the
# settled-markets re-fetch).  Run from the repo root.
#
# Usage:
#   scripts/retrain_forecast_no.sh                 # auto window, refresh Kalshi cache
#   scripts/retrain_forecast_no.sh --no-refresh-kalshi
#   scripts/retrain_forecast_no.sh --dry-run-cache # only preview cache extension
#
set -euo pipefail

cd "$(dirname "$0")/.."
PY="venv/bin/python"
TS="$(date -u +%Y%m%d_%H%M%S)"
BACKUP_DIR="data/models/backup_${TS}"
HIST_CACHE="data/backtest/band_arb_hist_cache.json"
KALSHI_CACHE="data/backtest/kalshi_markets_cache.json"

REFRESH_KALSHI=1
DRY_RUN_CACHE=0
for arg in "$@"; do
  case "$arg" in
    --no-refresh-kalshi) REFRESH_KALSHI=0 ;;
    --dry-run-cache)     DRY_RUN_CACHE=1 ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

# ── 1. Extend the historical weather cache ────────────────────────────────────
log "1/7  Extend $HIST_CACHE (auto window → today)"
if [[ "$DRY_RUN_CACHE" == "1" ]]; then
  $PY scripts/extend_hist_cache.py --dry-run
  echo "dry-run-cache set — stopping after preview."
  exit 0
fi
$PY scripts/extend_hist_cache.py

# ── 2. Refresh the settled-markets cache ──────────────────────────────────────
# Kalshi's status=settled endpoint ages out old markets, so a plain re-fetch
# returns only a recent window and would DROP the older settled markets the
# archive holds.  We therefore MERGE: the builder (step 3, --merge-kalshi) fetches
# fresh and unions the new markets into the existing cache, preserving history.
# A timestamped backup is kept in case the merge misbehaves.
BUILD_ARGS=()
if [[ "$REFRESH_KALSHI" == "1" ]]; then
  log "2/7  Refresh $KALSHI_CACHE (merge fresh fetch → preserve history)"
  if [[ -f "$KALSHI_CACHE" ]]; then
    cp "$KALSHI_CACHE" "${KALSHI_CACHE}.bak_${TS}"
    echo "backed up Kalshi cache → ${KALSHI_CACHE}.bak_${TS}"
  fi
  BUILD_ARGS+=(--merge-kalshi)
else
  log "2/7  Skipping Kalshi-cache refresh (--no-refresh-kalshi); using cache as-is"
fi

# ── 3. Rebuild the base training CSV ──────────────────────────────────────────
log "3/7  build_kalshi_no_training_data.py ${BUILD_ARGS[*]}"
$PY scripts/build_kalshi_no_training_data.py "${BUILD_ARGS[@]}"

# ── 4. Rebuild the v2 feature CSV ─────────────────────────────────────────────
log "4/7  build_training_data_v2.py"
$PY scripts/build_training_data_v2.py

# ── 5. Back up the live models before overwrite ───────────────────────────────
log "5/7  Back up live models → $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp -v data/models/forecast_no_band_model_v2_high.pkl "$BACKUP_DIR/" 2>/dev/null || true
cp -v data/models/forecast_no_band_model_v2_low.pkl  "$BACKUP_DIR/" 2>/dev/null || true

# ── 6. Train (overwrites deployed V2 models in place) ─────────────────────────
# --production: fit the isotonic calibrator on the newest slice and skip the
# internal held-out test (validation is done out-of-band in step 7).
log "6/7  train_forecast_no_model_v2.py --production"
$PY scripts/train_forecast_no_model_v2.py --production

# ── 7. Validation for a manual promote/rollback decision ──────────────────────
log "7/7  Validation"
if [[ -f scripts/backtest_v2_model.py ]]; then
  $PY scripts/backtest_v2_model.py || echo "(backtest_v2_model.py exited non-zero — inspect above)"
elif [[ -f scripts/validate_forecast_no_model.py ]]; then
  $PY scripts/validate_forecast_no_model.py || echo "(validate exited non-zero — inspect above)"
else
  echo "No validation script found — review the train output's Brier/AUC/calibration table above."
fi

cat <<EOF

Done.  New V2 models are live on disk; the bot loads them on next restart.
Rollback if validation regressed:
  cp $BACKUP_DIR/*.pkl data/models/
EOF
