#!/usr/bin/env bash
# Run ONE labelled QAT sweep version end-to-end and document it. REUSES Phase-1 scripts only.
# One call = one isolated, documented version (this is what an external sweep loops over).
#
# Usage:
#   VERSION=V2_calib_percentile_99.99 DEVICE=2 CALIB_METHOD=percentile \
#     bash experiments/qat_iteration_2/run_version.sh
set -o pipefail
REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT
PY_TRAIN=/home/zcemml1/venvs/medtronic-qat-p311/bin/python
PY_TRT=/home/zcemml1/venvs/medtronic-trt/bin/python
cd "$REPO" || exit 1

VERSION="${VERSION:?VERSION required (e.g. V2_calib_percentile_99.99)}"
DEVICE="${DEVICE:?DEVICE required}"
EVAL_SET="${EVAL_SET:-full}"
OUT="$REPO/experiments/qat_iteration_2/sweeps/$VERSION"
mkdir -p "$OUT"
stamp() { date '+%H:%M:%S'; }
echo "[$(stamp)] === VERSION $VERSION  device=$DEVICE  eval=$EVAL_SET ==="

# --- record the full knob set for this run ---
cat > "$OUT/run_config.json" <<JSON
{ "version":"$VERSION", "lr0":"${LR0:-1e-3}", "lrf":"${LRF:-0.01}", "epochs":"${EPOCHS:-50}", "patience":"${PATIENCE:-10}",
  "batch":"${BATCH:-16}", "n_calib":"${N_CALIB:-128}", "weight_axis":"${WEIGHT_AXIS:-0}",
  "disable_layers":"${DISABLE_LAYERS:-}", "calib_method":"${CALIB_METHOD:-max}",
  "calib_percentile":"${CALIB_PERCENTILE:-99.99}", "calib_num_bins":"${CALIB_NUM_BINS:-2048}" }
JSON

# --- 1. TRAIN + calibrate + save state (qat_run.sh, NFS_OUT redirected into the version folder) ---
echo "[$(stamp)] 1/5 train (qat_run.sh)"
NFS_OUT="$OUT" RUN_NAME="$VERSION" PY="$PY_TRAIN" bash scripts/train/qat_run.sh || { echo "TRAIN FAILED"; exit 1; }
[ -f "$OUT/qat_modelopt_state_best.pt" ] || cp -p "$OUT/qat_modelopt_state.pt" "$OUT/qat_modelopt_state_best.pt" 2>/dev/null

# --- 2. EXPORT ONNX ---
echo "[$(stamp)] 2/5 export ONNX"
# BATCH=1 is FORCED here: export_qat_onnx.py reads $BATCH for the ONNX input shape, and the
# training BATCH must not leak into the deployment engine (which is always batch-1).
QAT_STATE="$OUT/qat_modelopt_state_best.pt" OUT_ONNX="$OUT/best_qat.onnx" DEVICE="$DEVICE" BATCH=1 \
  "$PY_TRAIN" -u scripts/export/export_qat_onnx.py || { echo "EXPORT FAILED"; exit 1; }

# --- 3. BUILD INT8 engine ---
echo "[$(stamp)] 3/5 build engine"
ONNX_PATH="$OUT/best_qat.onnx" ENGINE_PATH="$OUT/best_qat_int8.engine" DEVICE="$DEVICE" \
  "$PY_TRT" -u scripts/tensorrt/build_tensorrt_int8_qdq.py || { echo "BUILD FAILED"; exit 1; }

# --- 4. ACCURACY (pycocotools) ---
echo "[$(stamp)] 4/5 accuracy ($EVAL_SET)"
MODE=engine EVAL_SET="$EVAL_SET" DEVICE="$DEVICE" ENGINE_PATH="$OUT/best_qat_int8.engine" \
  "$PY_TRT" -u scripts/evaluate/evaluate_engine_map.py || { echo "EVAL FAILED"; exit 1; }

# --- 5. LATENCY (kernel; non-fatal if GPU not exclusive) ---
echo "[$(stamp)] 5/5 latency"
LAT_LOG="$OUT/_latency.log"
DEVICE="$DEVICE" BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-10}" ENGINE_PATH="$OUT/best_qat_int8.engine" \
  "$PY_TRT" -u scripts/benchmark/benchmark_latency_trt.py > "$LAT_LOG" 2>&1 || echo "  (latency non-exclusive / skipped)"
KM=$(grep -iE 'Kernel\(GPU\)' "$LAT_LOG" | head -1 | awk '{print $4}')
EXCL=$(grep -c 'exclusive' "$LAT_LOG")
printf '{"kernel_median_ms": %s, "exclusive_gate_lines": %s}\n' "${KM:-null}" "$EXCL" > "$OUT/latency.json"

# --- 6. DOCUMENT (version.md + master_comparison.md) ---
echo "[$(stamp)] doc"
VERSION="$VERSION" OUT="$OUT" "$PY_TRAIN" experiments/qat_iteration_2/gen_version_doc.py
echo "[$(stamp)] === DONE $VERSION ==="
