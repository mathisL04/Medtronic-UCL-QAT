#!/usr/bin/env bash
# One layer of the Week-8 per-layer QAT sweep, on MALMO (H100 NVL 94GB), GPU 1.
#
# TWO PHASES, deliberately separated:
#
#   PHASE=train   QAT fine-tune with every layer frozen except model.L, and stop.
#                 Produces qat_modelopt_state_best.pt + _trained_malmo.json.
#   PHASE=deploy  export Q/DQ ONNX -> build INT8 TensorRT engine -> full-val mAP
#                 at conf=0.001 -> trtexec latency (with and without CUDA graph)
#                 -> metrics.json.
#   PHASE=all     both, back to back (the original single-pass behaviour).
#
# WHY SPLIT. Two reasons, one practical and one methodological.
#   * Training is the long pole (~50 min/layer); deployment is ~13 min/layer.
#     Getting every layer TRAINED first means the expensive, failure-prone part
#     is banked before any TensorRT work starts, and a build failure late in the
#     sweep costs minutes to redo instead of an hour.
#   * Latency is only comparable across layers if the box was in a comparable
#     state when each reading was taken. Interleaved, the 18 readings would be
#     spread over ~20 h of someone else's workload on a shared GPU. Batched into
#     one deploy pass they sit inside a few hours, which is the closest this
#     shared box gets to a controlled comparison. It is still not an idle GPU --
#     see latency_exclusivity in metrics.json.
#
# Differences from the Geneva (A100) run_layer.sh, all deliberate:
#   * DEVICE defaults to 1 and is the only GPU touched (per the Malmo brief).
#   * Engines are named *_h100.engine: A100 engines are NOT portable across
#     architectures and the two must never be confused in the same directory.
#   * Training budget cut to 6 epochs / patience 3 (Geneva ran 25 / 10). The
#     PER-EPOCH validation still runs on the full 6,449-image val set -- the
#     budget cut is on training time only, nothing is traded away on evaluation.
#   * Param-free layers (Upsample/Concat: model.11/12/14/15/18/21) are recorded
#     as skipped instead of being launched into a guaranteed no-op training run.
#   * The orphan-killer from the Geneva script is GONE. Its ps/awk quoting was
#     broken and it was collateral damage from running shards in parallel; this
#     script runs strictly one layer at a time under a lock instead.
set -u
L=$1
PHASE=${PHASE:-all}
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-6}
PATIENCE=${PATIENCE:-3}
# TensorRT builder search depth, used only in the deploy phase. OPT_LEVEL=3 builds
# in 257 s vs 1531 s at level 5 (6.0x) and costs 7.0% on CUDA-graph median latency
# (0.734 vs 0.686 ms on model.0; 3 repeats, spread <0.3%). Every layer is built at
# the SAME level, so the layer-to-layer ranking is unaffected. Because deployment
# is now its own pass, this can be raised to 5 at deploy time without touching any
# of the training work.
OPT_LEVEL=${OPT_LEVEL:-3}
WORKERS=${WORKERS:-4}
BATCH=${BATCH:-16}

REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT
FROZEN=$REPO/runs_week8/week8_frozen_head/weights/best.pt
SWEEP=$REPO/experiments/week8/layer_sweep
LL=$(printf %02d "$L")
OUT=$SWEEP/layer_$LL
mkdir -p "$OUT"
PY311=/home/zcemml1/venvs/medtronic-qat-p311/bin/python
PYTRT=/home/zcemml1/venvs/medtronic-trt/bin/python
TRTEXEC=/home/zcemml1/venvs/medtronic-trt/bin/trtexec
ENGINE=$OUT/engine_int8_h100.engine
STATE=$OUT/qat_modelopt_state_best.pt
# Training churn goes to LOCAL NVMe, not NFS. The home quota is 50 GB and was at
# 93% (3.7 GB free) when this sweep was launched; /tmp on malmo is a 1.5 TB local
# disk with 1.3 TB free. train_qat.py already documents PROJECT as local scratch --
# the earlier version of this script pointed it at NFS, which contradicted that.
# Only the small final artifacts are copied back to NFS, per layer, immediately.
SCRATCH=${SCRATCH:-/tmp/zcemml1_qat_sweep}/layer_$LL
mkdir -p "$SCRATCH"
MARKER=$OUT/_trained_malmo.json
LOG=$OUT/run_malmo_$PHASE.log
: > "$LOG"
st(){ date '+%H:%M:%S'; }
say(){ echo "[$(st)] $*" | tee -a "$LOG"; }
fail(){ say "$1 -> layer $L failed"; L=$L OUT=$OUT GPU=$GPU EPOCHS=$EPOCHS \
  PATIENCE=$PATIENCE OPT_LEVEL=$OPT_LEVEL STATUS=$2 \
  $PYTRT "$SWEEP/make_metrics_malmo.py" 2>>"$LOG"; exit 0; }

# Cap BLAS/OMP threads. 64 cores x several thread pools starves the box and, under
# this host's vm.overcommit_memory=2, inflates committed address space for no gain.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4

say "=== layer $L  phase=$PHASE epochs=$EPOCHS patience=$PATIENCE workers=$WORKERS gpu=$GPU opt=$OPT_LEVEL (malmo/H100) ==="

# --- 0) layers with no parameters cannot be fine-tuned -----------------------
NOPARAM=" 11 12 14 15 18 21 "
if [[ "$NOPARAM" == *" $L "* ]]; then
  say "model.$L has 0 trainable parameters (Upsample/Concat) -- nothing to fine-tune"
  [ "$PHASE" = "train" ] && exit 0
  L=$L OUT=$OUT GPU=$GPU EPOCHS=$EPOCHS PATIENCE=$PATIENCE OPT_LEVEL=$OPT_LEVEL \
    STATUS=no_trainable_params $PYTRT "$SWEEP/make_metrics_malmo.py" 2>>"$LOG"
  exit 0
fi

# ============================ PHASE: train ===================================
if [ "$PHASE" = "train" ] || [ "$PHASE" = "all" ]; then
  run_train(){
    MODEL_PATH=$FROZEN FREEZE_EXCEPT=$L DEVICE=$GPU EPOCHS=$EPOCHS PATIENCE=$PATIENCE \
      BATCH=$BATCH LR0=0.01 IMG_SIZE=640 CACHE= WORKERS=$1 N_CALIB=128 \
      RUN_NAME=layer_$LL PROJECT=$SCRATCH \
      $PY311 -u "$REPO/scripts/train/train_qat.py" >> "$LOG" 2>&1
  }
  find_state(){
    local s
    s=$(find "$SCRATCH" -name 'qat_modelopt_state_best.pt' 2>/dev/null | head -1)
    [ -z "$s" ] && s=$(find "$SCRATCH" -name 'qat_modelopt_state.pt' 2>/dev/null | head -1)
    echo "$s"
  }

  T0=$(date +%s)
  run_train "$WORKERS" || say "TRAIN returned nonzero (workers=$WORKERS)"
  S=$(find_state)
  if [ -z "$S" ] && [ "$WORKERS" != "0" ]; then
    # Geneva died repeatedly on cv2 OutOfMemoryError from fork-based dataloader
    # workers under vm.overcommit_memory=2. Retry in-process before giving up.
    say "no state after workers=$WORKERS -- retrying in-process (workers=0)"
    run_train 0 || say "TRAIN returned nonzero (workers=0)"
    S=$(find_state)
  fi
  [ -n "$S" ] || fail "NO STATE" train_failed
  cp -p "$S" "$STATE"
  mkdir -p "$OUT/train"
  find "$SCRATCH" \( -name 'results.csv' -o -name 'args.yaml' \) -exec cp -p {} "$OUT/train/" \; 2>/dev/null
  T1=$(date +%s)
  say "QAT state: $S  (train wall $(( (T1-T0)/60 )) min)"
  # Marker records the budget this state was trained under, so the sweep driver
  # can tell "already trained at THIS budget" from "trained at some other one".
  printf '{"layer": %d, "epochs": %d, "patience": %d, "batch": %d, "workers": %d, "trained_on": "malmo / H100 NVL", "train_wall_s": %d, "val_during_training": "full_val_6449"}\n' \
    "$L" "$EPOCHS" "$PATIENCE" "$BATCH" "$WORKERS" "$((T1-T0))" > "$MARKER"
  rm -rf "$SCRATCH"        # state + results.csv are on NFS now; reclaim the scratch
  [ "$PHASE" = "train" ] && { say "=== layer $L TRAIN DONE ==="; exit 0; }
fi

# ============================ PHASE: deploy ==================================
[ -f "$STATE" ] || fail "NO QAT STATE (train phase not run?)" train_failed

# --- 1) export Q/DQ ONNX (CPU: export must not contend with the GPU) ---------
BASE_MODEL=$FROZEN QAT_STATE=$STATE \
  OUT_ONNX=$OUT/best_qat.onnx DEVICE=cpu BATCH=1 \
  $PY311 -u "$REPO/scripts/export/export_qat_onnx.py" >> "$LOG" 2>&1 \
  || say "EXPORT returned nonzero"
[ -f "$OUT/best_qat.onnx" ] || fail "NO ONNX" export_failed

# --- 2) build INT8 engine on THIS architecture -------------------------------
# A100 engines do not load on H100; every engine in this sweep is rebuilt here.
ONNX_PATH=$OUT/best_qat.onnx ENGINE_PATH=$ENGINE DEVICE=$GPU FP16=1 OPT_LEVEL=$OPT_LEVEL DETAILED=1 \
  $PYTRT -u "$REPO/scripts/tensorrt/build_tensorrt_int8_qdq.py" >> "$LOG" 2>&1 \
  || say "BUILD returned nonzero"
[ -f "$ENGINE" ] || fail "NO ENGINE" build_failed

# --- 3) accuracy: full val (6,449 imgs), conf=0.001 --------------------------
MODE=engine EVAL_SET=full DEVICE=$GPU ENGINE_PATH=$ENGINE \
  $PYTRT -u "$REPO/scripts/evaluate/evaluate_engine_map.py" >> "$LOG" 2>&1 \
  || say "ACCURACY returned nonzero"

# --- 4) latency: trtexec, identical flags to the Geneva runs ------------------
# Same knobs as the A100 sweep so hardware is the ONLY changed variable. The GPU
# is not exclusive (see metrics.json latency_exclusivity) -- flagged, not hidden.
lat(){ CUDA_VISIBLE_DEVICES=$GPU $TRTEXEC --loadEngine=$ENGINE --iterations=300 \
        --warmUp=100 --duration=0 --avgRuns=10 --noDataTransfers --useSpinWait "$@" \
        2>/dev/null | grep -oE 'median = [0-9.]+' | head -1 | awk '{print $3}'; }
NG=$(lat)
G=$(lat --useCudaGraph)
say "latency: no_graph=${NG:-n/a} ms  cuda_graph=${G:-n/a} ms"

# --- 5) metrics + master CSV -------------------------------------------------
L=$L OUT=$OUT GPU=$GPU NG=$NG G=$G EPOCHS=$EPOCHS PATIENCE=$PATIENCE OPT_LEVEL=$OPT_LEVEL STATUS=ok \
  $PYTRT "$SWEEP/make_metrics_malmo.py" 2>>"$LOG" | tee -a "$LOG"
say "=== layer $L DEPLOY DONE ==="
