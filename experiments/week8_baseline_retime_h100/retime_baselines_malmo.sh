#!/usr/bin/env bash
# ============================================================================
# Re-time the six project baselines on malmo's H100 with the SWEEP's harness.
#
# WHY THIS EXISTS
# ---------------
# Figure 2 of notebooks/week8_layer_sweep_plots.ipynb has to be drawn as two
# panels that cannot share an axis. The 18 week-8 sweep engines were timed on
# an H100, trtexec pure GPU compute, --useCudaGraph, inside one exclusive-GPU
# window. The baselines (V0, V1, FP32, FP16, INT8 PTQ, INT8 QAT V6) were only
# ever timed on geneva's A100, kernel-median, no CUDA graph. Put on one axis
# they would say INT8 QAT is 1.397 ms against sweep layers at 0.75 ms and imply
# the sweep engines are ~2x faster -- which is a GPU and a harness difference,
# not a model difference.
#
# This script removes that confound: it rebuilds each baseline's engine on THIS
# H100 from the same ONNX that produced the measured A100 engine (sha256 checked
# before every build), then times it with byte-identical trtexec flags to the
# sweep. Afterwards every latency number in the project is on one axis.
#
# WHAT IS AND IS NOT REPRODUCED
#   reproduced exactly : the ONNX (sha256-gated), workspace 8 GiB, TF32 off,
#                        builder optimization level 3, batch 1, imgsz 640, and
#                        for PTQ the A100 entropy calibration cache, so the INT8
#                        scales are the SAME numbers -- only kernel selection
#                        is allowed to differ.
#   deliberately new   : the GPU (H100 vs A100) and therefore the autotuned
#                        kernels. That is the variable under test.
#   NOT re-measured    : accuracy. Every mAP in the project already comes from
#                        one harness (full val, 6,449 imgs, conf=0.001) and is
#                        already comparable; the ONNX is bit-identical, so the
#                        rebuild cannot move it. Set EVAL=1 to re-check anyway.
#
# EXCLUSIVITY
#   The project rule is: latency only on a verified-idle GPU. Unlike the sweep
#   runner -- which proceeds after its deadline and flags the row -- this script
#   EXITS rather than measure, because a contended reading here would defeat the
#   entire purpose of the run. Exclusivity is re-verified immediately before the
#   timing pass and recorded per reading.
#
# USAGE
#   nohup bash retime_baselines_malmo.sh > retime_nohup.log 2>&1 &
#
# KNOBS
#   GPU=            target GPU. Default: auto-pick the first with no foreign
#                   compute process. Set explicitly to pin one.
#   IDLE_WAIT_MIN=90  how long to wait for a free GPU before giving up.
#   OPT_LEVEL=3     builder optimization level. 3 is both the TensorRT default
#                   (so it is what the A100 baselines were built at) and what
#                   the sweep set explicitly. Do not change without rebuilding
#                   the sweep too.
#   EVAL=0          1 = also re-run full-val mAP on each rebuilt engine
#                   (6 x 6,449 images, adds hours). Off by default.
#   DRY_RUN=1       print the plan, verify every input, launch nothing.
# ============================================================================
set -u

REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT
EXP=$REPO/experiments/week8_baseline_retime_h100
# Engines live off the NFS home: the quota there is at 94% and these are
# throwaway build products. Only provenance, logs and the CSV land in the repo,
# matching the sweep's "engines cleared, provenance kept" convention.
WORK=${WORK:-/tmp/zcemml1_week8_baseline_retime_h100}
PYTRT=/home/zcemml1/venvs/medtronic-trt/bin/python
TRTEXEC=/home/zcemml1/venvs/medtronic-trt/bin/trtexec

OPT_LEVEL=${OPT_LEVEL:-3}
IDLE_WAIT_MIN=${IDLE_WAIT_MIN:-90}
EVAL=${EVAL:-0}
DRY_RUN=${DRY_RUN:-0}

CSV=$EXP/results_baselines_malmo_h100.csv
JOURNAL=$EXP/retime_log.txt
mkdir -p "$WORK" "$EXP"

st(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(st)] $*" | tee -a "$JOURNAL"; }
die(){ say "FATAL: $*"; exit 1; }

# ---------------------------------------------------------------------------
# The six baselines.
#   tag | label | onnx | expected sha256 | kind | A100 kernel-median ms
# kind: fp32 / fp16  -> scripts/tensorrt/build_tensorrt_engine.py
#       ptq          -> same, PRECISION=int8, reusing the A100 calib cache
#       qdq          -> scripts/tensorrt/build_tensorrt_int8_qdq.py
# The A100 column is carried through only so the CSV shows old and new side by
# side; it is NOT re-measured here.
# ---------------------------------------------------------------------------
ROWS=(
"v0_frozen_fp16|V0  frozen baseline (FP16)|$REPO/runs_week8/week8_frozen_head/weights/best.onnx|c98d029e6ee3b37e73328a1b2d2bf5a27e0fac8ec7807df5c76f39e9dd5af3e7|fp16|1.253"
"v1_frozen_qat0|V1  frozen + 0-iter QAT (INT8)|$REPO/experiments/week8_frozen_qat/qat0/best_qat.onnx|504c7610f7e8eaae2453f792407abd6a915a67051ab836dcd069540c8c7dab55|qdq|1.503"
"v2_fp32|FP32 engine (V2)|$REPO/models/yolo26n_sanoscience_full_left/baseline/best.onnx|ac188a00ce0b8c90f07985f93473ee1f61541ddfb3b5b219ada2d4fe7ed911f7|fp32|2.019"
"v3_fp16|FP16 engine (V3)|$REPO/models/yolo26n_sanoscience_full_left/baseline/best.onnx|ac188a00ce0b8c90f07985f93473ee1f61541ddfb3b5b219ada2d4fe7ed911f7|fp16|1.137"
"v4_int8_ptq|INT8 PTQ (V4)|$REPO/models/yolo26n_sanoscience_full_left/baseline/best.onnx|ac188a00ce0b8c90f07985f93473ee1f61541ddfb3b5b219ada2d4fe7ed911f7|ptq|1.091"
"v6_int8_qat|INT8 QAT (V6)|$REPO/models/yolo26n_sanoscience_full_left/qat/v6_final/best_qat.onnx|dc44838a0e09d5e030e9b05144d3cea6c21bf82ae84979f2ed89874fdee42e60|qdq|1.397"
)
CALIB_CACHE_SRC=$REPO/models/yolo26n_sanoscience_full_left/baseline/int8_ptq/best_int8_entropy.calib_cache

# ---------------------------------------------------------------------------
# Foreign compute processes on a GPU. Compute apps only -- this box always has
# a graphics context resident, and counting that means no GPU is ever idle.
# Our own PIDs are excluded by owner.
# ---------------------------------------------------------------------------
foreign(){
  local g=$1 pids p n=0
  pids=$(nvidia-smi -i "$g" --format=csv,noheader,nounits \
           --query-compute-apps=pid 2>/dev/null | tr -d ' ')
  for p in $pids; do
    [ -z "$p" ] && continue
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$USER" ] || n=$((n+1))
  done
  echo "$n"
}
pick_idle(){
  local g
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    [ "$(foreign "$g")" = "0" ] && { echo "$g"; return 0; }
  done
  return 1
}

say "=== baseline re-time on $(hostname) : opt=$OPT_LEVEL eval=$EVAL work=$WORK ==="

# --- 0) verify every input before touching a GPU -----------------------------
for r in "${ROWS[@]}"; do
  IFS='|' read -r tag label onnx sha kind a100 <<< "$r"
  [ -f "$onnx" ] || die "$tag: missing ONNX $onnx"
  got=$(sha256sum "$onnx" | awk '{print $1}')
  [ "$got" = "$sha" ] || die "$tag: ONNX sha256 mismatch
     expected $sha
     got      $got
   The measured A100 engine came from the expected file. Refusing to re-time a
   different model and call it the same baseline."
  say "verified $tag  ($kind)  $(basename "$onnx")"
done
[ -f "$CALIB_CACHE_SRC" ] || die "missing A100 calibration cache $CALIB_CACHE_SRC"
[ -x "$TRTEXEC" ] || die "missing trtexec $TRTEXEC"
say "all 6 ONNX inputs sha256-verified; calibration cache present"

if [ "$DRY_RUN" = "1" ]; then
  say "DRY_RUN -- plan only"
  for g in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
    say "  GPU $g: $(foreign "$g") foreign compute proc(s)"
  done
  say "  would write: $CSV"
  exit 0
fi

# --- 1) wait for a verified-idle GPU ----------------------------------------
if [ -n "${GPU:-}" ]; then
  DEADLINE=$(( $(date +%s) + IDLE_WAIT_MIN * 60 ))
  while [ "$(foreign "$GPU")" != "0" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
    say "GPU $GPU pinned but has $(foreign "$GPU") foreign proc(s); waiting (until $(date -d "@$DEADLINE" '+%H:%M'))"
    sleep 120
  done
  [ "$(foreign "$GPU")" = "0" ] || die "GPU $GPU never went idle within ${IDLE_WAIT_MIN}m. Not measuring: a contended reading is exactly what this run exists to avoid."
else
  DEADLINE=$(( $(date +%s) + IDLE_WAIT_MIN * 60 ))
  while ! GPU=$(pick_idle); do
    [ "$(date +%s)" -lt "$DEADLINE" ] || die "no GPU went idle within ${IDLE_WAIT_MIN}m. Not measuring: a contended reading is exactly what this run exists to avoid."
    say "no idle GPU (0:$(foreign 0) 1:$(foreign 1) 2:$(foreign 2) 3:$(foreign 3)); waiting (until $(date -d "@$DEADLINE" '+%H:%M'))"
    sleep 120
  done
fi
GPU_NAME=$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader)
say "GPU $GPU ($GPU_NAME) is exclusively ours -- proceeding"

# --- 2) build all six engines on this GPU ------------------------------------
# Each build gets its own directory with a symlink to the ONNX, because
# build_tensorrt_engine.py derives the engine path from the ONNX path and would
# otherwise write next to (and shadow) the committed A100 artifacts.
# Sets BUILT (engine path) and returns the BUILDER's exit status. Deliberately
# not via stdout: `x=$(build_one ...)` would make $? the echo's status and every
# build would look like it succeeded.
BUILT=""
build_one(){
  local tag=$1 onnx=$2 kind=$3 rc
  local d=$WORK/$tag; mkdir -p "$d"
  local link=$d/$(basename "$onnx")
  ln -sf "$onnx" "$link"
  local log=$EXP/${tag}_build.log
  BUILT=""
  case $kind in
    fp32|fp16)
      DEVICE=$GPU PRECISION=$kind ONNX_PATH=$link ALLOW_TF32=0 WORKSPACE_GB=8 \
        $PYTRT -u "$REPO/scripts/tensorrt/build_tensorrt_engine.py" > "$log" 2>&1
      rc=$?; BUILT=$d/best_${kind}.engine ;;
    ptq)
      cp -n "$CALIB_CACHE_SRC" "$d/best_int8_entropy.calib_cache"
      DEVICE=$GPU PRECISION=int8 CALIBRATOR=entropy INT8_ALLOW_FP16=0 \
        ONNX_PATH=$link ALLOW_TF32=0 WORKSPACE_GB=8 \
        $PYTRT -u "$REPO/scripts/tensorrt/build_tensorrt_engine.py" > "$log" 2>&1
      rc=$?; BUILT=$d/best_int8.engine ;;
    qdq)
      DEVICE=$GPU OPT_LEVEL=$OPT_LEVEL ONNX_PATH=$link \
        ENGINE_PATH=$d/engine_int8_h100.engine WORKSPACE_GB=8 \
        $PYTRT -u "$REPO/scripts/tensorrt/build_tensorrt_int8_qdq.py" > "$log" 2>&1
      rc=$?; BUILT=$d/engine_int8_h100.engine ;;
  esac
  return $rc
}

declare -A ENGINE
for r in "${ROWS[@]}"; do
  IFS='|' read -r tag label onnx sha kind a100 <<< "$r"
  say "building $tag ($kind) ..."
  t0=$(date +%s)
  build_one "$tag" "$onnx" "$kind"; rc=$?; e=$BUILT
  t1=$(date +%s)
  if [ $rc -ne 0 ] || [ ! -f "$e" ]; then
    say "  BUILD FAILED for $tag (rc=$rc) -- see ${tag}_build.log; row will be marked build_failed"
    ENGINE[$tag]=""
    continue
  fi
  ENGINE[$tag]=$e
  say "  built in $((t1-t0))s -> $(basename "$e") ($(stat -c%s "$e") bytes, sha $(sha256sum "$e" | cut -c1-12))"
  # provenance sidecars are written next to the engine in $WORK; copy into the repo
  for p in "$e".provenance.json "${e%.engine}.engine.provenance.json"; do
    [ -f "$p" ] && cp -f "$p" "$EXP/${tag}_$(basename "$p")"
  done
done

# --- 3) re-verify exclusivity, then time everything in one short window ------
N=$(foreign "$GPU")
if [ "$N" != "0" ]; then
  say "WARNING: GPU $GPU picked up $N foreign process(es) during the ~25 min build."
  say "         Waiting up to 30 min for it to clear before timing."
  D2=$(( $(date +%s) + 1800 ))
  while [ "$(foreign "$GPU")" != "0" ] && [ "$(date +%s)" -lt "$D2" ]; do sleep 60; done
  [ "$(foreign "$GPU")" = "0" ] || die "GPU $GPU still contended. Engines are built and cached in $WORK; re-run to time them once the box frees."
  say "         cleared -- timing now"
fi

# Byte-identical flags to run_layer_malmo.sh step 4. Do not edit one without
# the other: these flags ARE the comparability.
lat(){ CUDA_VISIBLE_DEVICES=$GPU $TRTEXEC --loadEngine="$1" --iterations=300 \
        --warmUp=100 --duration=0 --avgRuns=10 --noDataTransfers --useSpinWait "${@:2}" \
        2>/dev/null | grep -oE 'median = [0-9.]+' | head -1 | awk '{print $3}'; }

echo "tag,label,kind,onnx_sha256,engine_sha256,gpu,gpu_name,opt_level,latency_no_graph_ms,latency_graph_ms,latency_exclusivity,a100_kernel_median_ms,status" > "$CSV"
for r in "${ROWS[@]}"; do
  IFS='|' read -r tag label onnx sha kind a100 <<< "$r"
  e=${ENGINE[$tag]}
  if [ -z "$e" ]; then
    echo "$tag,\"$label\",$kind,$sha,,,$GPU,\"$GPU_NAME\",$OPT_LEVEL,,,,$a100,build_failed" >> "$CSV"
    continue
  fi
  x=$(foreign "$GPU")
  NG=$(lat "$e")
  G=$(lat "$e" --useCudaGraph)
  x2=$(foreign "$GPU")
  excl=$([ "$x" = "0" ] && [ "$x2" = "0" ] && echo exclusive || echo "contended:${x}->${x2}")
  esha=$(sha256sum "$e" | awk '{print $1}')
  say "$tag: no_graph=${NG:-n/a} ms  cuda_graph=${G:-n/a} ms  ($excl)  [A100 kernel-median was $a100]"
  echo "$tag,\"$label\",$kind,$sha,$esha,$GPU,\"$GPU_NAME\",$OPT_LEVEL,${NG:-},${G:-},$excl,$a100,ok" >> "$CSV"
done

# --- 4) optional accuracy re-check ------------------------------------------
if [ "$EVAL" = "1" ]; then
  for r in "${ROWS[@]}"; do
    IFS='|' read -r tag label onnx sha kind a100 <<< "$r"
    e=${ENGINE[$tag]}; [ -z "$e" ] && continue
    say "eval $tag (full val, conf=0.001) ..."
    MODE=engine EVAL_SET=full DEVICE=$GPU ENGINE_PATH=$e \
      $PYTRT -u "$REPO/scripts/evaluate/evaluate_engine_map.py" > "$EXP/${tag}_accuracy.log" 2>&1 \
      || say "  eval returned nonzero for $tag"
  done
fi

say "=== done. table: $CSV ==="
say "engines left in $WORK (off-quota); delete when the numbers are committed"
