#!/usr/bin/env bash
# Driver for the Week-8 per-layer QAT sweep on MALMO / H100 NVL, GPU 1.
#
# Usage:  PHASE=train  bash run_sweep_malmo.sh 0 1 2 ... [layers]
#         PHASE=deploy bash run_sweep_malmo.sh 0 1 2 ... [layers]
#
# STRICTLY ONE LAYER AT A TIME, under a lock. The Geneva run failed not on the
# science but on concurrency: three shards were launched into the same layer
# directories at once, two trainings wrote the same run.log and the same
# qat_modelopt_state_best.pt, and the host ran out of committable memory
# (cv2 "Insufficient memory" on a 3.5 MB allocation). Serialising removes all
# three failure modes; the lock makes a second accidental launch a no-op.
#
# Both phases share the lock, so a deploy pass cannot be started on top of a
# still-running train pass.
set -u
PHASE=${PHASE:-train}
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-6}
PATIENCE=${PATIENCE:-3}
OPT_LEVEL=${OPT_LEVEL:-3}
WORKERS=${WORKERS:-4}
REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT
SWEEP=$REPO/experiments/week8_layer_sweep
LOCK=$SWEEP/.sweep_malmo.lock
JOURNAL=$SWEEP/sweep_malmo_log.txt
st(){ date '+%Y-%m-%d %H:%M:%S'; }

case "$PHASE" in train|deploy) ;; *) echo "PHASE must be train or deploy"; exit 2 ;; esac

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "another sweep already holds $LOCK -- refusing to start a second one"
  exit 1
fi

echo "[$(st)] SWEEP start phase=$PHASE (malmo/H100 gpu $GPU, epochs=$EPOCHS patience=$PATIENCE opt=$OPT_LEVEL): $*" >> "$JOURNAL"
for L in "$@"; do
  LL=$(printf %02d "$L")
  OUT=$SWEEP/layer_$LL
  M=$OUT/metrics.json
  MARKER=$OUT/_trained_malmo.json

  # Resume. Each phase has its own definition of "already done":
  #   train  -> a Malmo marker exists AND it records the budget we are asking for.
  #             (A Geneva-trained state has no marker, so it is retrained -- which
  #             is what we want: layers 0/13/16/17 ran at 15-25 epochs and would
  #             otherwise sit at the top of the ranking on budget alone.)
  #   deploy -> metrics.json already carries an H100 hardware record.
  if [ "$PHASE" = "train" ]; then
    if [ -f "$MARKER" ] && grep -q "\"epochs\": $EPOCHS," "$MARKER" 2>/dev/null; then
      echo "[$(st)] layer $L already trained on malmo at epochs=$EPOCHS -> skip" >> "$JOURNAL"; continue
    fi
  else
    if [ -f "$M" ] && grep -q '"gpu": "NVIDIA H100' "$M" 2>/dev/null; then
      echo "[$(st)] layer $L already has an H100 metrics record -> skip" >> "$JOURNAL"; continue
    fi
  fi

  echo "[$(st)] layer $L START ($PHASE)" >> "$JOURNAL"
  PHASE=$PHASE GPU=$GPU EPOCHS=$EPOCHS PATIENCE=$PATIENCE WORKERS=$WORKERS OPT_LEVEL=$OPT_LEVEL \
    bash "$SWEEP/run_layer_malmo.sh" "$L" >> "$SWEEP/layer_${LL}_malmo_${PHASE}.log" 2>&1

  if [ "$PHASE" = "train" ]; then
    if [ -f "$MARKER" ]; then
      echo "[$(st)] layer $L TRAINED  $(tr -d '{}\"' < "$MARKER" | cut -c1-120)" >> "$JOURNAL"
    else
      echo "[$(st)] layer $L TRAIN FAILED (no marker)" >> "$JOURNAL"
    fi
  else
    if [ -f "$M" ]; then
      echo "[$(st)] layer $L DEPLOYED  $(grep -oE '"(status|map50_95|latency_graph_ms)": [^,]+' "$M" | tr '\n' ' ')" >> "$JOURNAL"
    else
      echo "[$(st)] layer $L DEPLOY FAILED (no metrics.json)" >> "$JOURNAL"
    fi
  fi
done
echo "[$(st)] SWEEP COMPLETE phase=$PHASE: $*" >> "$JOURNAL"
