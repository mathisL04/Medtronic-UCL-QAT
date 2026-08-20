#!/usr/bin/env bash
# Arms the Week-8 DEPLOY pass to fire the moment the TRAIN pass finishes.
#
# The two passes were split so that every latency reading is taken inside one
# short window instead of being spread across ~20 h of a shared GPU's varying
# load (see the header of run_layer_malmo.sh). That only pays off if deploy
# starts promptly when training ends -- hence this waiter, rather than a human
# watching the journal.
#
#   nohup bash queue_deploy_malmo.sh > queue_deploy_nohup.log 2>&1 &
#
# Knobs:
#   IDLE_WAIT_MIN=45  how long to wait for GPU $GPU to have NO other user's
#                     compute process before starting. On expiry it proceeds
#                     anyway -- a shared box may never go idle, and every row
#                     records what was actually resident (latency_exclusivity),
#                     so a contended reading is flagged, not hidden.
#                     Set 0 to skip the wait entirely.
#   OPT_LEVEL=3       TensorRT builder depth. MEASURED on this box (malmo GPU 1),
#                     layer 0, 3 trtexec repeats each, spread <0.4%,
#                     --noDataTransfers so this is pure GPU compute:
#                       level 1:  graph 0.762 ms   no-graph 1.048 ms
#                       level 3:  graph 0.735 ms   no-graph 0.969 ms   build  257 s
#                       level 5:  graph 0.686 ms   no-graph 0.864 ms   build 1531 s
#                     Level 3 is the knee: it is FASTER to build than level 1 and
#                     4% quicker at inference, while level 5 costs 6x the build
#                     (+6 h over 24 layers) for a further 7%. An earlier attempt
#                     here measured level 3 as ">28 min, unfinished" and switched
#                     to level 1 on that basis -- that build was competing with
#                     the training pass for the same GPU and the reading was an
#                     artefact of self-contention, not of the builder level.
#                     Every layer builds at the SAME level regardless, so the
#                     layer-to-layer RANKING -- the point of the sweep -- is
#                     unaffected; only absolute latency moves.
#   DRY_RUN=1         print the plan and exit without waiting or launching.
set -u
SWEEP=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/experiments/week8_layer_sweep
GPU=${GPU:-1}
EPOCHS=${EPOCHS:-6}
PATIENCE=${PATIENCE:-3}
OPT_LEVEL=${OPT_LEVEL:-3}
IDLE_WAIT_MIN=${IDLE_WAIT_MIN:-45}
DRY_RUN=${DRY_RUN:-0}
# All 24. The 18 trainable layers deploy their QAT state; the 6 param-free ones
# (11 12 14 15 18 21) short-circuit to a no_trainable_params row, which is what
# makes the master CSV a complete 24-row table instead of one with silent holes.
LAYERS=${LAYERS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"}
JOURNAL=$SWEEP/sweep_malmo_log.txt
Q=$SWEEP/queue_deploy_malmo.log
st(){ date '+%Y-%m-%d %H:%M:%S'; }
say(){ echo "[$(st)] $*" | tee -a "$Q"; }

# Other users' compute processes on our GPU. Compute apps only: this box always
# has Xorg holding a graphics context, and counting that would mean the GPU never
# looks idle. Our own PIDs are excluded by matching the process owner.
foreign(){
  local pids n=0
  pids=$(nvidia-smi -i "$GPU" --format=csv,noheader,nounits \
           --query-compute-apps=pid 2>/dev/null | tr -d ' ')
  for p in $pids; do
    [ -z "$p" ] && continue
    [ "$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')" = "$USER" ] || n=$((n+1))
  done
  echo "$n"
}

say "=== deploy queued: layers [$LAYERS] gpu=$GPU opt=$OPT_LEVEL idle_wait=${IDLE_WAIT_MIN}m ==="

if [ "$DRY_RUN" = "1" ]; then
  say "DRY_RUN -- plan only, nothing launched"
  say "would wait on: $(pgrep -u "$USER" -f 'run_sweep_malmo.sh' | tr '\n' ' ')"
  say "foreign compute procs on GPU $GPU right now: $(foreign)"
  say "trained markers present: $(ls "$SWEEP"/layer_*/_trained_malmo.json 2>/dev/null | wc -l)/18"
  exit 0
fi

# --- 1) wait for the train pass ---------------------------------------------
# Wait on the DRIVER pid, not the journal: the journal's COMPLETE line is written
# after the last layer, but the driver still holds the flock until it exits, and
# run_sweep_malmo.sh refuses (flock -n) rather than queueing.
TRAIN_PID=$(pgrep -u "$USER" -f 'bash run_sweep_malmo.sh' | head -1)
if [ -n "$TRAIN_PID" ]; then
  say "waiting for train driver pid $TRAIN_PID to exit"
  while kill -0 "$TRAIN_PID" 2>/dev/null; do sleep 60; done
  say "train driver exited"
else
  say "no train driver running -- proceeding immediately"
fi

if grep -q "SWEEP COMPLETE phase=train" "$JOURNAL" 2>/dev/null; then
  say "journal confirms train pass completed"
else
  say "WARNING: no 'SWEEP COMPLETE phase=train' in the journal -- the train pass"
  say "         may have been killed. Deploying anyway; layers with no QAT state"
  say "         get a train_failed row rather than being silently dropped."
fi
say "trained markers: $(ls "$SWEEP"/layer_*/_trained_malmo.json 2>/dev/null | wc -l)/18"

# --- 2) give the GPU a chance to clear --------------------------------------
if [ "$IDLE_WAIT_MIN" != "0" ]; then
  DEADLINE=$(( $(date +%s) + IDLE_WAIT_MIN * 60 ))
  while [ "$(foreign)" != "0" ] && [ "$(date +%s)" -lt "$DEADLINE" ]; do
    say "GPU $GPU has $(foreign) other compute process(es); waiting (until $(date -d "@$DEADLINE" '+%H:%M'))"
    sleep 120
  done
fi
N=$(foreign)
if [ "$N" = "0" ]; then
  say "GPU $GPU is exclusively ours -- latency readings will be clean"
else
  say "proceeding with $N other compute process(es) on GPU $GPU; every row records"
  say "this in latency_exclusivity, so the column is flagged rather than trusted"
fi

# --- 3) run it ---------------------------------------------------------------
say "launching deploy pass"
PHASE=deploy GPU=$GPU EPOCHS=$EPOCHS PATIENCE=$PATIENCE OPT_LEVEL=$OPT_LEVEL \
  bash "$SWEEP/run_sweep_malmo.sh" $LAYERS >> "$SWEEP/sweep_malmo_deploy_nohup.log" 2>&1
say "=== deploy pass returned (exit $?) ==="
say "master table: $SWEEP/results_master_malmo_h100.csv"
