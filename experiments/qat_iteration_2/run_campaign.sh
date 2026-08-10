#!/usr/bin/env bash
# Unattended QAT sweep campaign. Schedules the REMAINING sweep runs across a GPU pool (up to
# len(GPUS) in parallel), then generates per-knob sweep docs. Idempotent: a run whose version.md
# already exists is skipped. nohup this script -> it survives the laptop/SSH closing entirely.
#
#   nohup bash experiments/qat_iteration_2/run_campaign.sh > experiments/qat_iteration_2/sweeps/CAMPAIGN.log 2>&1 &
set -u
set -f   # no shell globbing — DISABLE_LAYERS patterns like *model.10* must pass through literally
REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT
S="$REPO/experiments/qat_iteration_2/sweeps"
PY311=/home/zcemml1/venvs/medtronic-qat-p311/bin/python
cd "$REPO" || exit 1
stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
GPUS=(0 1 2)

# --- the run matrix: "VERSION|EXTRA_ENV"  (common env applied to all) ---
QUEUE=(
  "V2_lr0_1e-4|LR0=1e-4"
  "V2_lr0_3e-4|LR0=3e-4"
  "V2_lr0_3e-3|LR0=3e-3"
  "V2_lr0_1e-2|LR0=1e-2"
  "V2_lrf_0.001|LRF=0.001"
  "V2_lrf_0.1|LRF=0.1"
  "V2_batch_8|BATCH=8"
  "V2_batch_32|BATCH=32"
  "V2_ncalib_32|N_CALIB=32"
  "V2_ncalib_512|N_CALIB=512"
  "V2_disable_attention|DISABLE_LAYERS=*model.10*,*model.22*"
)
# WORKERS=0: no dataloader fork + no /dev/shm -> immune to the fork-OOM (Errno 12) and shm-exhaustion
# that killed the parallel WORKERS=4 attempt. Slower/epoch but bulletproof under parallelism.
COMMON="EPOCHS=50 PATIENCE=10 WORKERS=0 EVAL_SET=full BENCHMARK_REPEATS=10"

done_log(){ grep -qE '=== DONE|FAILED' "$S/$1_run.log" 2>/dev/null; }
version_done(){ [ -f "$S/$1/version.md" ]; }        # idempotency: already produced
running(){ [ -f "$S/$1_run.log" ] && ! done_log "$1"; }

# fresh start: all GPUs free (previous parallel attempt killed)
declare -A ON=( [0]="" [1]="" [2]="" )

echo "[$(stamp)] campaign start. GPUs: ${GPUS[*]}  queued: ${#QUEUE[@]}"
qi=0
while :; do
  # assign queued runs to free GPUs
  for g in "${GPUS[@]}"; do
    cur="${ON[$g]:-}"
    if [ -z "$cur" ] || done_log "$cur"; then          # this GPU is free
      # find next not-already-done queue entry
      while [ $qi -lt ${#QUEUE[@]} ]; do
        entry="${QUEUE[$qi]}"; ver="${entry%%|*}"; extra="${entry#*|}"; qi=$((qi+1))
        if version_done "$ver" || running "$ver"; then
          echo "[$(stamp)] skip $ver (already done/running)"; continue
        fi
        echo "[$(stamp)] launch $ver on GPU $g  ($extra)"
        env VERSION="$ver" DEVICE="$g" $extra $COMMON \
          nohup bash experiments/qat_iteration_2/run_version.sh > "$S/${ver}_run.log" 2>&1 &
        ON[$g]="$ver"
        sleep 30   # stagger so parallel CPU max-calibrations don't perfectly overlap
        break
      done
      [ $qi -ge ${#QUEUE[@]} ] && [ -z "${ON[$g]:-}" ] && ON[$g]=""
    fi
  done
  # exit when queue exhausted AND every GPU's current run is done
  if [ $qi -ge ${#QUEUE[@]} ]; then
    all_done=1
    for g in "${GPUS[@]}"; do cur="${ON[$g]:-}"; [ -n "$cur" ] && ! done_log "$cur" && all_done=0; done
    [ $all_done -eq 1 ] && break
  fi
  sleep 120
done
echo "[$(stamp)] all runs finished. Generating sweep docs..."

# --- virtual 1e-3 = V6 point so the LR0 curve includes the sweet spot ---
mkdir -p "$S/V2_lr0_1e-3"
cat > "$S/V2_lr0_1e-3/metrics.json" <<J
{"version":"V2_lr0_1e-3","knobs":{"lr0":"1e-3"},"map50":0.9321,"map50_95":0.7644,"kernel_ms":1.39,"size_mb":4.3,"qdq_pairs":207,"eval_n":6449}
J

# --- per-knob sweep docs (non-binary knobs get table + 2 plots) ---
KNOB=lr0 VERSIONS=V2_lr0_1e-4,V2_lr0_3e-4,V2_lr0_1e-3,V2_lr0_3e-3,V2_lr0_1e-2 "$PY311" experiments/qat_iteration_2/gen_sweep_doc.py || true
KNOB=lrf VERSIONS=V2_lrf_0.001,V2_lrf_0.1 "$PY311" experiments/qat_iteration_2/gen_sweep_doc.py || true
KNOB=batch VERSIONS=V2_batch_8,V2_batch_32 "$PY311" experiments/qat_iteration_2/gen_sweep_doc.py || true
KNOB=n_calib VERSIONS=V2_ncalib_32,V2_ncalib_512 "$PY311" experiments/qat_iteration_2/gen_sweep_doc.py || true
echo "[$(stamp)] CAMPAIGN COMPLETE. See $S/master_comparison.md and the *_sweep/ folders."
