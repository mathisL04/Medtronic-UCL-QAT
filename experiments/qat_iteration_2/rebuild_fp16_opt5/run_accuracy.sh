#!/usr/bin/env bash
# Full-val accuracy for every rebuilt FP16+opt5 engine, split across two idle GPUs.
set -u
REPO=/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT
DIR="$REPO/experiments/qat_iteration_2/rebuild_fp16_opt5"
PY=/home/zcemml1/venvs/medtronic-trt/bin/python
GPUS=(1 3)   # idle GPUs
RUNS=(V2_lr0_1e-4 V2_lr0_3e-4 V2_lr0_3e-3 V2_lr0_1e-2 V2_lrf_0.001 V2_lrf_0.1 \
      V2_batch_8 V2_batch_32 V2_ncalib_32 V2_ncalib_512 V2_disable_attention)
cd "$REPO" || exit 1

i=0
for r in "${RUNS[@]}"; do
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  eng="$DIR/$r.engine"
  echo "[$(date +%H:%M:%S)] eval $r on GPU $g"
  MODE=engine EVAL_SET=full DEVICE=$g ENGINE_PATH="$eng" \
    nohup "$PY" -u scripts/evaluate/evaluate_engine_map.py > "$DIR/${r}_acc.log" 2>&1 &
  i=$((i+1))
  # launch 2 at a time (one per GPU), then wait for this pair before next
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "[$(date +%H:%M:%S)] ALL ACCURACY DONE"
touch "$DIR/_accuracy_done"
