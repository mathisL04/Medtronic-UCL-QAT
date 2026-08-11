# PTQ baseline — INT8 + FP16 fallback (max-quant)

Fresh PTQ baseline for QAT Iteration 2: FP32 → INT8 with **FP16 fallback** (INT8 where TensorRT
judges it profitable; every other layer in FP16, not FP32). Calibrated on the deterministic seed-42
500-frame set (disjoint from val).

## Reproduce (reused Phase-1 scripts only)

```bash
# TensorRT venv
PY=/home/zcemml1/venvs/medtronic-trt/bin/python

# 1. calibration set (deterministic, already present)
#    python scripts/data/make_calib_set.py            # seed 42, 500 frames

# 2. build INT8+FP16 engine
set -a; source experiments/qat_iteration_2/configs/ptq_int8fp16.env; set +a
DEVICE=<idle> $PY scripts/tensorrt/build_tensorrt_engine.py
#    -> writes models/.../baseline/best_int8.engine; relocated here as best_int8_fp16.engine

# 3. accuracy (full val, conf 0.001)
MODE=engine EVAL_SET=full DEVICE=<idle> \
  ENGINE_PATH=experiments/qat_iteration_2/ptq_baseline/best_int8_fp16.engine \
  $PY scripts/evaluate/evaluate_engine_map.py

# 4. latency (idle-gated CUDA-event kernel)
DEVICE=<idle> ENGINE_PATH=experiments/qat_iteration_2/ptq_baseline/best_int8_fp16.engine \
  $PY scripts/benchmark/benchmark_latency_trt.py
```

## Files

| File | What |
|---|---|
| `best_int8_fp16.engine` | the engine (gitignored) |
| `best_int8_fp16.engine.provenance.json` | build flags, INT8/FP16 layer counts, size |
| `best_int8_fp16.engine.map_full.json` | accuracy: full 6,449 val, conf 0.001 |
| `best_int8_fp16.latency.json` | kernel latency (idle-gated) + throughput |
| `comparison.md` | side-by-side vs FP16 / FP32 / V4 (same-session) |

Results filled in by the run — see `comparison.md`.
