# QAT Iteration 2 — hyperparameter dependency study

**Re-iteration of the QAT work, focused on hyperparameter dependencies and quantifying how they
affect the model's accuracy and latency.**

This is an **additive** experimental phase. It does **not** modify Phase 1 — the validated reference
pipeline (`docs/01`–`docs/07`, `models/`, `scripts/`) stays exactly as it is. This folder holds
**only** the new phase's configs, results, and analysis. It **reuses the existing pipeline scripts**
in `scripts/` — no framework code is copied or duplicated here.

## Goal

1. **Establish a clean, maximally-quantized PTQ baseline** (INT8 + FP16 fallback) from the FP32
   model — the deployment-style "max-quant" engine where non-INT8 layers run FP16, not FP32.
2. **Then** (next phase, not yet) systematically vary QAT hyperparameters and quantify their effect
   on accuracy / latency against this baseline.

## Phase layout

| Path | What it holds |
|---|---|
| `ptq_baseline/` | The INT8+FP16 PTQ baseline: engine (gitignored) + committed provenance / accuracy / latency sidecars + comparison. |
| `configs/` | Env-var config sets (the "knobs") that drive the reused Phase-1 scripts. No script copies. |
| `sweeps/` | *(next phase)* QAT hyperparameter runs. |
| `analysis/` | *(next phase)* Plots + quantified hyperparameter dependencies. |

## Reused Phase-1 scripts (not duplicated)

| Step | Script | Invocation |
|---|---|---|
| Calibration set | `scripts/data/make_calib_set.py` | deterministic, seed 42, 500 train frames |
| Build engine | `scripts/tensorrt/build_tensorrt_engine.py` | `PRECISION=int8 INT8_ALLOW_FP16=1` |
| Accuracy (mAP) | `scripts/evaluate/evaluate_engine_map.py` | `MODE=engine EVAL_SET=full` (conf 0.001) |
| Latency (kernel) | `scripts/benchmark/benchmark_latency_trt.py` | idle-gated CUDA-event kernel timing |

## References (Phase 1, full val, conf 0.001)

```text
FP32 engine  mAP50-95 0.7747
FP16 engine  mAP50-95 0.7748
INT8 PTQ V4  mAP50-95 0.7571   (INT8 + FP32 fallback)
QAT V6       mAP50-95 0.7644
```
