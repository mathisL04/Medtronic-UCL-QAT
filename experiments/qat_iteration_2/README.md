# QAT Iteration 2 — hyperparameter dependency study

**Re-iteration of the QAT work, focused on hyperparameter dependencies and quantifying how they
affect the model's accuracy and latency.**

This is an **additive** experimental phase. It does **not** modify Phase 1 — the validated reference
pipeline (`docs/00`–`docs/08`, `models/`, `scripts/`) stays exactly as it is. This folder holds
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
| `sweeps/` | The 13 OFAT hyperparameter runs + `master_comparison.md`. |
| `rebuild_fp16_opt5/` | All 11 sweep engines rebuilt at FP16+opt5 and **re-timed in one exclusive idle-GPU session** -- the comparable latency table. Start at `BEFORE_AFTER.md`. |
| `engine_verification/` | Per-layer precision breakdown for FP32 / FP16 / PTQ / V4 / V6, from `verify_engines.py`. |
| `profiling_exports/` | Per-kernel and per-region profiles behind `notebooks/qat_vs_ptq_kernel_profiling.ipynb`. |

## Method references

`01_qat_framework_deep_dive.md`, `05_library_stack_explained.md`,
`06_qdq_placement_deep_dive.md` and `qat_method_reference.md` explain the method itself and
are the intended starting point for anyone picking this up.

The pre-run design documents (`02_monitoring_scheme.md`, `03_sweep_action_plan.md`,
`04_per_sweep_monitoring.md`) recorded intent rather than findings -- and the
`monitoring/qat_monitor.py` that `02` specified was never built. They live on the
**`archive/legacy-scripts`** branch; nothing on `main` depends on them.

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
