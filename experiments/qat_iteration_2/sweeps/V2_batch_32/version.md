# V2_batch_32

**Active knob(s):** BATCH=32 — everything else at baseline.

## Hyperparameter status (all default except the tested knob)
| knob | this run | baseline | changed? |
|---|---|---|---|
| LR0 | 1e-3 | 1e-3 | — |
| LRF | 0.01 | 0.01 | — |
| EPOCHS | 50 | 50 | — |
| PATIENCE | 10 | 10 | — |
| WEIGHT_AXIS | 0 | 0 | — |
| DISABLE_LAYERS | (none) | (none) | — |
| CALIB_METHOD | max | max | — |
| CALIB_PERCENTILE | 99.99 | 99.99 | — |
| N_CALIB | 128 | 128 | — |
| BATCH | 32 | 16 | ✅ CHANGED |

## Metrics (same method as baseline: full-val pycocotools conf 0.001 · CUDA-event kernel)
| metric | this run | V2_baseline (=V6) | Δ |
|---|---|---|---|
| mAP50 | 0.9437 | 0.9321 | +0.0116 |
| mAP50-95 | 0.7800 | 0.7644 | +0.0156 |
| kernel ms | not measured (non-exclusive) | 1.390 | — |
| size MB | 4.94 | — | — |
| Q/DQ pairs | 207 | 207 | — |
| eval set | full (6449) | full (6449) | — |
