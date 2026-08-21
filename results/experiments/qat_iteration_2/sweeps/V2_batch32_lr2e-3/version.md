# V2_batch32_lr2e-3

**Active knob(s):** LR0=2e-3, BATCH=32 — everything else at baseline.

## Hyperparameter status (all default except the tested knob)
| knob | this run | baseline | changed? |
|---|---|---|---|
| LR0 | 2e-3 | 1e-3 | ✅ CHANGED |
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
| mAP50 | 0.9434 | 0.9321 | +0.0113 |
| mAP50-95 | 0.7799 | 0.7644 | +0.0155 |
| kernel ms | 1.397 | 1.390 | +0.0070 |
| size MB | 4.29 | — | — |
| Q/DQ pairs | 207 | 207 | — |
| eval set | full (6449) | full (6449) | — |
