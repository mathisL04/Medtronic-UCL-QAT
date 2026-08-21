# V2_lr0_3e-3

**Active knob(s):** LR0=3e-3 — everything else at baseline.

## Hyperparameter status (all default except the tested knob)
| knob | this run | baseline | changed? |
|---|---|---|---|
| LR0 | 3e-3 | 1e-3 | ✅ CHANGED |
| EPOCHS | 50 | 50 | — |
| PATIENCE | 10 | 10 | — |
| WEIGHT_AXIS | 0 | 0 | — |
| DISABLE_LAYERS | (none) | (none) | — |
| CALIB_METHOD | max | max | — |
| CALIB_PERCENTILE | 99.99 | 99.99 | — |
| N_CALIB | 128 | 128 | — |
| BATCH | 16 | 16 | — |

## Metrics (same method as baseline: full-val pycocotools conf 0.001 · CUDA-event kernel)
| metric | this run | V2_baseline (=V6) | Δ |
|---|---|---|---|
| mAP50 | 0.9320 | 0.9321 | -0.0001 |
| mAP50-95 | 0.7644 | 0.7644 | +0.0000 |
| kernel ms | not measured (non-exclusive) | 1.390 | — |
| size MB | 4.34 | — | — |
| Q/DQ pairs | 207 | 207 | — |
| eval set | full (6449) | full (6449) | — |
