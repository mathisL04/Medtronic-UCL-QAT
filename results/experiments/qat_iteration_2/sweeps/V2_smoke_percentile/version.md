# V2_smoke_percentile

**Active knob(s):** EPOCHS=1, CALIB_METHOD=percentile — everything else at baseline.

## Hyperparameter status (all default except the tested knob)
| knob | this run | baseline | changed? |
|---|---|---|---|
| LR0 | 1e-3 | 1e-3 | — |
| EPOCHS | 1 | 50 | ✅ CHANGED |
| PATIENCE | 10 | 10 | — |
| WEIGHT_AXIS | 0 | 0 | — |
| DISABLE_LAYERS | (none) | (none) | — |
| CALIB_METHOD | percentile | max | ✅ CHANGED |
| CALIB_PERCENTILE | 99.99 | 99.99 | — |
| N_CALIB | 128 | 128 | — |
| BATCH | 16 | 16 | — |

## Metrics (same method as baseline: full-val pycocotools conf 0.001 · CUDA-event kernel)
| metric | this run | V2_baseline (=V6) | Δ |
|---|---|---|---|
| mAP50 | 0.8541 | 0.9321 | -0.0780 |
| mAP50-95 | 0.6403 | 0.7644 | -0.1241 |
| kernel ms | not measured (non-exclusive) | 1.390 | — |
| size MB | 4.36 | — | — |
| Q/DQ pairs | 207 | 207 | — |
| eval set | val100 (100) | full (6449) | — |
