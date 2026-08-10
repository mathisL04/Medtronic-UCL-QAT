# V2_disable_attention

**Active knob(s):** DISABLE_LAYERS=*model.10*,*model.22* — everything else at baseline.

## Hyperparameter status (all default except the tested knob)
| knob | this run | baseline | changed? |
|---|---|---|---|
| LR0 | 1e-3 | 1e-3 | — |
| LRF | 0.01 | 0.01 | — |
| EPOCHS | 50 | 50 | — |
| PATIENCE | 10 | 10 | — |
| WEIGHT_AXIS | 0 | 0 | — |
| DISABLE_LAYERS | *model.10*,*model.22* | (none) | ✅ CHANGED |
| CALIB_METHOD | max | max | — |
| CALIB_PERCENTILE | 99.99 | 99.99 | — |
| N_CALIB | 128 | 128 | — |
| BATCH | 16 | 16 | — |

## Metrics (same method as baseline: full-val pycocotools conf 0.001 · CUDA-event kernel)
| metric | this run | V2_baseline (=V6) | Δ |
|---|---|---|---|
| mAP50 | 0.9335 | 0.9321 | +0.0014 |
| mAP50-95 | 0.7580 | 0.7644 | -0.0064 |
| kernel ms | 1.381 | 1.390 | -0.0090 |
| size MB | 6.44 | — | — |
| Q/DQ pairs | 175 | 207 | — |
| eval set | full (6449) | full (6449) | — |
