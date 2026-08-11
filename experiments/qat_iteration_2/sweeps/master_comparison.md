# QAT Iteration 2 — master comparison

All versions vs the V6 QAT reference (defaults). Same metric methods as the locked PTQ baseline.

| version | active knob | mAP50 | mAP50-95 | Δ50-95 | kernel ms | size MB | Q/DQ |
|---|---|---|---|---|---|---|---|
| V6_reference (defaults) | — (anchor) | 0.9321 | 0.7644 | 0.0000 | 1.39 | 4.3 | 207 |
| V2_lr0_3e-4 | LR0=3e-4 | 0.9320 | 0.7644 | +0.0000 | 1.382 | 4.31 | 207 |
| V2_lr0_3e-3 | LR0=3e-3 | 0.9320 | 0.7644 | +0.0000 | — | 4.34 | 207 |
| V2_lr0_1e-4 | LR0=1e-4 | 0.9320 | 0.7644 | +0.0000 | 1.381 | 4.33 | 207 |
| V2_batch_8 | BATCH=8 | 0.9234 | 0.7318 | -0.0326 | — | 4.93 | 207 |
| V2_lr0_1e-2 | LR0=1e-2 | 0.9322 | 0.7647 | +0.0003 | 1.560 | 4.35 | 207 |
| V2_lrf_0.001 | LRF=0.001 | 0.9298 | 0.7262 | -0.0382 | — | 4.38 | 207 |
| V2_lrf_0.1 | LRF=0.1 | 0.9347 | 0.7671 | +0.0027 | — | 4.47 | 207 |
| V2_ncalib_32 | N_CALIB=32 | 0.9414 | 0.7609 | -0.0035 | 1.518 | 4.34 | 207 |
| V2_batch_32 | BATCH=32 | 0.9437 | 0.7800 | +0.0156 | — | 4.94 | 207 |
| V2_disable_attention | DISABLE_LAYERS=*model.10*,*model.22* | 0.9335 | 0.7580 | -0.0064 | 1.381 | 6.44 | 175 |
| V2_ncalib_512 | N_CALIB=512 | 0.9361 | 0.7608 | -0.0036 | — | 4.31 | 207 |
