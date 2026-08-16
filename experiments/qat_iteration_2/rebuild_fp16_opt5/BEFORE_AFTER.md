# FP16+opt5 rebuild — FINAL comparison (no retraining)

Latency: all 11 re-timed in ONE exclusive idle-GPU3 session (0% util, no foreign proc),
300 iters after 50 warmup, CUDA-event compute-only. median spread 0.038 ms.
Accuracy: full-val 6449-img pycocotools conf 0.001 (deterministic). disk/dev-mem: static engine props.

PTQ INT8 floor: ~1.082 ms @ 0.7564.

| model | knob | INT8 layers | old mAP | new mAP | old ms | new ms (clean) | disk MB | dev-mem MB |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| V2_batch_32 | batch=32 | 143 | 0.7801 | 0.7797 | 1.386 | 1.200 | 4.79 | 9.7 |
| V2_lrf_0.1 | lrf=0.1 | 143 | 0.7671 | 0.7671 | — | 1.226 | 4.80 | 9.9 |
| V2_lr0_1e-2 | lr0=1e-2 | 143 | 0.7647 | 0.7641 | 1.560 | 1.203 | 4.81 | 9.8 |
| V2_lr0_3e-3 | lr0=3e-3 | 144 | 0.7644 | 0.7643 | — | 1.209 | 4.78 | 9.9 |
| V2_lr0_1e-4 | lr0=1e-4 | 143 | 0.7644 | 0.7640 | 1.381 | 1.214 | 4.86 | 9.8 |
| V2_lr0_3e-4 | lr0=3e-4 | 143 | 0.7644 | 0.7641 | 1.382 | 1.212 | 4.78 | 9.7 |
| V2_ncalib_32 | n_calib=32 | 144 | 0.7609 | 0.7610 | 1.518 | 1.207 | 4.81 | 9.8 |
| V2_ncalib_512 | n_calib=512 | 144 | 0.7608 | 0.7614 | — | 1.238 | 4.79 | 9.9 |
| V2_disable_attention | disable attn | 117 | 0.7580 | 0.7588 | 1.381 | 1.226 | 5.55 | 9.9 |
| V2_batch_8 | batch=8 | 143 | 0.7318 | 0.7305 | 1.391 | 1.218 | 4.79 | 9.9 |
| V2_lrf_0.001 | lrf=0.001 | 143 | 0.7262 | 0.7259 | — | 1.219 | 4.85 | 9.7 |
