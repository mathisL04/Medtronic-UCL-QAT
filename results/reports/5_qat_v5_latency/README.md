# V5 latency — final distributions (compute level, CUDA-event)

Same model (`results/models/yolo26n_sanoscience_full_left/qat/v5_10ep/qat_modelopt_state_best.pt`, 10-epoch QAT best
state -- V5 was superseded by V6 and now lives on the `archive/legacy-scripts` branch),
batch=1, val100, N=1000 (10x100), **GPU 2, idle/exclusive for the whole run**
(`exclusive_gpu: true`). Both measured at the **pure-GPU compute level** (CUDA
events, no H2D/D2H), so they are comparable as compute cost.

Raw data: the two `*.provenance.json` in this directory.

```text
                    PyTorch fake-quant forward     TensorRT INT8 kernel
                    (Q/DQ in FP32, eager,           (real INT8 engine,
                     NOT deployment)                 deployable)
median                 80.051 ms                      1.3862 ms
mean                   80.105 ms                      1.3971 ms
std                     0.480 ms  (0.6%)              0.0330 ms  (2.4%)
min  (best case)       79.300 ms                      1.3788 ms
max  (worst case)      86.034 ms  (+7.5%)             1.6686 ms  (+20%)
p95                    80.741 ms                      1.4526 ms
p99                    81.807 ms                      1.5357 ms
compute ratio          ~57.7x the INT8 kernel         1x
```

Notes:
- **Labels are honest.** PyTorch = fake-quant *simulation* forward (not real INT8,
  not deployment). TensorRT = real INT8 kernel (deployable).
- **Spread.** In absolute terms PyTorch swings more (+-0.48 ms) but it is an 80 ms
  number (0.6% std); the TRT kernel is tighter absolutely (+-0.033 ms) but noisier
  as a fraction (2.4% std, worst case +20%) since at ~1.4 ms small jitter is a
  larger percentage. Both means ~= medians -> near-symmetric, no heavy skew.
- Scripts: `scripts/benchmark_latency_pytorch_qat.py` (on the `archive/legacy-scripts` branch) (PyTorch forward CUDA-event),
  `scripts/benchmark/benchmark_latency_trt.py` (TRT kernel CUDA-event).
