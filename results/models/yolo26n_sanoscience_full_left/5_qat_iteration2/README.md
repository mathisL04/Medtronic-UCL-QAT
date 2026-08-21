# Stage 7: QAT Iteration 2 — the deployment candidate

**This is the best model in the project.** Best accuracy of any precision, at
competitive latency, with no retraining beyond the sweep itself.

```text
QAT batch_32, rebuilt FP16 + opt5
  mAP50-95   0.7797     (vs FP32 0.7747, FP16 0.7748, PTQ 0.7571, QAT V6 0.7644)
  latency    1.200 ms   (PTQ floor is 1.082 ms; FP32 2.019 ms)
  size       4.79 MB disk / 9.7 MB device memory
```

It beats FP32 on accuracy while running 1.7x faster, which none of the earlier
stages managed.

## Why the artifacts are not duplicated here

Unlike stages 0–4, this stage's artifacts stay in the experiment that produced them:

```text
results/experiments/qat_iteration_2/sweeps/V2_batch_32/           the training run
    args.yaml  run_config.json  results.csv  metrics.json
    qat_provenance.json
    best_qat.onnx.provenance.json
    best_qat_int8.engine.provenance.json
    latency.json  _latency.log  version.md

results/experiments/qat_iteration_2/rebuild_fp16_opt5/            the rebuild that made it fast
    BEFORE_AFTER.md                 all 11 sweep models, re-timed in one idle-GPU session
    V2_batch_32.engine.map_full.json
    rebuild_all.py  rebuild_latency.json
```

Copying them would create a second copy free to drift from the run that generated it,
and the sweep's value is that all 11 runs sit side by side under one comparison. The
provenance sidecars carry the sha256 of every artifact, so the canonical location is
unambiguous.

## What made it fast

The engine was **not retrained**. Iteration 2 found that QAT's apparent ~0.3 ms
latency penalty against PTQ was roughly 60% a build artifact rather than a property
of the model: rebuilding with FP16 fallback and builder optimisation level 5 recovered
it to ~0.12 ms above the PTQ floor. See `docs/06_qat.md`, section
"The recovery — a smarter build, no retraining".

## Reproducing

Both steps reuse the shared pipeline, no code copied:

```bash
source ~/venvs/medtronic-trt/bin/activate
# rebuild from the retained Q/DQ ONNX
FP16=1 OPT_LEVEL=5 DEVICE=<idle gpu> \
  ONNX_PATH=results/experiments/qat_iteration_2/sweeps/V2_batch_32/best_qat.onnx \
  ENGINE_PATH=<out>.engine \
  python scripts/tensorrt/build_tensorrt_int8_qdq.py
```

Accuracy and latency then come from `scripts/evaluate/evaluate_engine_map.py` and
`scripts/benchmark/benchmark_latency_trt.py` as everywhere else in this project.
