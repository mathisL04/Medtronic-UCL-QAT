# TensorRT FP32 Engine (V2 Baseline)

This document records the first TensorRT stage: converting the validated FP32 `best.onnx` (docs/02) into a TensorRT **FP32** engine and characterising it. This engine is the project's **V2 baseline**.

Why an FP32 engine before FP16/INT8: it isolates the two effects that a naive "PyTorch vs FP16 engine" comparison would tangle together — TensorRT's graph/kernel optimisation, and precision reduction. The FP32 engine changes **only** the runtime (PyTorch → TensorRT) at **identical precision**, so:

```text
V1  PyTorch FP32          the original baseline (docs/01)
V2  TensorRT FP32   <-     this stage: TensorRT optimisation, ZERO precision change
    TensorRT FP16          precision effect, measured against V2 (apples-to-apples)
    TensorRT INT8 / QAT    quantisation effect, measured against V2
```

Without V2, an FP16-vs-PyTorch delta mixes both effects. V2 is the clean control every downstream engine is compared against.

---

## Environment

The TensorRT stages run on **Geneva** (the same A100 as the docs/01 latency baseline), so latency numbers are directly comparable. TensorRT is not installed system-wide; it was pip-installed into a **dedicated** venv, kept separate from the training/export venv so its CUDA-13 libraries cannot disturb the `torch` (CUDA-12) used elsewhere.

```text
host:          Geneva (geneva.ee.ucl.ac.uk)
gpu:           NVIDIA A100-SXM4-80GB
driver:        595.71.05  (CUDA 13.2 capable)
venv:          ~/venvs/medtronic-trt   (dedicated, TensorRT-only)
python:        3.9.25
```

The pip TensorRT wheel does **not** ship `trtexec`, so the engine is built and driven through the **TensorRT Python API** (which also fits the repo's "flat script" convention better than a CLI call).

Install:

```bash
python3 -m venv ~/venvs/medtronic-trt
source ~/venvs/medtronic-trt/bin/activate
pip install tensorrt==11.1.0.106       # CUDA-13 build, matches the driver
pip install cuda-python onnxruntime opencv-python-headless numpy pynvml
```

---

## Overview flow

```text
best.onnx  (validated FP32, opset 17, static [1,3,640,640])
        │
        │  scripts/build_tensorrt_engine.py   (PRECISION=fp32, TF32 OFF)
        │  TensorRT parses the ONNX and autotunes kernels on the live GPU
        ▼
best_fp32.engine   (GPU-specific plan; NOT committed)
        │
        ├──► scripts/validate_engine_parity.py   engine vs ONNX on val100 -> PASS
        │                                         => inherits the ONNX accuracy
        │
        └──► scripts/benchmark_latency_trt.py     idle-gated latency on the A100
```

---

## Tooling (committed)

Three scripts, all following the repo convention (flat, `# Settings` block, env-var run-time knobs, `DEVICE` strict):

```text
scripts/build_tensorrt_engine.py    ONNX -> .engine. PRECISION knob (fp32 | fp16),
                                    TF32 control, workspace size. Builds, saves,
                                    deserialises to verify, writes a provenance sidecar.

scripts/validate_engine_parity.py   Runs the engine (TensorRT) and the ONNX
                                    (onnxruntime, CPU) on the same val100 frames and
                                    compares final detections (matched-pair coord/conf).
                                    Contains the reusable TRTEngine inference wrapper.

scripts/benchmark_latency_trt.py    Single-frame (batch=1) latency, mirroring
                                    benchmark_latency.py: idle-GPU gate, RAM preload,
                                    warmup, pooled per-stage medians, per-repeat
                                    contention snapshots.
```

`DEVICE` has no default in all three — a mis-attributed GPU is how a latency (or a build) number becomes unusable. For the build, idle also matters because TensorRT **times candidate kernels on the live GPU** to pick the fastest; a contended GPU can bias that choice.

---

## The TF32 decision (important)

On Ampere+, TensorRT runs FP32 conv/matmul in **TF32** (TensorFloat-32, a 10-bit-mantissa tensor-core mode) **by default**. TF32 is itself a mild precision reduction — which would quietly contaminate the one thing a V2 baseline exists to control for. This was caught by the parity gate:

```text
engine vs ONNX parity, val100:
  TF32 default (on):   91/100 pass,  max coord diff 1.26 px,   max conf diff 1.7e-02
  TF32 disabled:      100/100 pass,  max coord diff 1.5e-04 px, max conf diff 3.3e-05
```

The V2 engine is therefore built with **TF32 disabled** (`config.clear_flag(trt.BuilderFlag.TF32)`) so it is true FP32 — the same precision as the PyTorch/ONNX baseline. (A faster TF32 build is available via `ALLOW_TF32=1`, but it is a different precision and not the baseline.)

---

## Results

### Build

```text
engine:        models/yolo26n_sanoscience_full_left/best_fp32.engine   (13.7 MB)
tensorrt:      11.1.0.106
built on:      A100-SXM4-80GB (GPU 2, idle), TF32 disabled (true FP32)
layers:        469
input:         images   [1, 3, 640, 640]  FLOAT
output:        output0  [1, 300, 6]        FLOAT   (end-to-end, NMS in-engine)
```

The engine is **GPU-specific** (built for sm_80 / this TensorRT build) and is **gitignored** — never committed. It is rebuilt on whatever GPU deploys. The build writes a `best_fp32.engine.provenance.json` sidecar (TRT version, GPU, source-ONNX sha256, flags).

### Accuracy (parity → inherit)

The engine matches the FP32 ONNX to floating-point noise, so it inherits the accuracy validated in docs/02:

```text
parity (engine vs ONNX, val100):  100/100 PASS
  max coord diff:  1.5e-04 px
  max conf diff:   3.3e-05
inherited accuracy:  mAP50 0.9394   mAP50-95 0.7595
```

### Latency (batch=1, verified-idle A100, pooled 10x100 = 1000 samples)

```text
Stage         Median     P95      P99
Preprocess    1.640 ms   2.522    2.941
Inference     2.301 ms   2.506    2.969
Postprocess   0.024 ms   0.030    0.038
Total         3.986 ms   5.027    6.604      ->  250.9 FPS
```

### V1 vs V2 — TensorRT optimisation at zero precision cost

```text
                   V1 (PyTorch FP32)   V2 (TensorRT FP32)   change
Total (median)        8.642 ms            3.986 ms          2.17x faster
FPS                   115.7               250.9             2.17x
Core inference        6.844 ms            2.301 ms          2.97x faster
mAP50 / mAP50-95      0.934 / 0.782       0.9394 / 0.7595   unchanged
```

TensorRT's optimisation alone gives ~2.2x end-to-end (nearly 3x on core inference) with no accuracy change.

Caveats on the comparison:

```text
- Compare TOTALS, not per-stage. V2's NMS is baked into the engine (counted under
  Inference); V1 did NMS on CPU (counted under its Postprocess 0.238 ms).
- Instrumentation differs slightly: V1 used Ultralytics' internal speed counters;
  V2 uses wall-clock around each stage. This does not materially affect totals.
- Preprocess (1.64 ms, CPU) is now ~41% of the V2 total. As FP16/INT8 shrink
  inference further, CPU preprocess becomes the bottleneck.
```

---

## Versions

```text
tensorrt        11.1.0.106
cuda-python     13.0.3   (cuda-bindings 13.0.3)
onnxruntime     1.19.2   (CPU, for the parity reference)
opencv          5.0.0.93 (headless)
numpy           2.0.2
pynvml          13.0.1
python          3.9.25
```

---

## Reproduce

```bash
source ~/venvs/medtronic-trt/bin/activate

# build (true FP32, idle GPU)
PRECISION=fp32 DEVICE=2 python scripts/build_tensorrt_engine.py

# accuracy parity vs the ONNX baseline
DEVICE=2 python scripts/validate_engine_parity.py

# latency (idle-gated, 10 x 100 frames)
DEVICE=2 BENCHMARK_REPEATS=10 python scripts/benchmark_latency_trt.py
```

---

## Next stage

The FP16 engine (`PRECISION=fp16` with the same build script), validated for accuracy against V2 and benchmarked on the same idle A100. Every downstream engine — FP16, INT8/PTQ, QAT — is compared against this V2 baseline (3.986 ms, mAP50 0.9394).

Note on doc numbering: this FP32-engine stage was not in the original plan (which went ONNX → FP16 directly). The FP16 / INT8 / QAT stage docs may need renumbering to sit after this one.
