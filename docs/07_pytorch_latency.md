# PyTorch-side Latency Study (raw models, NO TensorRT)

Measures the **eager PyTorch models themselves** — before any ONNX/TensorRT
conversion — to quantify the pre-conversion cost. This is deliberately **separate
from the TensorRT engine comparison** (docs/03–06): the point is what raw PyTorch
costs, so the TensorRT speedup can be attributed to the conversion, not the
precision.

Script: `scripts/benchmark_latency_pytorch.py` (`MODEL_MODE={fp32|fp16|qat}`).
Sidecars: `reports/pytorch_latency/*.provenance.json`.

## Method

```text
models:   fp32 = best.pt (V1) | fp16 = best.pt with model.half() | qat = fake-quant (qat_v5)
column A: pure-GPU compute -- CUDA events around the forward pass only (trtexec-method)
column B: Ultralytics pipeline -- model.predict() speed counters (preprocess+inference+postprocess)
protocol: val100, batch=1, 10x100 = 1000 samples, idle-gated exclusive, MEDIAN
run:      one session, GPU 2 (A100-SXM4-80GB), torch 2.7.0+cu128, seed 42
```

## Results (fresh, one session)

```text
model                       A: pure-GPU forward   B: Ultralytics pipeline (pre+inf+post)
PyTorch FP32                    12.085 ms            9.941 ms   (inf 7.372)
PyTorch FP16 (clean half)       13.802 ms            9.996 ms   (inf 7.823)
PyTorch QAT (fake-quant sim)    80.563 ms           58.575 ms   (inf 56.446)
```

## Finding 1 -- `.half()` gives NO speedup in raw PyTorch

FP16 is **not faster** than FP32 in eager PyTorch — it is slightly **slower**:

```text
             A: pure-GPU forward      B: pipeline
FP32              12.085 ms            9.941 ms
FP16              13.802 ms            9.996 ms   <- FP16 slower / equal, NOT faster
```

On this tiny launch-bound model each op is dominated by kernel-launch overhead, so
FP16's arithmetic advantage does nothing while dtype handling adds a little.
**The FP16 win is a TensorRT/runtime thing, not a precision thing** — `model.half()`
alone buys nothing.

## Finding 2 -- raw PyTorch vs the TensorRT engine: 6–12x, and FP16 WIDENS it

```text
             raw PyTorch (A, unfused fwd)   TensorRT engine (kernel)   ratio
FP32               12.085 ms                    2.019 ms               6.0x
FP16               13.802 ms                    1.137 ms              12.1x   <- gap widens
```

FP32 is ~6x, FP16 ~12x. Because PyTorch gains nothing from FP16 but the engine gains
a lot, **converting to TensorRT is where the precision speedup actually appears.**
This is the quantitative justification for the whole ONNX→TensorRT pipeline.

## Caveats (all apply to the table above)

```text
- EAGER-MODE: per-op launch overhead -- these are NOT clean engine numbers like TensorRT's.
- QAT row = FAKE-QUANT SIMULATION (Q/DQ done in FP32), NOT real 8-bit. It is slow because it
  SIMULATES quantization (~6-8x FP32), not because "INT8 is slow". Do not read it as INT8.
- FP16 is CLEAN (pure model.half(), no autocast / FP32 islands were needed -- verified). Honest
  limit: PyTorch may still internally upcast individual ops (e.g. softmax); not audited per-op.
- Column A is the UNFUSED eager forward; Ultralytics predict (column B) FUSES Conv+BN, which is
  why B-inference < A. That gap is FUSION, not conversion. Column A is kept unfused for ALL three
  so the column is internally consistent -- the fake-quant model cannot be fused the same way.
- Raw PyTorch, NO conversion -- this is the pre-TensorRT cost, on purpose.
```

## Week-3 reference (NOT mixed into the table above)

An earlier FP32 pipeline number of **8.642 ms** exists from a prior session
(`benchmark_latency.py`, Ultralytics predict counters, val100 — the V1 baseline
figure quoted in docs/03). It is kept only as a historical reference and is **not**
combined with the same-session table above: mixing one old-session row with fresh
ones reintroduces a cross-session seam. The fresh FP32-B here (9.941 ms) is the
number to use for internal comparison.
