# TensorRT FP32 Engine (V2 Baseline)

Converts the validated FP32 `best.onnx` (docs/02) into a TensorRT **FP32** engine — the project's **V2 baseline**.

Why FP32 before FP16/INT8: it separates two effects a "PyTorch vs FP16" comparison would tangle together — TensorRT's kernel optimisation, and precision reduction. V2 changes only the runtime, at identical precision.

```text
V1  PyTorch FP32       docs/01 (model + accuracy), docs/07 (latency)
V2  TensorRT FP32  <-  this stage: optimisation only, ZERO precision change
V3  TensorRT FP16      docs/04: precision only, measured against V2
    INT8 / QAT         docs/05, docs/06
```

---

## Environment

```text
host:      geneva.ee.ucl.ac.uk      gpu: A100-SXM4-80GB
driver:    595.71.05                python: 3.9.25
venv:      ~/venvs/medtronic-trt    (TensorRT-only, separate from the training venv)
tensorrt:  10.16.1.11
```

The pip wheel ships no `trtexec`, so the engine is built through the TensorRT Python API.

---

## Running the scripts

**One script per job. The precision is chosen by `PRECISION` at build time; everything downstream is chosen by `ENGINE_PATH`.** Nothing else changes between FP32 and FP16 — same scripts, same source ONNX.

```bash
source ~/venvs/medtronic-trt/bin/activate
ENG=models/yolo26n_sanoscience_full_left/1_fp32/best_fp32.engine    # swap to best_fp16.engine for V3

# 1. BUILD -- PRECISION picks fp32 or fp16; writes best_<precision>.engine
PRECISION=fp32 DEVICE=0 python scripts/tensorrt/build_tensorrt_engine.py

# 2. FAITHFULNESS -- engine vs ONNX detections (smoke test, not accuracy)
ENGINE_PATH=$ENG DEVICE=0 python scripts/evaluate/validate_engine_parity.py

# 3. ACCURACY -- mAP via pycocotools. MODE=engine or MODE=onnx, same metric code
MODE=engine ENGINE_PATH=$ENG DEVICE=0 python scripts/evaluate/evaluate_engine_map.py

# 4. LATENCY -- batch=1, idle-gated, 10 repeats x 100 images
ENGINE_PATH=$ENG DEVICE=0 BENCHMARK_REPEATS=10 python scripts/benchmark/benchmark_latency_trt.py
```

The only edits needed to run the whole chain on a different precision:

```text
build      PRECISION=fp32   ->  PRECISION=fp16
2,3,4      ENG=...best_fp32.engine  ->  ENG=...best_fp16.engine
```

Useful knobs (all have defaults; `DEVICE` has none and is required):

```text
DEVICE                 required everywhere -- no default, refuses to guess a GPU
PRECISION              fp32 | fp16                          (build)
ALLOW_TF32=1           keep TF32 on -- a DIFFERENT precision (build, see below)
MODE                   engine | onnx                        (evaluate_engine_map)
CONF                   0.25 parity/latency, 0.001 for mAP   (do not mix these up)
BENCHMARK_REPEATS      10                                   (latency)
GATE_ALLOW_IDLE_MIB    0 = strict; tolerate dormant contexts above 0 (latency)
```

---

## Where precision is actually set

All of it is these six lines in `scripts/tensorrt/build_tensorrt_engine.py`:

```python
if PRECISION == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)      # enable half precision

if not ALLOW_TF32:
    config.clear_flag(trt.BuilderFlag.TF32)    # unconditional: every precision

fp16_flag    = config.get_flag(trt.BuilderFlag.FP16)   # read back, not assumed
tf32_enabled = config.get_flag(trt.BuilderFlag.TF32)
```

Two independent flags, both recorded in the provenance sidecar:

```text
             FP16 flag   TF32     meaning
V2 FP32         off       off     true FP32 everywhere
V3 FP16         on        off     FP16 where TensorRT picks it,
                                  true FP32 on the layers it does not
```

### The TF32 decision

TF32 is **not** "the FP32 setting" — it is a reduced-precision tensor-core format that TensorRT substitutes for FP32 maths **by default** on Ampere+:

```text
          exponent   mantissa
FP32          8         23
TF32          8         10     <- FP32's range, FP16's mantissa
FP16          5         10
```

Leaving it on would make the "FP32 baseline" silently reduced-precision. Measured on this model:

```text
FP32 engine, TF32 on:   91/100 parity pass,  max conf diff 1.7e-02
FP32 engine, TF32 off: 100/100 parity pass,  max conf diff 3.3e-05
```

So V2 is built **TF32 off**. The flag is cleared for *every* precision, not just fp32: TF32 governs layers that run in FP32, which in a mixed-precision FP16 engine includes its FP32 fallback layers — leaving it on there would have made the V2-vs-V3 delta measure two changes at once. `ALLOW_TF32=1` gives the faster build, but it is a different precision and not the baseline.

---

## Results

### Build

```text
engine:    best_fp32.engine  (12.8 MB)   sha256 5909ffcb37249c5e...
built on:  A100 (GPU 0), TF32 disabled, 469 layers
input:     images   [1,3,640,640] FLOAT
output:    output0  [1,300,6]     FLOAT   (NMS in-engine, end-to-end)
```

The engine is GPU-specific and **gitignored**; the `.provenance.json` / `.parity.json` / `.map.json` sidecars are committed. TensorRT autotunes by timing kernels on the live GPU, so builds are **not deterministic** — the same ONNX and flags can produce a different engine and size. The sha256 in provenance is the stable identifier, not the byte count.

### Accuracy

```text
parity vs ONNX (conf 0.25):  100/100 PASS,  234/234 detections
  max coord diff  1.831e-04 px      median  7.629e-06 px
  max conf diff   3.809e-05         median  1.192e-07

mAP (pycocotools, conf 0.001):
  ONNX          0.9350 / 0.7572
  FP32 engine   0.9350 / 0.7572     identical
```

Parity at this level means the engine is numerically the ONNX; the mAP measurement confirms it independently. The gap to docs/02 (0.9394) is the metric implementation — Ultralytics vs pycocotools on the same ONNX — not an accuracy change. All precision comparisons stay inside the pycocotools column.

### Latency (batch=1, 10x100 = 1000 samples, GPU 0)

```text
Stage          Mean      Std   Median      Min      P95      P99      Max
Preprocess    1.517    0.511    1.372    0.972    2.442    2.793    9.866
Inference     2.350    0.140    2.318    2.298    2.502    2.705    5.505
Postprocess   0.023    0.014    0.021    0.017    0.035    0.058    0.307
Total         3.889    0.547    3.736    3.299    4.820    5.452   12.449
                                                          ->  267.7 FPS
```

Run with `GATE_ALLOW_IDLE_MIB=600`: **not exclusive**, two dormant contexts (496 MiB, 0% util) tolerated, `exclusive_gpu: false` in the record. Evidence they did not interfere — inference landed within 0.7% of an earlier exclusive run with *lower* variance (std 0.140 vs 0.253).

**Compare Inference across runs, not Total.** Total carries CPU-side preprocess variance unrelated to the engine.

### V1 vs V2

```text
                   V1 PyTorch   V2 TensorRT   change
Total (median)      8.642 ms     3.736 ms     2.31x faster
Core inference      6.844 ms     2.318 ms     2.95x faster
mAP50 / mAP50-95    0.934/0.782  0.9350/0.7572  unchanged
```

Caveats: V2's NMS is in-engine (counted under Inference), V1's was CPU-side; V1 used Ultralytics' counters, V2 wall-clock. V1 mAP is full-set Ultralytics and V2 is val100 pycocotools — "unchanged" means neither stage cost accuracy, not a like-for-like metric.

---

## The gate

`benchmark_latency_trt.py` refuses to run if another compute process is on the target GPU — a mis-attributed or contended GPU is how a latency number becomes unusable. `GATE_ALLOW_IDLE_MIB` (default **0** = strict) sets a per-process memory ceiling below which a dormant foreign process is tolerated, for shared servers where the strict rule is otherwise unsatisfiable:

```text
- memory-based only; GATE_UTIL_THRESHOLD still gates utilisation
- unknown-memory processes always block, never assumed small
- per-repeat contention snapshots unchanged, so a process waking mid-run shows up
- provenance records exclusive_gpu + tolerated PIDs, so a non-exclusive run
  can never be silently compared against an exclusive one
```

---

## Next stage

FP16 (V3) — docs/04. Every downstream engine is compared against this V2 baseline: **mAP50 0.9350 / mAP50-95 0.7572, 3.736 ms total / 2.318 ms inference.**
