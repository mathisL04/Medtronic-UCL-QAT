# TensorRT FP16 Engine (V3)

This document records the FP16 stage: building a TensorRT **FP16** engine from the same validated `best.onnx` (docs/02) and measuring what half precision costs in accuracy and buys in speed, **against the V2 FP32 engine** (docs/03) rather than against PyTorch.

```text
V1  PyTorch FP32         docs/01
V2  TensorRT FP32        docs/03 -- runtime change only, zero precision change
V3  TensorRT FP16   <-   this stage: precision change only, measured against V2
```

The engine is built with the **same script and the same source ONNX** as V2. Precision is a run-time knob (`PRECISION=fp16`), not a different program.

---

## Controlling the comparison

The requirement for this stage was that the FP16 and FP32 engines differ in **precision and nothing else**. Both provenance sidecars were diffed field by field to confirm it:

```text
field             FP32 (V2)               FP16 (V3)               same?
precision         fp32                    fp16                    NO   (intended)
fp16_flag         False                   True                    NO   (intended)
tf32_enabled      False                   False                   yes
tensorrt_version  10.16.1.11              10.16.1.11              yes
workspace_gib     8.0                     8.0                     yes
num_layers        469                     469                     yes
gpu_name          A100-SXM4-80GB          A100-SXM4-80GB          yes
device_index      0                       0                       yes
host              geneva.ee.ucl.ac.uk     geneva.ee.ucl.ac.uk     yes
source_onnx sha   ac188a00...             ac188a00...             yes
io_tensors        identical               identical               yes
```

`precision` and `fp16_flag` are the same knob, so precision is the sole difference. No optimisation profile exists on either side — the ONNX is static `[1,3,640,640]`, so that variable is absent by construction.

### The TF32 trap

Getting `tf32_enabled` to match required a fix. The build script originally cleared the TF32 builder flag only on the fp32 path:

```python
if PRECISION == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)
elif PRECISION == "fp32" and not ALLOW_TF32:     # <-- elif: unreachable for fp16
    config.clear_flag(trt.BuilderFlag.TF32)
```

TensorRT's TF32 default is **on** (verified directly: `config.get_flag(trt.BuilderFlag.TF32)` returns `True` on a fresh config). So the first FP16 engine was built with TF32 **enabled** while V2 had it **disabled**.

That matters because FP16 in TensorRT is *mixed* precision: most layers run FP16, numerically sensitive layers fall back to FP32, and TF32 governs how **those fallback layers** compute. A V2-vs-V3 delta would then have measured two changes — FP16 on the fast path and TF32 on the slow path. Not hypothetical: docs/03 measured TF32 alone breaking parity on 9% of frames.

The flag is now cleared for **every** precision unless `ALLOW_TF32=1`, and provenance records the flags **read back off the builder config** rather than tracked in local variables — so the record states what the builder was actually configured with, not what the script assumed it asked for.

---

## Results

### Build

```text
engine:        models/yolo26n_sanoscience_full_left/best_fp16.engine   (7.0 MB)
sha256:        08ed6b0899dbc650...
tensorrt:      10.16.1.11
built on:      A100-SXM4-80GB (GPU 0), TF32 disabled
layers:        469
input:         images   [1, 3, 640, 640]  FLOAT
output:        output0  [1, 300, 6]        FLOAT   (end-to-end, NMS in-engine)
size vs V2:    7.0 MB  vs  12.8 MB   (0.55x)
```

I/O stays FP32 in both engines, so the inference wrapper is unchanged across precisions — only the internal compute precision differs.

The engine is **GPU-specific** and **gitignored**; the `.provenance.json`, `.parity.json` and `.map.json` sidecars are committed.

### Parity vs the FP32 ONNX — FAIL, and that is the correct result

```text
parity (engine vs FP32 ONNX, val100, conf 0.25):  77/100  FAIL
  max coord diff:  1.426 px
  max conf diff:   1.127e-01     (tolerance conf_atol 5e-3)
failure modes:
  ~21 frames  coord/conf over tolerance
    2 frames  box count changed (2<->3)
```

**This FAIL is expected physics, not a defect, and the tolerances were deliberately left unchanged.** They were calibrated for FP32, where the engine agrees with the ONNX to ~1e-04 px — a bit-exactness bar that a 10-bit-mantissa format cannot meet. Loosening `conf_atol` after seeing the result would have manufactured a PASS that meant nothing.

Reading the numbers instead of the verdict:

```text
- Boxes stay sub-2-pixel. Spatial agreement is intact.
- Confidence is what FP16 perturbs (up to 0.113).
- The box-count changes are detections crossing the conf 0.25 filter, an
  artefact of thresholding a perturbed score -- not a missed or invented object.
```

#### Why the per-frame rate overstates it

A frame fails if **any** detection in it exceeds tolerance, so the frame-level
rate is a max over a *varying* number of detections — it partly measures how many
tools happen to be in shot rather than how far the engine drifted:

```text
fail rate by detections in frame:
  1 box:    1/24 fail    4%
  2 boxes:  9/36 fail   25%
  3 boxes:  4/25 fail   16%
  4 boxes:  6/12 fail   50%
  5 boxes:  3/ 3 fail  100%      (small buckets: 12 and 3 frames)
```

A 5-detection frame gets five chances to breach the line; a 1-detection frame gets
one. The parity script therefore also reports the pooled per-detection distribution,
which does not carry that bias:

```text
per-detection deltas, 226 matched pairs across 97 comparable frames
                       median        p95        p99        max     over atol
coord_diff (px)     8.636e-02  2.962e-01  9.283e-01  1.426e+00      3  (1.3%)
conf_diff           2.959e-04  1.287e-02  3.470e-02  1.127e-01     25  (11.1%)
matched IoU           0.99716    min 0.98052

within tolerance:  201/226 detections (88.9%)   vs  77/100 frames (77.0%)
```

The typical detection is far more faithful than the frame verdict suggests: median
confidence drift is 3e-04, roughly 17x *below* the 5e-03 tolerance, and median box
drift is 0.09 px. What fails is a thin tail — p99 confidence drift is 3.5e-02.
Spatially the engine barely moves at all (1.3% of detections over a 1 px bar).

For comparison, V2 FP32 on the same measure:

```text
coord_diff (px)     7.629e-06  6.104e-05  1.425e-04  1.831e-04      0  (0.0%)
conf_diff           1.192e-07  3.418e-06  1.681e-05  3.809e-05      0  (0.0%)
matched IoU           1.00000    min 1.00000
within tolerance:  234/234 detections (100.0%)
```

PASS/FAIL is still decided per frame — changing the gate's semantics is a separate
decision from reporting a better statistic. The distribution is what should be read
when interpreting a reduced-precision engine; it will matter more for INT8, where
deviations are larger.

**Parity therefore cannot carry the accuracy claim for reduced precision.** For V2 it could: the engine matched the ONNX to floating-point noise, so it inherited the ONNX's measured mAP. That inheritance argument breaks the moment precision changes. Accuracy for FP16 is **measured**, not inherited.

### Accuracy (measured, pycocotools, val100 @ conf 0.001)

```text
                    mAP50     mAP50-95    detections
ONNX (CPU)          0.9350     0.7572        986
V2 FP32 engine      0.9350     0.7572        986
V3 FP16 engine      0.9348     0.7572        989

V3 vs V2 delta     -0.0002     0.0000
```

FP16 costs **2e-04 mAP50 and nothing measurable at mAP50-95**. The FP32 engine matches the ONNX exactly at the metric level, which independently corroborates the 100/100 V2 parity result.

Measurement notes:

```text
- conf 0.001 is the mAP protocol (docs/02), NOT a deployment threshold; it keeps
  the low-confidence tail so the full precision-recall curve is built.
- All three rows are measured by scripts/evaluate_engine_map.py with identical
  metric code (MODE=onnx | MODE=engine). This is deliberate: the docs/02 figures
  (0.9394 / 0.7595) came from Ultralytics' mAP implementation, and scoring an
  engine with pycocotools against an Ultralytics number would fold a metric
  change into what is meant to be a precision delta.
- pycocotools vs Ultralytics on the same ONNX: 0.9350 vs 0.9394 mAP50 (0.0044),
  0.7572 vs 0.7595 mAP50-95 (0.0023). That gap is the metric implementation,
  and it is why V3 is compared against the pycocotools V2 row.
- Detection cap is not binding: busiest image has 62 detections against a 300 cap.
```

### Latency (batch=1, 10x100 = 1000 pooled samples, GPU 0)

V2 and V3 were benchmarked **back-to-back on the same card under identical conditions** — same seed, repeats, warmup, image size, and the same two tolerated dormant contexts. Measuring them in separate windows would put thermal/driver/load drift inside a comparison meant to isolate precision.

```text
V3 FP16
Stage          Mean      Std   Median      Min      P95      P99      Max
Preprocess    1.453    0.466    1.326    0.998    2.263    2.538    8.386
Inference     1.588    0.082    1.561    1.535    1.726    1.947    2.767
Postprocess   0.022    0.005    0.021    0.019    0.028    0.042    0.105
Total         3.063    0.491    2.922    2.580    3.889    4.175   10.331
                                                        ->  342.2 FPS
```

### V2 vs V3 — the precision effect

```text
stage              V2 FP32     V3 FP16    speedup
Preprocess         1.372 ms    1.326 ms     1.03x    CPU, precision-invariant
Inference          2.318 ms    1.561 ms     1.49x    <- the precision effect
Postprocess        0.021 ms    0.021 ms     1.00x
Total              3.736 ms    2.922 ms     1.28x
FPS                   267.7       342.2     1.28x
engine size         12.8 MB      7.0 MB     0.55x
mAP50               0.9350      0.9348    -0.0002
mAP50-95            0.7572      0.7572     0.0000
```

**Report both speedups.** FP16 accelerates inference only: 1.49x is the precision effect, 1.28x is what a deployment actually sees. Quoting total alone understates the precision work; quoting inference alone overstates the deployment benefit.

Preprocess is now **45%** of the V3 total (1.326 of 2.922 ms) and is precision-invariant CPU work. It is already the largest single stage and will dominate further as INT8 shrinks inference — the next meaningful latency win is moving preprocess onto the GPU, not reducing precision further.

Both runs record `exclusive_gpu: false`: two dormant `uceeesi` contexts (496 MiB, 0% util) were tolerated via `GATE_ALLOW_IDLE_MIB=600` (see docs/03). Evidence they did not interfere: V2's inference median landed within 0.7% of an earlier *exclusive* run with **lower** variance (std 0.140 vs 0.253 ms). Contention would raise both.

---

## Reproduce

```bash
source ~/venvs/medtronic-trt/bin/activate
ENG=models/yolo26n_sanoscience_full_left/best_fp16.engine

# build (TF32 off for every precision unless ALLOW_TF32=1)
PRECISION=fp16 DEVICE=<idle_gpu> python scripts/build_tensorrt_engine.py

# faithfulness smoke test vs the FP32 ONNX (expected to FAIL at FP32 tolerances)
ENGINE_PATH=$ENG DEVICE=<idle_gpu> python scripts/validate_engine_parity.py

# accuracy -- this is the claim, not parity
MODE=engine ENGINE_PATH=$ENG DEVICE=<idle_gpu> python scripts/evaluate_engine_map.py

# latency (requires a fully idle GPU)
ENGINE_PATH=$ENG DEVICE=<idle_gpu> BENCHMARK_REPEATS=10 python scripts/benchmark_latency_trt.py
```

---

## Next stage

INT8 / PTQ (docs/05), which adds a calibration pass. The accuracy method established here carries over: parity is a smoke test, `evaluate_engine_map.py` produces the number, and every precision is compared against the pycocotools V2 row.
