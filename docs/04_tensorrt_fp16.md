# TensorRT FP16 Engine (V3)

Builds a TensorRT **FP16** engine from the same `best.onnx`, measured against the V2 FP32 engine (docs/03) — not against PyTorch, so the delta isolates precision.

**Result: 1.49x faster inference, 0.55x engine size, for -0.0002 mAP50.**

---

## Running it

Identical to docs/03. Two things change for the whole chain:

```text
build       PRECISION=fp32  ->  PRECISION=fp16
everything  ENG=...best_fp32.engine  ->  ENG=...best_fp16.engine
```

```bash
source ~/venvs/medtronic-trt/bin/activate
ENG=models/yolo26n_sanoscience_full_left/best_fp16.engine

PRECISION=fp16 DEVICE=0 python scripts/build_tensorrt_engine.py
ENGINE_PATH=$ENG DEVICE=0 python scripts/validate_engine_parity.py            # expected FAIL, see below
MODE=engine ENGINE_PATH=$ENG DEVICE=0 python scripts/evaluate_engine_map.py   # the accuracy claim
ENGINE_PATH=$ENG DEVICE=0 BENCHMARK_REPEATS=10 python scripts/benchmark_latency_trt.py
```

The build differs by exactly one line executing (`scripts/build_tensorrt_engine.py`):

```python
if PRECISION == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)      # <- only this
if not ALLOW_TF32:
    config.clear_flag(trt.BuilderFlag.TF32)    # same for both precisions
```

---

## What actually gets reduced to FP16

`BuilderFlag.FP16` is **permission, not obligation**. It does not cast the whole network; it lets the builder use FP16 kernels where its timing and precision heuristics choose to. TensorRT FP16 is *mixed* precision.

**Verified for this engine:**

```text
weights      mostly stored FP16. Engine is 7.0 MB vs V2's 12.8 MB (0.55x).
             Weights dominate engine size, so ~half means the bulk converted.
I/O          UNCHANGED, still FP32:
               input   images  [1,3,640,640] FLOAT
               output  output0 [1,300,6]     FLOAT
             So the calling code is identical across precisions -- the wrapper
             in validate_engine_parity.py works on both without modification.
TF32         off, same as V2 (provenance tf32_enabled: false on both)
```

**General TensorRT semantics** (applies to this engine, but the per-layer split was not measured here):

```text
activations  intermediate tensors between FP16 layers flow as FP16. This is
             where most of the speed comes from -- halved memory traffic, not
             just faster maths.

biases       not the precision-critical step. Tensor Core FMA multiplies in
             FP16 but ACCUMULATES IN FP32, and bias is applied in that wider
             accumulator before the result is cast back down. So a bias never
             sees an FP16-precision addition against an FP16-precision sum.

fallback     numerically sensitive layers (normalisations, reductions,
             accumulation-heavy ops) stay in higher precision where the builder
             prefers it. These are the layers TF32 would have governed -- which
             is exactly why TF32 had to be turned off here too, not just for
             FP32 (docs/03).
```

The short version: **weights and activations go to FP16; accumulation and bias addition stay wider; I/O stays FP32; some layers do not convert at all.** That layered conservatism is why FP16 error does not compound through 469 layers, and why accuracy holds.

---

## Controlling the comparison

Both provenance sidecars diffed field by field — precision is the only difference:

```text
field             V2 FP32              V3 FP16              same?
precision         fp32                 fp16                 NO  (intended)
fp16_flag         False                True                 NO  (intended)
tf32_enabled      False                False                yes
tensorrt_version  10.16.1.11           10.16.1.11           yes
workspace_gib     8.0                  8.0                  yes
num_layers        469                  469                  yes
gpu_name          A100-SXM4-80GB       A100-SXM4-80GB       yes
device_index      0                    0                    yes
source_onnx sha   ac188a00...          ac188a00...          yes
io_tensors        identical            identical            yes
```

`precision` and `fp16_flag` are the same knob. No optimisation profile exists on either side (static `[1,3,640,640]`), so that variable is absent by construction.

**The TF32 trap.** The build script originally cleared TF32 only on the fp32 path (`elif PRECISION == "fp32"`), so the first FP16 engine inherited TensorRT's default TF32=**on** while V2 had it off. Since TF32 governs the FP32 fallback layers inside a mixed-precision engine, the V2-vs-V3 delta would have measured FP16 on the fast path *and* TF32 on the slow path. Now cleared unconditionally.

---

## Results

### Parity — FAIL, and that is correct

```text
77/100 frames pass     201/226 detections within tolerance
  max coord diff  1.426 px       median  8.636e-02 px
  max conf diff   0.113          median  2.959e-04     (conf_atol 5e-3)
```

**Tolerances were deliberately not relaxed.** They encode FP32 bit-exactness (V2 hits ~1e-04 px), which a 10-bit mantissa cannot meet. Loosening `conf_atol` after seeing the result would manufacture a meaningless PASS.

Reading the distribution instead of the verdict: median confidence drift is **17x below** tolerance, median box drift is 0.09 px, matched IoU median 0.997. Only a thin tail fails. Spatially the engine barely moves — 1.3% of detections exceed a 1 px bar.

Note the per-frame rate is a max over a *varying* number of detections, so it partly measures how many tools are in shot:

```text
1 box: 4% fail   2 boxes: 25%   3: 16%   4: 50%   5: 100%   (last buckets: 12 and 3 frames)
```

That is why the script also reports pooled per-detection stats — **88.9% of detections pass vs 77.0% of frames.** For V2 the same table is 234/234 with median drift 7.6e-06 px and matched IoU 1.00000.

**Parity cannot carry the accuracy claim for reduced precision.** V2 could inherit the ONNX's mAP because it matched to floating-point noise; that argument breaks the moment precision changes. FP16 accuracy is measured.

### Accuracy (pycocotools, val100 @ conf 0.001)

```text
                 mAP50    mAP50-95   dets
ONNX (CPU)       0.9350    0.7572     986
V2 FP32          0.9350    0.7572     986
V3 FP16          0.9348    0.7572     989
delta           -0.0002    0.0000
```

Why it holds: mAP is ranking-based. A detection moving 0.91 -> 0.89 does not change its order against background at 0.05, and a 0.09 px shift does not cross an IoU threshold. **FP16 perturbs confidence scores without changing what the model detects.**

All three rows use identical metric code (`MODE=onnx | engine`) — the docs/02 figures came from Ultralytics, and mixing metrics would fold a metric change into the precision delta.

### Latency (batch=1, 10x100, GPU 0, same session as V2)

```text
Stage          Mean      Std   Median      Min      P95      P99      Max
Preprocess    1.453    0.466    1.326    0.998    2.263    2.538    8.386
Inference     1.588    0.082    1.561    1.535    1.726    1.947    2.767
Postprocess   0.022    0.005    0.021    0.019    0.028    0.042    0.105
Total         3.063    0.491    2.922    2.580    3.889    4.175   10.331
                                                          ->  342.2 FPS
```

### V2 vs V3

```text
stage            V2 FP32    V3 FP16   speedup
Preprocess       1.372 ms   1.326 ms   1.03x    CPU, precision-invariant
Inference        2.318 ms   1.561 ms   1.49x    <- the precision effect
Postprocess      0.021 ms   0.021 ms   1.00x
Total            3.736 ms   2.922 ms   1.28x
FPS                 267.7      342.2   1.28x
engine size       12.8 MB     7.0 MB   0.55x
mAP50              0.9350     0.9348  -0.0002
```

**Report both speedups.** 1.49x is the precision effect; 1.28x is what deployment sees. Preprocess is now **45%** of the V3 total and is precision-invariant CPU work — the next latency win is moving it onto the GPU, not reducing precision further.

Both runs record `exclusive_gpu: false` (two dormant contexts tolerated via `GATE_ALLOW_IDLE_MIB=600`, see docs/03). Measured back-to-back in one session so the pair is internally consistent.

---

## Caveats

```text
- 100 images / 237 boxes. A -0.0002 mAP50 delta is inside the noise floor of a
  set this size. Read as "no detectable degradation on val100". Re-run on the
  full 6,449-image set before FP16 goes anywhere that matters.
- Neither latency run had an exclusive GPU.
- Compare Inference, not Total, across runs.
```

---

## Next stage

INT8 / PTQ (docs/05), which adds a calibration pass. Expect accuracy to actually fight back there — INT8 maps activations onto 256 levels via a calibrated scale, so a poor scale clips real signal, unlike FP16 which only rounds. The method carries over: parity as smoke test, `evaluate_engine_map.py` for the number, everything compared against the pycocotools V2 row.
