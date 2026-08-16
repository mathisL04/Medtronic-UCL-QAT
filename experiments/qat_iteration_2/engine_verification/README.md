# Production engine verification — per-layer precision, Q/DQ, reformats, latency

Full per-layer audit of the six production TensorRT engines. Goal: confirm what
each engine actually runs at (not what we intended), account for the Q/DQ, and
test the **launch-bound hypothesis** with reformat-cost and cross-precision
latency measured in one exclusive session.

Reproduce: `DEVICE=2 python experiments/qat_iteration_2/engine_verification/verify_engines.py`
Raw per-engine data: `<engine>/layer_info.json` (inspector dump), `layers_parsed.json`,
`profile_per_layer.json`, `summary.json`; index: `_index.json`.

---

## Method — and a deviation from the requested tool (read first)

**`trtexec` is not installed in this environment.** The `tensorrt` package here is
the pip wheel (10.16.1.11), which ships the Python API but **no `trtexec` binary**,
and there is no `/usr/src/tensorrt` sample tree. The requested command
(`trtexec --loadEngine --exportLayerInfo --exportProfile --separateProfileRun`)
therefore cannot be run as written. Rather than stop, I reproduced the exact data
those flags export via the TensorRT Python API, which is the same underlying source:

```
requested trtexec flag        equivalent used here
--loadEngine                  runtime.deserialize_cuda_engine(bytes)         (no rebuild)
--exportLayerInfo             EngineInspector.get_engine_information(JSON)    (static)
--exportProfile / dumpProfile IProfiler attached to the execution context    (instrumented run)
--separateProfileRun          a SEPARATE clean CUDA-event timing run, profiler OFF
```

No engine was rebuilt — every number below comes from loading the existing `.engine`
files. No training/export/build script was modified; `verify_engines.py` is new and
read-only w.r.t. the engines.

**GPU state at measurement time** (timings are only trustworthy on an exclusive idle GPU):

```
gpu0: A100-80GB  0% util, ~15 GB resident (foreign)      — not used
gpu1: A100-80GB  100% util, 80 GB (foreign, saturated)   — not used
gpu2: A100-80GB  0% util, 14 MiB — EXCLUSIVE & IDLE       — ALL timings ran here
gpu3: A100-80GB  our V2_batch32_lr2e-3 training           — separate device, untouched
```

All four GPUs are identical A100-SXM4-80GB, so the engines are portable across them.
Timings below ran on **gpu2, exclusive, 300 timed iters after 50 warmup**. They are
trustworthy. The instrumented per-layer profile ran separately (100 iters); its
**absolute** times are inflated ~1.4–1.5× by instrumentation and are used only for
per-layer *shares*, never as latency.

---

## 1. Precision breakdown — all six engines

Precision is read from each layer's **output** `Format/Datatype` (there is no
"Precision" field on an inspector layer — the same logic `build_tensorrt_engine.py`
relies on). "detailed" = the engine was serialized with `ProfilingVerbosity.DETAILED`,
which is required for any per-layer readout.

| engine | file | layers | detailed? | INT8 | FP16 | FP32 | INT32 | reformats |
|---|---|--:|:--:|--:|--:|--:|--:|--:|
| FP32 | baseline/fp32/best_fp32.engine | 355 | **no** | — | — | — | — | — |
| FP16 | baseline/fp16/best_fp16.engine | 231 | **no** | — | — | — | — | — |
| V4_int8_ptq | baseline/int8_ptq/best_int8.engine | 191 | yes | 147 | 0 | 42 | 2 | 45 |
| **PTQ_int8_fp16** | ptq_baseline/best_int8_fp16.engine | 189 | yes | **149** | **0** | **38** | 2 | 45 |
| PTQ_maxfp16 | ptq_baseline/best_int8_fp16_max.engine | 198 | yes | 149 | 48 | 1 | 0 | 53 |
| V6_qat | qat/v6_final/best_qat_int8.engine | 253 | **no** | — | — | — | — | — |

**PTQ_int8_fp16 matches the expected split.** Expected "149 INT8 / 40 FP32 / 0 FP16";
actual is **149 INT8 / 0 FP16 / 40 non-INT8**, where the 40 splits precisely into
**38 FP32 + 2 INT32** (the 2 INT32 are detection-index tensors in the NMS region — see
§2). This is a labelling refinement, not a discrepancy: the "40 FP32" figure counted
the 2 INT32 as float.

**Three engines carry no per-layer precision** (`detailed=no`):
- **FP32 / FP16** — irrelevant to a per-layer audit (single precision by definition;
  totals still measured: 355 and 231 layers).
- **V6_qat — this one matters and is a real gap.** `build_tensorrt_int8_qdq.py` (the
  QAT build path) never sets `profiling_verbosity=DETAILED`, so the QAT production
  engine's per-layer precision is **unrecoverable without a rebuild**, which the task
  forbids. What *is* provable about V6 is in §3 and §5 (layer count + latency only).

> Contradiction flagged, not reconciled: `build_tensorrt_engine.py:174` *does* set
> `DETAILED`, yet the FP32/FP16 engines read back non-detailed. They were most likely
> built before that line was added (or by an older path). Doesn't affect conclusions
> — noting it per instruction.

---

## 2. Non-INT8 layers vs expectation — where INT8 stops

For the three INT8 engines with detailed info, the per-module precision is **identical
in structure**: the float layers appear only in the attention blocks, the head, and the
NMS region. Everything else is 100% INT8.

**PTQ_int8_fp16 — per-module precision (module = Ultralytics `model.N`):**

```
model.0–9   backbone convs/blocks   INT8 only                         100% INT8
model.10    backbone attention      INT8 10 | FP32 14                  ← attention
model.11    neck                    INT8 only
model.13,14 neck                    INT8 only
model.16,17 neck                    INT8 only
model.19,20 neck                    INT8 only
model.22    head attention          INT8 12 | FP32 10                  ← attention
model.23    detection head          INT8 17 | FP32 10                  ← head
(no model.N) NMS / end2end          6 "kgen" layers incl. 2 INT32     ← NMS region
```

**This confirms the expectation exactly:**
- **Backbone `model.0–9` and neck `model.11–21` are 100% INT8** — every conv quantised. ✅
- Float survives **only** at `model.10` and `model.22` (the two attention blocks),
  `model.23` (the detection head), and 6 module-less `kgen` layers (TensorRT
  JIT pointwise kernels forming the NMS/end2end region — they carry the 2 INT32
  detection-index tensors and the FP32 box/score outputs). ✅

The engine's single output is `output0 [1, 300, 6]` FP32 (NMS-integrated, ≤300
detections × 6), input `images [1,3,640,640]` FP32 — so float I/O plus float
attention/head/NMS is exactly the non-INT8 surface, and it is the same set across
V4, PTQ_int8_fp16, and PTQ_maxfp16. The only difference between them is *what*
precision the non-INT8 region uses:

```
V4_int8_ptq    attention/head/NMS run FP32   (147 INT8 / 42 FP32 / 2 INT32)
PTQ_int8_fp16  attention/head/NMS run FP32   (149 INT8 / 38 FP32 / 2 INT32)   ← +2 more convs INT8 than V4
PTQ_maxfp16    attention/head/NMS run FP16   (149 INT8 / 48 FP16 / 1 FP32)    ← float region pushed to FP16
```

`max`-mode did **not** add INT8 coverage (still 149) — it only moved the residual
float work from FP32 to FP16. That is consistent with the earlier finding that the
attention softmax/matmul and NMS index ops are functional ops, not quantisable modules.

---

## 3. Q/DQ accounting

The "207 Q/DQ pairs" is a property of the **V6 QAT ONNX** (`quantize_linear_nodes` in
its provenance), not of the PTQ engines — PTQ uses implicit quantization (calibration
cache) and has **zero** Q/DQ nodes in its ONNX, yet still produces 45 reformats. So
Q/DQ node count and engine reformat count are unrelated quantities.

**What is proven:**
- In TensorRT, Q/DQ ONNX nodes are never preserved as distinct engine layers. A
  Q→conv→DQ triple around a weight fuses into a single INT8 conv kernel; Q/DQ that sit
  at a precision boundary that cannot fuse become standalone reformat/quantize nodes.
- The PTQ engines demonstrate the endpoint of that fusion directly (§5): 45–53
  reformats, of which only 3 (PTQ_int8_fp16) are true FP32→INT8 precision crossings.

**What is NOT available:** the fused-vs-standalone split of V6's 207 pairs. The V6
engine was built without DETAILED verbosity, so its layers/reformats can't be
enumerated, and rebuilding is out of scope. The one **proven** structural fact about V6:

```
V6_qat  engine layers: 253      (vs PTQ_int8_fp16: 189  → +64 layers)
```

**Inferred (not proven):** those ~64 extra layers are largely un-fused Q/DQ/reformat
nodes. Explicit-Q/DQ (QAT) engines routinely fuse less aggressively than
implicit-quant (PTQ) engines, and the layer-count delta lines up with V6 being slower
(§5). Plausible and consistent, but not directly observable here without a DETAILED
rebuild.

---

## 4. Reformat cost — the launch-bound test (priority)

**Reformat boundaries, PTQ_int8_fp16 (45 reformat layers = 24% of 189):**

```
INT8 -> INT8   x36   layout-only shuffles between INT8 kernel tactics (e.g. NC/32HW32 <-> linear)
FP32 -> FP32   x6    layout shuffles inside the float region
FP32 -> INT8   x3    the ONLY true precision-boundary reformats (quantization entry)
```

Key measured fact: **80% of the reformats (36/45) are INT8→INT8 layout conversions**,
not precision conversions. Only 3 reformats actually cross FP32→INT8. So the reformat
overhead is dominated by INT8 tensor-layout juggling between successive INT8 kernels,
**not** by float↔int precision conversion. PTQ_maxfp16 adds 8 more reformats (53 total)
purely from the new INT8↔FP16 boundaries it introduced (5 INT8→FP16, 2 FP16→INT8).

**Instrumented per-layer share (relative, inflated absolute — use share only):**

```
V4_int8_ptq     reformats = 18.9% of instrumented per-layer time
PTQ_int8_fp16   reformats = 19.1%
PTQ_maxfp16     reformats = 21.9%
```

Roughly **one fifth** of the INT8 engine's kernel time does zero arithmetic — it only
moves bytes between layouts. On a compute-bound workload that fraction would be
negligible; here it is large, which is itself a launch/overhead-bound signature.

---

## 5. Cross-precision latency — clean, exclusive gpu2, one session

CUDA-event compute-only time (`execute_async_v3` between events), 300 iters after 50
warmup, profiler OFF. Trustworthy.

| engine | precision | layers | median ms | mean ms | min | max |
|---|---|--:|--:|--:|--:|--:|
| FP32 | FP32 | 355 | 2.024 | 2.066 | 2.012 | 2.586 |
| FP16 | FP16 | 231 | 1.153 | 1.168 | 1.148 | 4.578* |
| V4_int8_ptq | INT8/FP32 | 191 | 1.081 | 1.084 | 1.078 | 1.235 |
| PTQ_int8_fp16 | INT8/FP32 | 189 | 1.092 | 1.099 | 1.088 | 2.367* |
| PTQ_maxfp16 | INT8/FP16 | 198 | 1.070 | 1.075 | 1.067 | 1.606 |
| V6_qat | INT8 (QAT) | 253 | 1.398 | 1.403 | 1.394 | 1.560 |

\* isolated max outliers (one-off scheduler hiccups); medians/means are tight.

**Launch-bound hypothesis: SUPPORTED.**

```
compute-bound expectation:  INT8 ~2x faster than FP16 ~2x faster than FP32
observed:                   INT8 (1.09) only ~5% faster than FP16 (1.15)
                            FP16 only 1.76x FP32, not ~2-4x
```

- **INT8 ≈ FP16.** Dropping from 16-bit to 8-bit arithmetic buys ~5%, not ~2×. If
  arithmetic were the bottleneck INT8 would pull far ahead. It doesn't → the model is
  too small (2.5M params, 6.1 GFLOPs) to saturate the A100 at batch 1; kernel-launch and
  scheduling overhead dominate.
- **Latency tracks layer count, not FLOPs (inferred).** FP32 has 88% more layers than
  PTQ_int8_fp16 (355 vs 189) and is 85% slower (2.02 vs 1.09 ms) — nearly 1:1. More
  kernels launched, more time, roughly independent of per-kernel precision. This is a
  correlation across 6 points, not a controlled proof (compute differs too), but it is
  the classic launch-bound fingerprint and is consistent with §4 (≈20% of time is
  pure data-movement reformats).

**V6 QAT is the slowest INT8 engine (proven):** 1.398 ms vs ~1.08–1.09 ms for the PTQ
INT8 engines — ~28% slower — despite being nominally INT8 and being the **most
accurate** model produced (mAP50-95 0.7801). The proven correlate is its layer count:
253 vs 189 (+64 layers → +64 kernel launches). Under a launch-bound regime, more
launches ⇒ more latency, so the accuracy win carries a real latency cost. (Whether the
extra layers are specifically un-fused Q/DQ is inferred — §3 — since V6 lacks detailed
info.)

---

## 6. Proven / inferred / unavailable — summary

**Proven (measured directly on the engines):**
- Per-layer precision for V4, PTQ_int8_fp16, PTQ_maxfp16 (table §1).
- PTQ_int8_fp16 = 149 INT8 / 38 FP32 / 2 INT32 / 0 FP16 — matches expectation (149 INT8,
  40 non-INT8).
- Backbone `model.0–9` + neck `model.11–21` = 100% INT8; float only at attention
  (`model.10`, `model.22`), head (`model.23`), and NMS (6 `kgen`, incl. 2 INT32).
- Reformat counts and boundary datatypes; 36/45 are INT8→INT8 layout, only 3 are
  FP32→INT8 precision crossings.
- Clean cross-precision latencies on exclusive gpu2 (table §5).
- Layer counts for all six engines; V6_qat = 253 layers, 1.398 ms.

**Inferred (reasoned from proven data, not directly observed):**
- Launch-bound *causation* (layer-count↔latency correlation; compute also varies).
- V6's +64 layers are largely un-fused Q/DQ/reformat nodes.
- `kgen` layers = the NMS/end2end region.
- Reformat "cost" from instrumented share (instrumentation inflates absolute time).

**Not available without a rebuild (out of scope):**
- V6_qat per-layer precision and fused-vs-standalone Q/DQ split — engine built without
  `DETAILED` verbosity (`build_tensorrt_int8_qdq.py` does not set it).
- FP32/FP16 per-layer precision — non-detailed, but moot (single precision).

**Caveats:**
- trtexec unavailable → Python-API equivalents used (same underlying data; noted §Method).
- Instrumented profile times are inflated ~1.4–1.5× — used for shares only, never latency.
- gpu3 was running our own `V2_batch32_lr2e-3` training on a separate device; all
  measurements ran on the exclusive, idle gpu2 and are unaffected.
