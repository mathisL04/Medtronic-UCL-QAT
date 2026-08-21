# PTQ INT8+FP16 baseline — LOCKED

Built with the reused Phase-1 script `scripts/tensorrt/build_tensorrt_engine.py`
(`PRECISION=int8 INT8_ALLOW_FP16=1`), calibrated on the deterministic seed-42 500-frame set.
TensorRT 10.16.1.11, YOLO26n, input `[1,3,640,640]` → output `[1,300,6]`.

---

## 1. Canonical baseline — `best_int8_fp16.engine` (LOCKED)

The **default** INT8+FP16 build (TensorRT's autotuner uses FP16 where profitable, keeps the rest
FP32). Phase reference — **use these numbers everywhere**:

```text
mAP50-95   0.7564          (full 6,449 val, conf 0.001, pycocotools)
mAP50      0.9281
kernel     1.082 ms        (CUDA-event, idle-gated, EXCLUSIVE GPU-0, same-session sweep)
size       4.29 MB         (engine file on disk)
coverage   149 INT8 / 189 layers   (0 FP16 emitted, 40 FP32 fallback)
```

Run-to-run GPU-state variation is ~±0.05 ms (an earlier session read 1.126); 1.082 is the value from
the sweep that also produced the FP16/FP32 rows below, so all latencies are identical-conditions.

---

## 2. Comparison — all rows EXCLUSIVE GPU-0, same session

`kernel` = CUDA-event median, idle-gated, exclusive, one session (`latency_sweep_waiter`).

| engine | INT8 | mAP50-95 | Δ vs FP32 | kernel ms | size |
|---|---|---|---|---|---|
| FP32 | 0 | 0.7747 | — | 2.011 | 12.8 MB |
| FP16 | 0 | 0.7748 | +0.0001 | 1.248 | 7.0 MB |
| **INT8+FP16 (canonical)** | **149** | **0.7564** | **−0.0183** | **1.082** | **4.29 MB** |
| INT8+FP16 max (pushed) | 149 | 0.7513 | −0.0234 | 1.060 | 4.21 MB |
| INT8+FP32 (V4) | 147 | 0.7571 | −0.0176 | 1.073 | 4.2 MB |

**Finding:** FP32 is the only latency outlier (~2.0 ms, ~2×). FP16 and every INT8 variant cluster in
the same **~1.06–1.25 ms band** — the differences inside that band are within run-to-run GPU-state
noise (launch-bound model). So INT8's −0.018 mAP buys **no** latency advantage over FP16; the three
INT8 builds (default / max / V4) are latency-indistinguishable.

---

## 3. Backbone INT8 count — reconciled

Both figures are correct; they differ only by where the backbone boundary is drawn:

```text
model.0–9   (conv stem + C3k2/C2f blocks + SPPF)   = 60 INT8   (pure conv feature extractor)
model.10    (PSA attention block: 10 INT8 convs)   = 10 INT8
------------------------------------------------------------------
backbone as 0–9 = 60 INT8    |    backbone as 0–10 (incl. attention) = 70 INT8
```

The §4 forward-flow uses **model.0–9 = 60** for the pure-conv backbone, model.10 listed separately.

---

## 4. Forward flow — CANONICAL default engine (149 INT8 · 0 FP16 · 40 FP32)

```text
INPUT [1,3,640,640]                                             FP32  (engine I/O)
  │ reformat → INT8
BACKBONE
  model.0–9   conv stem + C3k2/C2f + SPPF          INT8   60        (100% INT8)
  model.10    PSA attention: convs                 INT8   10
              PSA attention: softmax/matmul/mul     FP32   14        (autotuner kept FP32)
NECK / PAN
  model.11–21 up/down convs, concat, C3k2          INT8   50        (100% INT8)
  model.22    attention: convs                     INT8   12
              attention: softmax/matmul             FP32   10
DETECT HEAD (model.23)
  cls/box conv branches                            INT8   17
  decode + NMS-free postprocess                    FP32   10        (TopK/Tile/Mod/gather region)
  reformat/const glue                              FP32    6
OUTPUT [1,300,6]                                               FP32  (engine I/O)
--------------------------------------------------------------------------------
TOTAL: 149 INT8 · 0 FP16 · 40 FP32
```

In the canonical engine **there is no FP16** — the 40 non-INT8 layers are FP32 (attention math + head
index region + glue). INT8 = every convolution (backbone + neck + head branches + 4 attention convs).

### 4b. "Pushed-to-limit" data point — `best_int8_fp16_max.engine` (NOT the baseline)

Forcing FP16 onto the attention float core drives FP32 from 40 → 1, giving **149 INT8 · 48 FP16 · 1
FP32 · 3 integer** — but it **regressed accuracy to 0.7513** at 1.060 ms (no latency gain). Recorded
only as evidence that maximal quantization does not pay off here. Its per-layer flow (attention →
FP16, head decode → FP16, indices → integer) is the max-FP16 variant, kept for reference, **not** the
baseline.

---

**Baseline LOCKED** on the canonical default engine: **mAP50-95 0.7564 · 1.082 ms · 4.29 MB · 149
INT8** (all reference latencies same-GPU exclusive). Ready for the QAT hyperparameter sweep.
