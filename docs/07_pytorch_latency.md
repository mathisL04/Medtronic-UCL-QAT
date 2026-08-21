# PyTorch-side Latency Study (raw models, NO TensorRT)

Measures the **eager PyTorch models themselves** — before any ONNX/TensorRT
conversion — to quantify the pre-conversion cost. This is deliberately **separate
from the TensorRT engine comparison** (docs/03–06): the point is what raw PyTorch
costs, so the TensorRT speedup can be attributed to the conversion, not the
precision.

Script: `scripts/benchmark/benchmark_latency_pytorch.py` (`MODEL_MODE={fp32|fp16|qat}`).
Sidecars: `results/reports/2_pytorch_latency/*.provenance.json`.

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

## Session seam: why the two tables are not merged

The FP32 pipeline figure below (**8.642 ms**, `benchmark_latency.py`, Ultralytics
predict counters, val100) comes from an earlier session on a different day. It is the
V1 baseline quoted in docs/03. It is **not** combined with the same-session table
above: mixing one old-session row into fresh ones reintroduces exactly the
cross-session seam this study exists to avoid. For internal comparison against the
QAT and FP16 rows, use the fresh FP32-B figure (9.941 ms). For the number the
TensorRT stages were historically measured against, use 8.642 ms.

Both are correct; they answer different questions.

---

## The FP32 PyTorch baseline (earlier session, moved here from docs/01)

This is the original single-frame PyTorch measurement: the **8.642 ms** figure that
stages 3-6 quote as the pre-conversion reference. It predates the same-session table
above and is deliberately kept separate from it, for the reason given in
*Session seam* immediately above. It lived in `docs/01` until the documentation was
reorganised; nothing about the measurement has changed.

**Deployment latency (batch=1, Geneva A100) — reported baseline.** Single-frame inference timed with `scripts/benchmark/benchmark_latency.py`: one image per `model.predict()` call (batch=1), wall-clock around each call, the 100-frame validation subset preloaded to RAM, warmup, then pooled per-stage **medians** over 10 repeats × 100 frames (1000 samples) on a verified-idle A100.

```text
preprocess    1.552 ms   (median)
inference     6.844 ms   (median)
postprocess   0.238 ms   (median)
total         8.642 ms   (median)  ->  115.7 FPS
```

(Stage figures are each the median of their own column, so they need not sum exactly to the median total. Here they differ by 0.008 ms: 1.552 + 6.844 + 0.238 = 8.634 against a reported total of 8.642.)

Run conditions:

```text
Date:        16 July 2026
Host:        Geneva
GPU:         A100-SXM4-80GB, GPU 3, explicitly pinned
Model:       YOLO26n, batch=1
Samples:     10 repeats x 100 frames = 1000
Seed:        42
Image size:  640
conf:        0.25
Warmup:      30 images
Priority:    nice -n 10
Contention:  none on any repeat (OtherProc=0, memory flat at 1407 MiB)
GPU peak:    30-33%, which is our own batch=1 load
```

Full distribution, in milliseconds:

```text
stage          mean     std   median     min     p95     p99     max
preprocess    1.674   0.325    1.552   1.493   2.457   2.490   4.488
inference     6.884   0.260    6.844   6.705   7.040   8.236  10.704
postprocess   0.245   0.085    0.238   0.229   0.259   0.295   2.738
total         8.803   0.446    8.642   8.442   9.600  10.390  12.574
```

Contention only ever adds time, so the mean is dragged upward while the median and the min estimate the clean machine. FPS is quoted median-based at **115.7**; the mean-based figure is 113.6, giving a range of 113.6-115.7. Every downstream comparison in this project is median-based, so 115.7 is the number the TensorRT stages are measured against.

The maximum matters for a surgical detector, but a max over 1000 samples is hostage to one bad frame, which is why p95 and p99 are reported alongside it — they separate a real tail from a single artefact.

**The tails are CPU-side.** Preprocess has p95 2.457 and max 4.488 against a 1.552 median, which is letterbox jitter. Postprocess carries one 2.738 ms outlier against a 0.238 median. Inference is tight by comparison — p95 7.040 against a 6.844 median. Any tail quoted for this baseline is a CPU artefact, not the network.

Batch=1 is the right unit for this project: surgical frames arrive one at a time, so the cost that matters is what a single live frame takes end to end — not throughput amortised over a batch. That is also why this total is ~8× the *Validation throughput* figure above: that one batches, this one does not. Hardware rules out the alternative explanation — the throughput figure was measured on the **slower V100 yet reads lower**, so the gap is batch size, not the GPU.

**The idle-GPU gate.** Before claiming the GPU, the benchmark refuses to run unless the target device is idle for *compute* — checked via `nvmlDeviceGetComputeRunningProcesses`, ignoring graphics contexts such as Xorg — and it samples per-repeat GPU state to flag mid-run contention.

Why the gate matters. The 14 July FP32 baseline measured 16.928 ms median total; re-run on a verified-idle GPU it measured 8.642 ms — a 49% correction. Both runs used identical timing methodology (RAM preload, warmup, single-thread, `cudnn.benchmark`), so this correction is entirely environmental: the machine changed, not the code. (Caveat: the box became idle and the NVIDIA driver was reinstalled the same afternoon, so idle-versus-driver cannot be separated from the available data.) The code contribution is a separate, far smaller effect — adding warmup, single-threading, and `cudnn.benchmark` to the earlier un-instrumented benchmark moved median total by only −0.92 ms, against the −8.29 ms environmental swing. That such contention is routine rather than exceptional was confirmed on 2026-07-17, when a re-run was refused by the gate: another user's 4-GPU DDP training run held ~47 GB and 100% util across all four A100s. A latency figure is only comparable when measured on a verified-idle GPU.

**An earlier run, archived for context.** A first exploratory benchmark measured 16.103 ms median total (62.2 FPS), with preprocess 2.637, inference 10.155 and postprocess 5.005. It is recorded here for provenance only and is **not** a controlled comparison against the figures above: it predates device pinning, the idle gate, warmup and single-threading, and it ran in a different and noisier machine state, so no single cause can be attributed to the difference. Its postprocess figure is the clearest measure of how noisy that environment was — single-class NMS over roughly two boxes costs a fraction of a millisecond, so 5.005 ms was never NMS cost. The controlled before/after is the 14 July to 16 July pair described above, not this run.

**What changed in the benchmark itself.** The current `scripts/benchmark/benchmark_latency.py` differs from the earliest version in five ways, all of which exist to make a number attributable rather than to make it faster: `DEVICE` is strict and recorded, so a missing environment variable can no longer silently fall back to GPU 0; the GPU is gated via pynvml before CUDA initialises; utilisation, memory and process count are snapshotted around every repeat and written to the CSV, so a contended run is no longer indistinguishable from a clean one; median, p95 and p99 are reported alongside mean, std, min and max; and both the script and the subset generator are tracked in git rather than living only on Geneva's disk. Timing methodology is otherwise unchanged — same `result.speed[]` stage timings, same 100 RAM-preloaded frames, same warmup, same three stages.

Threading is **not** among the changes. `torch.set_num_threads(1)` and `torch.backends.cudnn.benchmark = True` are still set (`scripts/benchmark/benchmark_latency.py:230` and `:235`), exactly as in the archived `benchmark_fp32_geneva_ram_stable.py`. The hypothesis that single-threading was throttling preprocess and postprocess was tested and rejected: the entire code-side contribution is −0.92 ms, an order of magnitude too small to account for the gap that mattered. That gap was environmental.

---
