# Quantisation-Aware Training (V5)

QAT fine-tunes the converged V1 baseline (`best.pt`) with fake-quant nodes so the
INT8 scales are **learned**, not just calibrated (PTQ). The goal is INT8 accuracy
that approaches FP16/FP32 — recovering the accuracy PTQ loses.

## Where QAT runs, where it is measured

```text
train:    PyTorch  -- fake-quant / Q-DQ nodes, optimiser + labelled data
convert:  PyTorch -> ONNX (Q/DQ) -> TensorRT INT8 engine
measure:  TensorRT -- accuracy (mAP) and latency
```

QAT is a training-time technique: it runs in PyTorch, never on TensorRT. TensorRT
is inference-only — its job is to fold the learned scales into a real INT8 engine.
We do **not** "train on TensorRT".

## Method

**1. Fake quantisation.** `mtq.quantize(model, INT8_DEFAULT_CFG, forward_loop)`
inserts 608 `TensorQuantizer` nodes (253 active with scales) that simulate INT8 in
the forward pass. Backprop uses the **Straight-Through Estimator** (gradient passes
through the round as identity) so the network can train through the quantisers. The
fake-quant is kept ON during validation too, so the per-epoch mAP predicts the
deployed engine.

**2. Warm-start calibration.** Before fine-tuning, quantiser ranges are initialised
from **128 train frames (seed 42, one per episode)** — NVIDIA's step 1. It is a warm
start, not PTQ: training then adapts the scales over the whole train split.

**3. Fine-tuning from a converged model.** The weights are already trained, so a
**low, decaying LR** (`lr0=1e-3`, ~1% of V1's 0.01; `lrf=0.01` anneals to ~1e-5).
A high LR would retrain rather than adapt.

**4. Early stopping with patience** (`PATIENCE` env knob, default 0 = off).
`EPOCHS` becomes a **ceiling**; Ultralytics' `EarlyStopping` stops after `PATIENCE`
epochs with no improvement in the monitored metric, and the best-epoch checkpoint is
kept. In this Ultralytics (8.4.90) the detection `fitness` weights are `[0,0,0,1.0]`
= **pure mAP50-95**, so both the early-stopping and the best checkpoint track
**mAP50-95** (not the old blended fitness, not mAP50). The rule is "no NEW best for
`PATIENCE` epochs", which tolerates the noisy per-epoch wobble a naive
"consecutive-equal" test could not.

## The process (end to end)

```text
best.pt ─(train_qat.py)→ qat_modelopt_state_best.pt   [PyTorch, fake-quant, mto.save]
        ─(export_qat_onnx.py)→ best_qat.onnx          [Q/DQ ONNX, Py3.11 venv]
        ─(build_tensorrt_int8_qdq.py)→ best_qat_int8.engine  [INT8, no calibrator, TRT venv]
        ─(evaluate_engine_map.py / benchmark_latency_trt.py)→ mAP + latency
```

## Running the conversion (commands)

The QAT model uses **dedicated** export/build scripts — NOT the baseline
`export_onnx.py` / `build_tensorrt_engine.py`. Reason (see "Export recipe" below):
the QAT model is a modelopt fake-quant state and its ONNX carries Q/DQ nodes, so it
needs modelopt's blessed export and a calibrator-free INT8 build. Defaults point at
`models/yolo26n_sanoscience_full_left/qat/v6_final/`; override with the env vars shown.

```bash
M=models/yolo26n_sanoscience_full_left/qat/v6_final

# 1) QAT state -> Q/DQ ONNX      (Py3.11 export venv; CPU is fine -- it is a trace)
DEVICE=cpu ~/venvs/medtronic-qat-p311/bin/python scripts/export/export_qat_onnx.py
#   in:  $M/qat_modelopt_state_best.pt   (env QAT_STATE)   out: $M/best_qat.onnx (env OUT_ONNX)

# 2) Q/DQ ONNX -> INT8 engine    (TensorRT venv; IDLE GPU -- the build autotunes on the live GPU)
DEVICE=<idle_gpu> ~/venvs/medtronic-trt/bin/python scripts/tensorrt/build_tensorrt_int8_qdq.py
#   in:  $M/best_qat.onnx (env ONNX_PATH)   out: $M/best_qat_int8.engine (env ENGINE_PATH)

# 3) measure (SHARED scripts -- these work on any engine, baseline or QAT)
MODE=engine ENGINE_PATH=$M/best_qat_int8.engine DEVICE=<gpu> \
  ~/venvs/medtronic-trt/bin/python scripts/evaluate/evaluate_engine_map.py            # mAP (pycocotools)
ENGINE_PATH=$M/best_qat_int8.engine DEVICE=<idle_gpu> BENCHMARK_REPEATS=10 \
  ~/venvs/medtronic-trt/bin/python scripts/benchmark/benchmark_latency_trt.py         # latency
```

Baseline vs QAT scripts (same idea, different mechanism for the quantised model):

```text
step            baseline (best.pt)                 QAT (v6 state)
export -> ONNX  scripts/export/export_onnx.py      scripts/export/export_qat_onnx.py
build  -> eng   scripts/tensorrt/build_tensorrt_engine.py   scripts/tensorrt/build_tensorrt_int8_qdq.py
measure         scripts/evaluate/* + scripts/benchmark/*    (shared -- same scripts)
```

### Why the export takes TWO inputs (best.pt + the QAT state)

`export_qat_onnx.py` fetches **both** the baseline `best.pt` *and* the QAT state. This
is not a duplicate -- the QAT state is a modelopt **state (a recipe + weights), not a
standalone loadable model**, so it has to be replayed onto a base architecture:

```python
model = YOLO(best.pt).model      # 1. build the ARCHITECTURE (structure) from best.pt
mto.restore(model, qat_state)    # 2. replay onto it: RE-INSERT the 608 quantizers +
                                 #    LOAD the fine-tuned weights + learned INT8 scales
```

Think of it as a puzzle: **best.pt is the frame + original pieces; the QAT state (a)
ADDS new pieces -- the 608 quantiser nodes best.pt never had -- and (b) SWAPS the
weights for the fine-tuned ones + the learned scales.** So best.pt supplies the
*skeleton*, the QAT state supplies the *quantisers and the trained weights/scales*;
best.pt's own weights are loaded then overwritten. Neither file alone is enough.

Why modelopt works this way: the quantised model uses dynamically-generated classes
(`QuantConv2d`, ...) that Python's pickler cannot reload standalone -- which is why
`train_qat.py` disables Ultralytics' native `save_model` and uses `mto.save`/`restore`.
Saving the state and replaying it onto a fresh base model sidesteps that entirely.

## Parameters and setup

```text
lr0      1e-3     low fine-tune LR (env LR0)             amp      False  (fake-quant scales are FP32)
lrf      0.01     decay tail -> ~1e-5 (env LRF)          batch    16
EPOCHS   ceiling  (env; V5=10 fixed, V6=50)              imgsz    640    LOCKED (stride 32 + engine input)
PATIENCE 0=off    early-stop epochs (env; V6=10)         workers  0      (fork/overcommit guard)
quant    INT8_DEFAULT_CFG  608 quant / 253 scales        warmup   3      (Ultralytics default)
calib    128 train frames, seed 42 (MaxCalibrator)       device   single A100, Geneva (env DEVICE)
```

Config is by env knobs (repo convention): `EPOCHS PATIENCE LR0 LRF DEVICE RUN_NAME
WORKERS ...`. Structural paths (model, dataset) are hardcoded.

## Training / conversion scripts — what each does

```text
scripts/train/train_qat.py                 The QAT fine-tune. QATTrainer(DetectionTrainer): get_model()
                                     override inserts fake-quant BEFORE ModelEMA; warm-start calib;
                                     best-epoch checkpoint callback (mto.save on mAP50-95 improve);
                                     reload gate (608 quantisers round-trip); PATIENCE early-stop.
scripts/train/qat_run.sh                   Runner. Checkpoints to /tmp scratch (off the 50GB NFS quota),
                                     copies durable artifacts back to NFS on exit (best state +
                                     provenance + results). PY env override selects the venv.
scripts/export/export_qat_onnx.py           QAT state -> Q/DQ ONNX via modelopt get_onnx_bytes_and_metadata
                                     + Detect.export head mode (single [1,3,640,640]->[1,300,6]).
scripts/tensorrt/build_tensorrt_int8_qdq.py   Q/DQ ONNX -> INT8 .engine. Explicit quantization: sets the INT8
                                     flag, TensorRT reads scales from Q/DQ nodes, NO calibrator.
scripts/evaluate/evaluate_engine_map.py       Engine mAP (pycocotools, full val or val100).
scripts/benchmark/benchmark_latency_trt.py     Engine latency: CUDA-event kernel + pipeline, idle-gated.
scripts/benchmark/benchmark_latency_pytorch.py   PyTorch raw-model latency (fp32/fp16/qat
                                     fake-quant), reference only -- not deployment.
```

## Environment (V5 stack) -- a deliberate migration

```text
V2-V4 (TensorRT stages):  ~/venvs/medtronic-trt        Py3.9,  TensorRT 10.16.1.11
V5 QAT train + export:    ~/venvs/medtronic-qat-p311   Py3.11, torch 2.7.0+cu128, modelopt 0.33.1
(superseded)              ~/venvs/medtronic-qat         Py3.9,  modelopt 0.29.0
```

Forced, not incidental: modelopt 0.29 **cannot export** this model to Q/DQ ONNX
under any torch; modelopt >= 0.31 fixes it but needs Python >= 3.10, and 0.29 was the
last Py3.9 build. `train_qat.py` runs UNCHANGED under 0.33.1 (same 608/253, reload
gate passes) — the migration touched the environment, not the training logic.

## Results -- V5 (fixed-10) vs V6 (early-stopping)

Two runs, identical except the epoch/patience regime. Metrics are the **Ultralytics
fake-quant validation mAP** per epoch (full val, conf 0.001).

```text
              regime                 epochs   runtime   best mAP50-95   stop reason
V5   EPOCHS=10 (fixed)                 10      1h41m     0.7427 @ ep10   hit fixed limit (still climbing)
V6   EPOCHS=50, PATIENCE=10 (early)    35      6h41m     0.7593 @ ep25   PLATEAU (patience fired)
```

**Fluctuation (mAP50-95):**
```text
       best         min     max     mean    std     range
V5   0.7427@ep10   0.6614  0.7427  0.7091  0.0256  0.0813
V6   0.7593@ep25   0.6096  0.7593  0.7107  0.0346  0.1496   (noisier: LR stretched over 50-ep schedule)
```

**V6 engine — MEASURED (deployable, full val pycocotools + CUDA-event latency):**
```text
                     mAP50    mAP50-95    kernel(GPU)    total
FP32 engine          0.9325   0.7747       2.019 ms      3.736 ms   accuracy leader
FP16 engine          0.9327   0.7748       1.137 ms      2.922 ms   accuracy + speed leader (A100)
V6 QAT INT8 engine   0.9321   0.7644       1.397 ms      3.421 ms   <- MEASURED
V4 PTQ INT8          0.9282   0.7571        —             —         the bar V6 CLEARED (+0.0073)
V5 QAT INT8 engine   0.9283   0.7437       1.386 ms      3.789 ms   undertrained
```

Takeaways: **~25 epochs is the real convergence point** (V5's fixed 10 was
undertrained). **V6 recovered INT8 accuracy to 0.7644 — beats PTQ (+0.0073) and lands
just ~0.010 below FP16/FP32**, the accuracy win QAT is meant to deliver. Early stopping
found the plateau automatically instead of guessing the epoch count. The engine (0.7644)
sits a touch above the fake-quant training metric (0.7593) — normal pycocotools-vs-
Ultralytics noise; the engine faithfully reproduces the model. **Latency is unchanged
from V5 (~1.4 ms kernel, graph-determined) — still slower than FP16 on A100** (Q/DQ
reformat overhead; INT8's speed payoff is on edge HW / DLA, not A100).

**Evolution graph:** `reports/qat_training/qat_v5_v6_accuracy.png`
(mAP50-95 vs epoch, both runs, best epochs marked, patience window shaded, precision
bars overlaid). Full per-epoch tables + stats: `reports/qat_training/README.md`.

**Artifacts (force-added to git under `models/yolo26n_sanoscience_full_left/`):**
```text
qat/v6_final/qat_modelopt_state_best.pt   ep25 best (the V6 deployable model) + state/results/args
qat/v5_10ep/                              the V5 run + its Q/DQ ONNX + INT8 engine + sidecars
qat/smoke_1ep/                            the 1-epoch smoke
(training writes raw output to runs_qat/<RUN_NAME>/ which is gitignored; the best of each
 run is curated into models/.../qat/<run>/)
```

**Latency** is unchanged between V5 and V6 (graph-determined, not weight-dependent):
INT8 kernel ~1.39-1.43 ms, ~57x faster than the PyTorch fake-quant forward (~80 ms).
Full distributions: `reports/v5_latency/`.

## QAT Iteration 2 — OFAT hyperparameter sweep + build-side latency recovery

Two results, both in `experiments/qat_iteration_2/`: (1) an OFAT sweep over the QAT
knobs that produced a model **beating every precision on accuracy**, and (2) a
build-only rebuild that **recovered most of QAT's latency penalty at zero accuracy
cost** — no retraining in either the sweep-analysis or the recovery.

### The sweep — 11 runs, verified true OFAT

All from the V6 defaults (baseline = V6: `EPOCHS=50 PATIENCE=10 WORKERS=0`, full-val
eval), each moving exactly ONE knob. **OFAT verified from ground truth** (`args.yaml`
+ `qat_provenance.json`, not just the launcher config): every run moved exactly one
knob, all cross-run invariants (epochs/patience/imgsz/amp/seed=0/calib=42) constant,
each a real QAT model (253 quantisers-with-scales / 608 inserted; `disable_attention`
correctly drops to 221). No confounded runs.

```text
knob (vs V6 default)   mAP50-95   note
batch=32               0.7801     BEST OF ANY PRECISION — beats FP32 0.7747, FP16 0.7748
lrf=0.1                0.7671
lr0=1e-2               0.7647     LR0 ~invariant: early-stop normalises the schedule
lr0=1e-3 (=V6)         0.7644     (baseline point)
lr0=1e-4/3e-4/3e-3     0.7644     all ~0.7644 -> LR0 has no lasting effect
n_calib=32 / 512       ~0.7608    N_CALIB ~invariant (warm start only)
disable_attention      0.7580     disabling model.10/22 quant slightly WORSE + engine bloat
batch=8                0.7318     small batch hurts
lrf=0.001              0.7262     too little LR decay hurts
```

**Winner: `batch=32` -> mAP50-95 0.7801.** Confirmed LR-robust: a rescaled
`batch=32, lr0=2e-3` run gave 0.7799 (identical). Original-build latency was
**invariant across every knob (~1.39 ms)** — expected, since the knobs change weight
*values*, not graph *structure*, and latency depends on structure.

Master table + per-knob plots: `experiments/qat_iteration_2/sweeps/master_comparison.md`
and `*_sweep/`.

### Why QAT's engine was slower than PTQ (engine verification)

Full per-layer audit in `experiments/qat_iteration_2/engine_verification/`. Measured
clean on an exclusive idle A100:

```text
engine            layers   INT8   latency    note
PTQ INT8+FP16       189     149   1.091 ms   implicit quant, TensorRT places boundaries optimally
V6 QAT INT8         253      —    1.398 ms   explicit Q/DQ -> ~70 more kernels, worse fusion
FP16                231      —    1.153 ms
FP32                355      —    2.024 ms
```

Root cause is **launch-bound**: YOLO26n (2.5M params) never saturates the A100 at
batch 1, so latency tracks *number of kernel launches*, not arithmetic. QAT's explicit
Q/DQ graph fuses worse than PTQ's implicit quantisation -> ~70 extra launches -> the
~0.3 ms gap. (INT8 is only ~5% faster than FP16 here — the fingerprint of launch-bound.)

### The recovery — a smarter build, no retraining

The production QAT build (`build_tensorrt_int8_qdq.py`) sets only `INT8`. Rebuilding
the SAME `best_qat.onnx` with two extra builder flags recovers most of the gap:

```text
change              engine effect                        latency (V6)
INT8 only (prod)    112 layers FP32, 77 reformats        1.398 ms
+ FP16 flag         non-INT8 layers FP32->FP16           1.341 ms   (alone: small)
+ opt-level 5       max fusion search, reformats 77->15  1.362 ms   (alone: small)
+ FP16 + opt5       BOTH — the synergy                   1.197 ms   <- accuracy 0.7639 (held)
```

Neither flag alone helps; **together they recover ~65% of the penalty** (1.398 ->
1.197 ms) at **no accuracy cost** (V6 0.7644 -> 0.7639, −0.0005). This **supersedes**
the earlier "rebuilding INT8+FP16 didn't change the kernel" note below — that tested
FP16 *alone*; the opt-level-5 pairing was the missing lever.

### FINAL before/after — all 11 sweep models rebuilt (FP16+opt5)

Latency re-timed in ONE exclusive idle-GPU session (median spread 0.038 ms);
accuracy full-val 6449-img pycocotools; disk/dev-mem static engine props.
Full table: `experiments/qat_iteration_2/rebuild_fp16_opt5/BEFORE_AFTER.md`.

```text
model              INT8  old mAP  new mAP   old ms  new ms(clean)  disk MB  dev-mem MB
batch_32 (WINNER)   143  0.7801   0.7797    1.386   1.200          4.79     9.7
lrf_0.1             143  0.7671   0.7671      —     1.226          4.80     9.9
lr0_1e-2            143  0.7647   0.7641    1.560   1.203          4.81     9.8
lr0_3e-3            144  0.7644   0.7643      —     1.209          4.78     9.9
lr0_1e-4            143  0.7644   0.7640    1.381   1.214          4.86     9.8
lr0_3e-4            143  0.7644   0.7641    1.382   1.212          4.78     9.7
ncalib_32           144  0.7609   0.7610    1.518   1.207          4.81     9.8
ncalib_512          144  0.7608   0.7614      —     1.238          4.79     9.9
disable_attention   117  0.7580   0.7588    1.381   1.226          5.55     9.9
batch_8             143  0.7318   0.7305    1.391   1.218          4.79     9.9
lrf_0.001           143  0.7262   0.7259      —     1.219          4.85     9.7
```

Every model: INT8 core unchanged (143-144; `disable_attention` 117 by design),
accuracy flat (±0.001), latency down ~0.16-0.19 ms to a tight 1.20-1.24 ms band,
footprint ~4.8 MB disk / ~9.8 MB device scratch.

### PROVEN / INFERRED / open

```text
PROVEN
  - Clean OFAT across all 11 runs (ground-truth args.yaml + provenance).
  - batch=32 = 0.7801 mAP50-95, best of any precision, LR-robust (0.7799 rescaled).
  - QAT engine slower than PTQ because it has ~70 more kernels (253 vs 189), launch-bound.
  - FP16+opt5 rebuild: 1.40 -> 1.20 ms, accuracy-neutral, holds across all 11 models.
INFERRED
  - The ~70 extra layers are diffuse un-fused Q/DQ-boundary ops spread across the
    backbone/neck (model.2,4,6,8,13,16,19) — NOT concentrated hot-spots, and NOT
    reformats (rebuilt QAT has FEWER reformats than PTQ: 24 vs 45).
  - ~44 of the non-INT8 layers are INHERENT float (attention model.10/22, head
    model.23, NMS) — unrecoverable; PTQ has them too.
OPEN (only if the last ~0.11 ms is deployment-critical)
  - Q/DQ re-placement at EXPORT to reduce the diffuse overhead toward PTQ's 189
    layers. High-effort, diffuse, uncertain (see diagnosis in this experiment's
    thread); ceiling is MATCHING PTQ (~1.08 ms), not beating it.
  - CUDA graphs — cheaper alternative, attacks launch overhead directly, zero
    accuracy risk; try before any Q/DQ surgery.
```

### Deployment candidate

```text
QAT batch_32, rebuilt FP16+opt5:  mAP50-95 0.7797  @  1.200 ms  (4.79 MB disk / 9.7 MB scratch)
  = best accuracy of ANY precision, ~0.12 ms above the PTQ floor (1.082 ms), no retraining.
```

The "QAT is ~0.3 ms slow" story was ~60% a build artifact; recovered to ~0.12 ms.
If latency is the sole objective and accuracy is negotiable, PTQ (1.082 ms / 0.7564)
remains the floor — QAT's value is the accuracy, delivered at competitive latency
once built correctly.

## Export recipe (QAT fake-quant -> Q/DQ ONNX -> TensorRT INT8)

Hard-won and fragile. **Root cause of the original failure — the API, not torch.**
`torch.onnx.export()` under `export_torch_mode()` FAILS: the quantiser scale `_amax`
traces as a graph input where modelopt's INT8 symbolic needs an `onnx::Constant`
(`SymbolicValueError: got 'prim::Param'`) — identical on torch 2.6/2.7/2.8 and both
exporter backends. **The fix is modelopt's blessed API:**

```text
from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata
payload, _ = get_onnx_bytes_and_metadata(model, dummy, onnx_opset=17)
```

**Exact pinned deps (each fixes a real failure):**
```text
modelopt 0.33.1 (>=0.31, needs Py>=3.10) · torch 2.7.0 · onnx 1.17.0 (1.22 removes
onnx.reference.custom_element_types -> import error) · numpy <2 (2.x breaks modelopt's
importer) · onnxruntime, onnx_graphsurgeon, polygraphy  (modelopt [onnx] extra, pypi.nvidia.com)
```

**Head export mode** (`Detect.export=True`) collapses YOLO26n's 11-tensor forward to
the single `[1,3,640,640]->[1,300,6]` graph; the Q/DQ count drops 255->207 (the
one2many auxiliary heads and their quantisers are pruned — correct). **TensorRT INT8
build is calibrator-free** (explicit Q/DQ; `set_flag(INT8)`, scales read from nodes).

## Latency analysis -- why INT8 is NOT faster than FP16 here

> **Refined in Iteration 2 (see above).** The "rebuilding INT8+FP16 didn't change the
> kernel" claim below tested the FP16 flag *alone*. Pairing FP16 with builder
> `optimization_level=5` DOES recover latency (1.40 -> 1.20 ms, accuracy-neutral). The
> residual gap to PTQ is diffuse un-fused Q/DQ overhead, not FP32 fallback. Read the
> Iteration 2 section as the current conclusion; the paragraph below is retained as the
> original V6 reasoning.

INT8 kernel (1.429 ms) > FP16 (1.137 ms). Settled: **NOT FP32 fallback** (rebuilding
INT8+FP16 moved 56 layers FP32->FP16 but the kernel didn't change, 1.429->1.440);
**explicit-quantization semantics** forbid TensorRT from swapping Q/DQ-dictated INT8
for faster FP16; the **real cause is Q/DQ reformat overhead on a launch-bound tiny
model** (YOLO26n 2.5M params — INT8's arithmetic edge is ~nil, reformats add cost).
**On A100, FP16 is the right precision for this model**; INT8/QAT pays off on
INT8-accelerated edge HW (Jetson/DLA) or larger compute-bound models.

## Open items

- **DONE** — V6 exported -> INT8 engine -> measured: mAP50-95 **0.7644** (beats PTQ),
  kernel **1.397 ms**. All artifacts committed under `models/.../qat/v6_final/`.
- **DONE (Iteration 2)** — OFAT sweep (11 runs, verified clean): **batch=32 -> 0.7801**,
  best of any precision. Build-side latency recovery **1.40 -> 1.20 ms** (FP16+opt5),
  accuracy-neutral, across all 11 models. Deployment candidate: batch_32 rebuilt =
  **0.7797 @ 1.200 ms**. See the Iteration 2 section + `experiments/qat_iteration_2/`.
- Try **histogram/percentile calibration** (vs MaxCalibrator); the framework supports it
  (`CALIB_METHOD`), validated by the 1-epoch percentile smoke.
- **CUDA graphs** — untested launch-overhead lever; cheapest path to close the residual
  ~0.11 ms to PTQ, zero accuracy risk. Try before any Q/DQ-placement surgery.
