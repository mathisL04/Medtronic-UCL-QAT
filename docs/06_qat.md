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

**vs the precision bars (deployable pycocotools, for reference):**
```text
FP32 / FP16 engine   0.7747 / 0.7748   accuracy leader
V6 QAT INT8          0.7593*           beats PTQ, ~0.015 below FP16   (*fake-quant metric; engine pending)
V4 PTQ INT8          0.7571            the bar V6 CLEARED (+0.0022)
V5 QAT INT8 (engine) 0.7437            undertrained
```

Takeaways: **~25 epochs is the real convergence point** (V5's fixed 10 was
undertrained). **V6 recovered accuracy past PTQ** — the win QAT is meant to deliver —
but stays below FP16/FP32, as expected for 8-bit vs 16-bit. Early stopping found the
plateau automatically instead of guessing the epoch count.

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

INT8 kernel (1.429 ms) > FP16 (1.137 ms). Settled: **NOT FP32 fallback** (rebuilding
INT8+FP16 moved 56 layers FP32->FP16 but the kernel didn't change, 1.429->1.440);
**explicit-quantization semantics** forbid TensorRT from swapping Q/DQ-dictated INT8
for faster FP16; the **real cause is Q/DQ reformat overhead on a launch-bound tiny
model** (YOLO26n 2.5M params — INT8's arithmetic edge is ~nil, reformats add cost).
**On A100, FP16 is the right precision for this model**; INT8/QAT pays off on
INT8-accelerated edge HW (Jetson/DLA) or larger compute-bound models.

## Open items

- **Export V6** best state -> INT8 engine -> confirm the deployable pycocotools mAP
  (~0.759 expected) and lock in the "beats PTQ" result on the same metric basis.
- Try **histogram/percentile calibration** (vs MaxCalibrator); consider reducing warmup.
- INT8+FP16 build engine kept in scratch (does not help latency; not shipped).
