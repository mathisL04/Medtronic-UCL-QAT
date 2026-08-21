# Medtronic × UCL — QAT Research

Code, configs, baseline model and experiment outputs for the Medtronic × UCL summer
research project on optimising a **YOLO26n surgical-tool detector** for low-latency
TensorRT deployment.

The question the project set out to answer: **how far can INT8 quantisation be pushed
before accuracy breaks, and does quantisation-aware training (QAT) actually pay for
itself over post-training quantisation (PTQ)?**

Short answer, established over four stages: **QAT wins on accuracy and loses on
latency.** On this model the latency loss is a build-and-launch artifact, not an
arithmetic one, and most of it is recoverable. Detail below.

```text
Open-H Sanoscience surgical videos
        ↓  automatic segmentation-based YOLO labelling
YOLO26n baseline training (UCL Cork cluster)
        ↓  ONNX export (opset 17, parity-gated)
TensorRT FP32 (V2)  →  FP16 (V3)  →  INT8 PTQ (V4)
        ↓
QAT (V5 → V6 → Iteration-2 OFAT sweep)
        ↓  latency dissection: per-kernel profiling, fusion analysis, CUDA graphs
Week 8: per-layer QAT sensitivity sweep (freeze-all-but-one)
```

---

## 1. Infrastructure

Two shared multi-GPU boxes. **Which box a number came from matters** — see §2.

| host | GPUs | used for |
|---|---|---|
| `geneva.ee.ucl.ac.uk` | 4× A100-SXM4-80GB | FP32/FP16/PTQ baselines, QAT V5/V6, Iteration-2 sweep, per-kernel profiling |
| `malmo.ee.ucl.ac.uk` | 4× H100 NVL (95 GB), driver 595.71.05 | Week-8 per-layer sweep (train + deploy) |

Storage: NFS home is capped at a **50 GB quota** and runs near-full. On malmo, `/tmp`
is local NVMe with ~1.3 TB free — Ultralytics `PROJECT` points there during training
and only small final artifacts are copied back. Engines and ONNX are gitignored and
rebuildable; their sha256 lives in the provenance sidecars.

### Environments

Four venvs, deliberately split because the QAT and TensorRT stacks have incompatible pins.

| venv | Python | key packages | role |
|---|---|---|---|
| `~/venvs/medtronic-qat-p311` | 3.11.13 | `nvidia-modelopt 0.33.1`, `torch 2.7.0+cu128`, `onnx 1.17.0`, `numpy 1.26.4`, `ultralytics 8.4.90`, `pycocotools 2.0.11` | QAT fine-tune + Q/DQ ONNX export |
| `~/venvs/medtronic-trt` | 3.9.25 | `tensorrt 10.16.1.11`, `pycocotools 2.0.11` | engine build, mAP eval, latency |
| `~/venvs/medtronic-qat`, `~/venvs/medtronic-qats` | 3.9.25 | `nvidia-modelopt 0.29.0`, `torch 2.8.0` | legacy / pre-0.33 experiments |

**The pins are load-bearing** — each fixes a real failure:

```text
modelopt >= 0.31   requires Python >= 3.10 (hence the p311 venv)
onnx == 1.17.0     1.22 removes onnx.reference.custom_element_types -> import error
numpy < 2          2.x breaks modelopt's importer
torch 2.7.0+cu128  matched to the modelopt build
```

**Export gotcha (cost real time).** `torch.onnx.export()` under `export_torch_mode()`
fails: the quantiser scale `_amax` traces as a graph input where modelopt's INT8
symbolic needs an `onnx::Constant` (`SymbolicValueError: got 'prim::Param'`) —
identical on torch 2.6/2.7/2.8 and both exporter backends. The working path is
modelopt's own API:

```python
from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata
payload, _ = get_onnx_bytes_and_metadata(model, dummy, onnx_opset=17)
```

---

## 2. How everything is measured

The single most important thing in this repo. Numbers from different harnesses are
**not** interchangeable, and mixing them silently manufactures or hides quantisation
loss.

### Accuracy — mAP50-95

```text
metric      pycocotools COCOeval bbox   (NOT Ultralytics val)
eval set    full val, 6,449 images, 14,517 GT boxes
conf        0.001                       (NOT 0.25)
max_dets    300
imgsz       640
measured    through the TensorRT engine, post-deployment — not on the PyTorch model
```

Two rules that were learned the hard way:

- **`conf=0.001`, always.** `conf=0.25` truncates the PR curve and depresses mAP by
  roughly 5 points. It does *not* move P/R (Ultralytics reports those at max-F1, not
  at the conf floor). Any baseline-vs-quantised comparison at mismatched conf
  fabricates quantisation loss.
- **One metric implementation per comparison.** pycocotools and Ultralytics differ by
  ~0.004 mAP50 on the *same* ONNX. Every accuracy figure in this README is
  pycocotools, so the whole ladder is internally comparable.

Every measurement writes a `*.map_full.json` sidecar recording the engine sha256, eval
set, conf, and the full COCO stats vector.

### Latency — pure GPU compute time

The headline latency metric is **kernel-median GPU compute time at batch=1**, not
end-to-end wall time:

```text
tool        trtexec 10.16.1.11
flags       --iterations=300 --warmUp=100 --duration=0 --avgRuns=10
            --noDataTransfers --useSpinWait  [--useCudaGraph]
statistic   median of per-iteration GPU time
gating      EXCLUSIVE, verified-idle GPU
```

`--noDataTransfers` is what makes it *pure compute* — H2D/D2H are excluded, so the
number reflects the model and the engine, not the PCIe path or the host.

**Idle-gating is compute-only.** These boxes always have Xorg holding a graphics
context (~4 MiB, NVML type G). The gate checks
`nvmlDeviceGetComputeRunningProcesses` and ignores graphics contexts, otherwise an
idle box is wrongly refused. Utilisation is sampled several times over ~1 s, and the
gate runs *before* `torch.cuda.set_device`. `benchmark_latency.py` also snapshots GPU
state per repeat to flag mid-run contention.

**Why this rigour:** an early FP32 latency of ~17.8 ms and a later ~8.64 ms came from
the *same code*. The difference decomposed as **environment −8.39 ms** (idle GPU +
driver reboot) versus **the entire code bundle −0.92 ms** (`set_num_threads(1)` +
warmup + `cudnn.benchmark`) — an order of magnitude apart. Latency on a shared box is
a property of the box first and the model second. Report only from a verified-idle GPU.

---

## 3. The precision ladder

All accuracy on the harness in §2. Latency is kernel-median, batch=1, exclusive idle
**A100**, no CUDA graph.

| stage | precision | mAP50 | mAP50-95 | kernel ms | engine layers | note |
|---|---|---|---|---|---|---|
| PyTorch baseline | FP32 | — | — | 8.642 (e2e) | — | reference implementation |
| **V2** | FP32 engine | 0.9325 | **0.7747** | 2.019 | 355 | accuracy reference |
| **V3** | FP16 engine | 0.9327 | **0.7748** | **1.137** | 231 | free — accuracy-neutral, 1.78× faster |
| **V4** | INT8 PTQ | 0.9282 | 0.7571 | **1.091** | 189 | −0.0176 mAP50-95 vs FP32; the latency floor |
| **V6** | INT8 QAT | 0.9321 | 0.7644 | 1.397 | 253 | beats PTQ on accuracy, loses on latency |
| **Iter-2 best** | INT8 QAT (batch=32) | **0.9437** | **0.7801** | 1.386 | 245 | best accuracy of *any* precision |
| **Deployment candidate** | INT8 QAT, rebuilt FP16+opt5 | 0.9438 | **0.7797** | **1.200** | 245 | best accuracy, ~0.12 ms off the PTQ floor |

> **Latency figures vary by ~0.02–0.06 ms between measurement sessions**, even on an
> exclusive idle GPU — the PTQ engine has been timed at 1.091, 1.082 and 1.072 ms, and
> FP16 at 1.137 and 1.153 ms, in different sessions. Comparisons are only safe *within*
> a session; the table above is the primary session. This is why the Week-8 sweep timed
> all 18 engines in one window.

Read the two axes separately — they tell opposite stories.

**Accuracy: QAT works.** PTQ costs −0.0176 mAP50-95 against FP32. QAT recovers most of
it (V6, −0.0103), and the Iteration-2 sweep overtakes FP32 outright (0.7801 vs 0.7747).
FP16 is essentially free (+0.0001 over FP32) and should be the default whenever INT8
isn't required.

**Latency: QAT hurts.** The QAT INT8 engine (1.397 ms) is *slower than both* the PTQ
INT8 engine (1.091 ms) and plain FP16 (1.137 ms). INT8 buys only ~5% over FP16 even in
the PTQ case. That is the fingerprint of a **launch-bound** model: YOLO26n has 2.5 M
parameters and never saturates an A100, so kernel count dominates and INT8's arithmetic
advantage is ~nil.

---

## 4. QAT — method and the patience regime

QAT is a **fake-quantisation fine-tune** of the converged FP32 baseline: modelopt
inserts quantiser nodes (`INT8_DEFAULT_CFG`, 608 quantizers, 253 carrying scales),
warm-starts their ranges from 128 episode-diverse calibration frames (seed 42,
`max` method), then trains normally so the weights adapt to their own quantisation error.

```text
epochs        50 (ceiling, not a target)
patience      10          <- early stopping on val mAP50-95 plateau
lr0           1e-3        (1% of the baseline's 0.01 — the model is already converged)
lrf           0.01
batch         16 (V6) / 32 (Iteration-2 winner)
imgsz         640
amp           False       <- deliberate deviation, see below
val           full 6,449-image val set, every epoch
```

**The patience method.** Runs are given a generous 50-epoch ceiling and stopped by
`patience=10` on the validation-fitness plateau rather than at a fixed epoch count.
QAT from a converged checkpoint converges fast and then drifts; a fixed budget either
truncates a still-improving run or spends hours past the peak. Early stopping makes
the budget adaptive and — because best-checkpoint selection is driven by that same
validation fitness — keeps the selected checkpoint at the actual peak.

**Three hazards, all deliberate deviations, all documented:**

1. **`amp=False`.** Ultralytics enables AMP by default. Fake-quant inserts explicit
   quantise/dequantise ops whose scales are FP32; running them inside an FP16 autocast
   region mixes dtypes at the quantiser boundary and is a known source of wrong scales.
   The baseline *was* trained with AMP, so every QAT-vs-baseline comparison carries
   `amp=False` as a known difference.
2. **`save=False`.** Ultralytics' `save_model()` pickles the live module; modelopt
   generates `QuantConv2d` as a *dynamic* class that pickle cannot resolve, so native
   checkpointing raises at the end of every epoch. The durable checkpoint is written by
   `mto.save()` from an `on_fit_epoch_end` callback instead, which snapshots whenever
   validation fitness improves.
3. **Head export mode.** `Detect.export=True` collapses YOLO26n's 11-tensor forward to
   a single `[1,3,640,640] → [1,300,6]` graph; the Q/DQ count drops 255 → 207 as the
   one2many auxiliary heads and their quantisers are pruned. Correct, but it means the
   exported graph is not the training graph.

The TensorRT INT8 build is **calibrator-free**: explicit Q/DQ, `set_flag(INT8)`, scales
read directly from the `QuantizeLinear`/`DequantizeLinear` nodes.

### Iteration 2 — OFAT sweep

Eleven one-factor-at-a-time runs over `lr0`, `lrf`, `batch`, `n_calib` and attention
quantisation, each verified clean against its ground-truth `args.yaml`.

```text
batch=32          0.7801   <- winner, best of any precision; LR-robust (0.7799 rescaled)
lrf=0.1           0.7671
lr0 (4 values)    0.7640 - 0.7647   invariant
n_calib 32/512    0.7609 / 0.7614
disable attention 0.7580
batch=8           0.7318
lrf=0.001         0.7262
```

Batch size was the only knob that mattered much; learning rate was essentially
invariant across two orders of magnitude. **Kernel latency was invariant across every
knob (~1.39 ms)** — expected, since these knobs change weights, not graph structure.

---

## 5. Dissecting the QAT latency regression

The ~0.3 ms QAT-vs-PTQ gap was not accepted as a property of QAT. It was profiled.

**Tooling:** `trtexec --exportProfile --exportLayerInfo --profilingVerbosity=detailed
--separateProfileRun --noDataTransfers` on an exclusive idle A100, cross-checked against
the TensorRT Python `IProfiler`. The exported JSONs were parsed per kernel, tagged by
region (backbone / neck / attention / head / NMS) and by precision, and ranked by time —
see `qat_vs_ptq_kernel_profiling.ipynb`, `Per_kernel_json_file/` and
`experiments/qat_iteration_2/profiling_exports/`.

**What the profiles showed:**

```text
engine                layers  reformats  kernel ms
PTQ INT8 + FP16          189        149     1.091     implicit quant, TRT places boundaries optimally
V6 QAT INT8              253          —     1.398     explicit Q/DQ -> ~70 more kernels
FP16                     231          —     1.153
FP32                     355          —     2.024
```

The QAT engine carries **~70 more kernels** than PTQ. The dominant single cause:
**86 standalone SiLU kernels versus PTQ's 21** — activations split *out* of their conv
fusion, accounting for the +0.127 ms.

**A controlled experiment falsified the obvious explanation.** A tiny CNN
(3× Conv3×3→SiLU + 1×1 head) was built in three variants — FP16, INT8 PTQ, INT8
explicit Q/DQ — differing *only* in quantisation scheme (`experiments/fusion_demo/`).
Explicit Q/DQ did **not** break conv+SiLU fusion: all three blocks fused in every
variant, the Q/DQ variant had *fewer* kernels than FP16 (5 vs 6) and was the *fastest*.

So "explicit Q/DQ breaks fusion" is too simple. The real-model break comes from the
**interaction of Q/DQ with structural complexity** — C2f channel splits, residual adds,
and multi-consumer tensors where a conv's SiLU output feeds branching paths. Q/DQ is
necessary but not sufficient; the branching is what tips TensorRT into leaving the SiLU
standalone.

### Two recoveries

**Build-side (FP16 + optimisation level 5).** Neither flag does much alone; together
they are a synergy:

```text
INT8 only                    112 layers FP32, 77 reformats     1.398 ms
+ FP16 flag                  non-INT8 FP32 -> FP16             1.341 ms   (alone: small)
+ opt-level 5                max fusion search, reformats 77->15  1.362 ms   (alone: small)
+ FP16 + opt5                BOTH                              1.197 ms   <- accuracy held
```

**1.40 → 1.20 ms, accuracy-neutral (−0.0005), and it held across all 11 Iteration-2
models.** About 60% of the "QAT is slow" story was a build artifact, not QAT.

**Launch-side (CUDA graphs).** Since the model is launch-bound, capturing the whole
inference as one CUDA graph attacks the actual bottleneck with zero accuracy risk. On
the Week-8 engines it is worth a consistent **21–25% (median 22.4%)**:

```text
no CUDA graph   0.9716 ms median
+ CUDA graph    0.7548 ms median
```

Both recoveries are build/runtime changes. Neither required retraining.

---

## 6. Week 8 — per-layer QAT sensitivity sweep

### Why

The Iteration-2 result was a strong *aggregate*: QAT helps accuracy, costs latency,
most of the cost is recoverable. But it said nothing about **which layers** carry the
accuracy and which carry the cost, and the per-layer signal was buried in noise —
every run differed in weights, kernel layout and build simultaneously.

The only way to isolate it was a controlled sweep: **freeze the entire network except
one layer, QAT that layer alone, deploy it, measure it — then repeat for every layer.**

### Setup

A deliberately weakened baseline was used so the effect would be visible. Starting from
COCO `yolo26n.pt`, backbone and neck were frozen (`freeze=23`) and only the head trained
on the surgical data for a fixed 50 epochs → mAP50-95 **0.5376** (vs the fully-trained
baseline's 0.7815 — both Ultralytics-metric, so comparable to each other but not to the
pycocotools column in §3). Freezing costs ~0.24 mAP50-95: the COCO backbone is not
adapted to surgical imagery. Applying QAT with **zero training iterations** (calibrate-only, "PTQ via the
QAT path") drops it another −0.057 to **0.4892** — that is the V1 reference the sweep
measures against.

> **These Week-8 numbers are a separate lineage.** They are not comparable to the §3
> ladder; the baseline is intentionally much weaker. On this frozen model, 0-iteration
> QAT is worse than FP16 on *both* axes (−0.042 mAP50-95 and +0.250 ms) — the clearest
> single demonstration that INT8 without fine-tuning, on a launch-bound model, buys
> nothing.

```text
layers        24 (model.0 .. model.23), one at a time
epochs        6, patience 3        <- uniform budget across every layer
validation    full 6,449-image val set, every epoch
build         TensorRT 10.16.1.11, explicit Q/DQ, OPT_LEVEL=3, 8 GB workspace
phases        PHASE=train (all layers) THEN PHASE=deploy (all layers)
```

**Two deliberate design choices.** The training budget was cut from 12 epochs to 6 to
fit the sweep in the window, but **the evaluation set was never cut** — per-epoch
validation stays on the full 6,449 images. The point of the sweep is a *trend across
layers*; a cheaper fitness signal makes best-checkpoint selection noisy, which perturbs
the very ranking the sweep exists to produce. And the budget is **uniform**: a layer
trained longer than its peers would rank high on budget alone, a worse confound than
any latency offset.

Training and deployment were **split into separate phases** — all 18 layers trained
first, then all engines built and measured in one batch. This banks the expensive part
first and, more importantly, puts every latency reading in a single exclusive-GPU
window instead of spreading them across ~20 h of a shared box's varying load.

### Results

Ran 2026-08-21 on malmo/H100. Train pass ended 02:57:55; deploy pass 02:59:55 → 05:01:39
on an exclusive GPU, exit 0. 18 layers trained; 6 (`model.11/12/14/15/18/21`) returned
`no_trainable_params` — the Concat/Upsample nodes, expected.

`%gap` = fraction of the V1 (0.4892) → FP32 (0.7747) gap recovered.

| rank | layer | mAP50-95 | Δ vs V1 | %gap | graph ms |
|---|---|---|---|---|---|
| 1 | model.8 | **0.6675** | +0.1785 | **62.4%** | 0.7656 |
| 2 | model.6 | 0.6525 | +0.1635 | 57.2% | 0.7592 |
| 3 | model.13 | 0.6518 | +0.1628 | 57.0% | 0.7445 |
| 4 | model.10 | 0.6301 | +0.1411 | 49.4% | 0.7521 |
| 5 | model.4 | 0.6205 | +0.1315 | 46.0% | 0.7615 |
| 6 | model.7 | 0.6168 | +0.1278 | 44.7% | 0.7553 |
| 7 | model.9 | 0.6065 | +0.1175 | 41.1% | 0.7533 |
| 8 | model.5 | 0.5994 | +0.1104 | 38.6% | 0.7443 |
| 9 | model.16 | 0.5846 | +0.0956 | 33.4% | 0.7396 |
| 10 | model.3 | 0.5769 | +0.0879 | 30.7% | 0.7568 |
| 11 | model.19 | 0.5685 | +0.0795 | 27.8% | 0.7544 |
| 12 | model.2 | 0.5568 | +0.0678 | 23.7% | 0.7580 |
| 13 | model.17 | 0.5390 | +0.0500 | 17.4% | 0.7427 |
| 14 | model.1 | 0.5343 | +0.0453 | 15.8% | 0.7582 |
| 15 | model.23 | 0.5025 | +0.0135 | 4.6% | 0.7585 |
| 16 | model.0 | 0.4982 | +0.0092 | 3.1% | 0.7437 |
| 17 | model.20 | 0.4949 | +0.0059 | 2.0% | 0.7413 |
| 18 | model.22 | 0.4874 | −0.0016 | −0.7% | 0.7566 |

**The trend the sweep existed to produce.** Mid/late backbone — `model.4`–`model.10`
plus `model.13` — carries essentially all the recoverable accuracy. The stem
(`model.0`) and the head (`model.20`, `22`, `23`) buy nothing, and `model.22` is
fractionally negative. The bottom four sit *below* the frozen FP16 baseline (0.5313):
quantising them costs more than training them back recovers.

**Latency: no signal.** CUDA-graph 0.7396–0.7656 ms, no-graph 0.9532–1.0026 ms,
1306–1352 FPS, kernel count 235–256. A 3.5% spread, uncorrelated with which layer was
quantised. All 18 were measured in one exclusive window, so the readings are internally
comparable — there is simply nothing to rank. **Do not rank layers on latency.**

Caveat: early stopping cut some layers to 4–5 epochs (`model.2` to 4). `best_epoch` is
0-indexed, so `model.2`'s best was its *first* epoch. The effect is small but it does
slightly confound layer-vs-layer comparison.

Full write-up: **`experiments/week8_layer_sweep/RECAP.md`**. Source of every number:
`results_master_malmo_h100.csv`.

---

## 7. Where this leaves the project

**Deployment recommendation, unchanged:**

```text
QAT batch_32, rebuilt FP16 + opt-level 5
  mAP50-95 0.7797  |  1.200 ms kernel  |  4.79 MB disk / 9.7 MB device scratch
  best accuracy of any precision, ~0.12 ms above the PTQ floor, no retraining
```

If latency is the sole objective and accuracy is negotiable, **PTQ remains the floor**
(1.082 ms / 0.7564). If INT8 is not required at all, **FP16 is free** — accuracy-neutral
versus FP32 and faster than every INT8 engine here. INT8's arithmetic advantage needs
INT8-accelerated edge hardware (Jetson / DLA) or a compute-bound model; YOLO26n on an
A100 is neither.

**The opening the sweep creates.** Single-layer QAT is a *ranking* experiment, not a
deployment candidate — the best layer still sits 0.107 mAP50-95 below the FP32 baseline.
Its value is the ordering, and the ordering is clean enough to act on:

- **Combinations.** Quantise only the layers that pay (`model.4`–`10`, `13`) and leave
  the stem and head in float. The per-layer deltas are large and well-separated at the
  top, so a greedy or top-k combination is the obvious next experiment.
- **Mixed-precision by sensitivity.** The bottom four layers actively lose accuracy
  under INT8. Excluding them is nearly free and should be tested directly.
- **Q/DQ placement.** The residual ~0.11 ms to PTQ is diffuse un-fused Q/DQ overhead
  spread across `model.2/4/6/8/13/16/19` — the *same* mid-backbone region that carries
  the accuracy. Ceiling is matching PTQ, not beating it; high effort, uncertain payoff.
- **Histogram / percentile calibration** instead of the `max` calibrator — supported
  via `CALIB_METHOD`, validated by a 1-epoch smoke, never swept.

---

## 8. Repository layout

```text
scripts/      tooling: train/ (train_qat.py), tensorrt/, export/, evaluate/, benchmark/, data/
train/        training launchers (qat_run.sh) + week8_freeze_qat, full-dataset trainers
configs/      YAML configuration
models/       checkpoints, model cards, provenance sidecars
experiments/  fusion_demo, fusion_fix, qat_iteration_2, week8_frozen_qat, week8_layer_sweep
docs/         stage-by-stage documentation
reports/      training metrics, plots, latency summaries
notebooks/    kernel-comparison analysis + figures
Per_kernel_json_file/   trtexec per-kernel profiles (FP16 / PTQ / QAT)
```

### Documentation

```text
docs/01_dataset_labelling_training.md   dataset creation and YOLO26n training
docs/02_onnx_export.md                  ONNX export + PyTorch/ONNX parity
docs/03_tensorrt_fp32.md                TensorRT FP32 engine (V2 baseline)
docs/04_tensorrt_fp16.md                TensorRT FP16 engine (V3)
docs/05_tensorrt_int8_ptq.md            INT8 post-training quantisation
docs/06_qat.md                          QAT: V5/V6, Iteration 2, latency dissection
docs/07_pytorch_latency.md              PyTorch-side latency methodology
experiments/week8_layer_sweep/RECAP.md  Week-8 per-layer sweep, full recap
experiments/fusion_demo/RESULTS.md      controlled Q/DQ fusion experiment
```

### Shared TensorRT tooling

These are **not** precision-specific. Precision is a run-time knob, never a code change:

```text
scripts/tensorrt/build_tensorrt_engine.py      ONNX -> engine.   PRECISION={fp32|fp16}
scripts/tensorrt/build_tensorrt_int8_qdq.py    Q/DQ ONNX -> INT8 engine (calibrator-free)
scripts/export/export_qat_onnx.py              QAT state -> Q/DQ ONNX
scripts/evaluate/validate_engine_parity.py     engine vs ONNX detection parity
scripts/evaluate/evaluate_engine_map.py        engine mAP (pycocotools)
scripts/benchmark/benchmark_latency_trt.py     batch=1 idle-gated latency
```

Everything is driven by environment variables — `DEVICE` is deliberately **required**,
never guessed:

```bash
# FP32 / FP16 engines from the same ONNX
PRECISION=fp32 DEVICE=<idle_gpu> python scripts/tensorrt/build_tensorrt_engine.py
PRECISION=fp16 DEVICE=<idle_gpu> python scripts/tensorrt/build_tensorrt_engine.py

# INT8 from a QAT Q/DQ ONNX, with the deployment-quality knobs
FP16=1 OPT_LEVEL=5 ONNX_PATH=... ENGINE_PATH=... DEVICE=<idle_gpu> \
  python scripts/tensorrt/build_tensorrt_int8_qdq.py

# Measure
ENGINE_PATH=$ENG DEVICE=<idle_gpu> python scripts/evaluate/evaluate_engine_map.py
ENGINE_PATH=$ENG DEVICE=<idle_gpu> BENCHMARK_REPEATS=10 \
  python scripts/benchmark/benchmark_latency_trt.py

# Per-layer sweep (malmo)
PHASE=train  bash experiments/week8_layer_sweep/run_sweep_malmo.sh
PHASE=deploy bash experiments/week8_layer_sweep/run_sweep_malmo.sh
```

### Provenance

Every stage writes a `*.provenance.json` sidecar: tool versions, input and output
sha256, build configuration, timing. Reported numbers are traceable to the exact
artifact that produced them, and engines can be cleared from disk without losing the
record — they are rebuildable from the retained ONNX.

The script tree was reformatted in a later documentation pass. For the code exactly as
it ran the Week-8 sweep, use commit **`bb32730`**.
