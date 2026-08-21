# TensorRT INT8 PTQ (V4) — concluded experiment

Post-training quantisation of the same `best.onnx` to INT8, calibrated on held-out surgical frames, measured against V2 (FP32) and V3 (FP16).

**Status: concluded, not continued.** PTQ costs **−0.0300 mAP50-95** for a negligible batch=1 speed gain, and the loss is **localisation-dominated**. That finding is what motivates the QAT stage (docs/06) — it is a baseline to beat, not a dead end.

```text
V1  PyTorch FP32       docs/01
V2  TensorRT FP32      docs/03 -- runtime only, zero precision change
V3  TensorRT FP16      docs/04 -- precision only, -0.0002 mAP50
V4  TensorRT INT8  <-  this stage: PTQ, implicit quantisation + calibration
V5  QAT                docs/06 -- recover INT8 accuracy by training for it
```

---

## Implicit vs explicit quantisation (why this matters for QAT)

PTQ here uses **implicit** quantisation: TensorRT treats the network as float, a calibrator supplies activation ranges, and the builder applies INT8 **where it judges it profitable**. The network does not say where precision changes.

QAT uses **explicit** quantisation: Q/DQ (`QuantizeLinear`/`DequantizeLinear`) nodes in the ONNX *dictate* precision transitions, and the optimiser may not introduce conversions the graph does not specify.

```text
best.onnx today:  QuantizeLinear 0   DequantizeLinear 0   -> implicit only
```

The calibrator API used here is **deprecated**:

```text
IInt8EntropyCalibrator2 -> "[DEPRECATED] Deprecated in TensorRT 10.1.
                            Superseded by explicit quantization."
```

It works on 10.16.1.11 and is what "PTQ" classically means, so it was the right mechanism for this stage. It is also a reason not to build further on this path.

---

## Controlled setup

Identical to V2/V3 except precision and calibration:

```text
              precision  fp16   int8   tf32   calibrator   source onnx
V2 FP32       fp32       False  False  False  none         ac188a00
V3 FP16       fp16       True   False  False  none         ac188a00
V4 INT8       int8       False  True   False  entropy      ac188a00
```

Same TensorRT 10.16.1.11, same A100-SXM4-80GB, same 8 GiB workspace, TF32 off throughout. `INT8_ALLOW_FP16=0`, so non-INT8 layers fall back to **FP32, not FP16** — keeping V4 a single-variable ablation against V2 rather than the usual INT8+FP16 deployment config.

**One uncontrolled difference:** V4 was built on GPU 1; V2/V3 on GPU 0. Same A100 model, and autotuning is non-deterministic build-to-build anyway, but it is one more delta than the table shows.

---

## Calibration

```text
source split:   images/train  (episodes 000000-001280)
selection:      500 frames, ONE per episode, seed 42
set sha256:     ce7774ec1ff18953...
cache sha256:   7183aadaf1e80b5e...
manifest:       calib500_seed42_manifest.txt  (episode, frame, per-frame sha256)
```

**Why train, not a held-out val slice.** The split is by episode and frames within an episode are highly correlated, so taking "other frames" from a val episode would leak into both val100 and the full-6449 evaluation. Only episode-level separation is safe, and train already is one. `make_calib_set.py` enforces this with a hard refusal, verified at frame *and* episode level:

```text
frames vs val: 0    frames vs val100: 0    episodes vs val: 0    episodes vs val100: 0
```

**Why one frame per episode.** 500 random frames from 20,756 would cluster into far fewer episodes and give correlated samples — a smaller effective calibration set than the count implies.

**Entropy over min/max.** Conv/SiLU activations have long right tails; min/max lets one outlier set the scale and compress the informative range into a few of 256 levels. Entropy picks a KL-minimising threshold and clips the tail. This choice is implicated in the result — see Accuracy.

---

## Results

### Build

```text
engine:   best_int8.engine  (4.2 MB)   sha256 82ad97d5fd32a414...
built on: A100 (GPU 1), TF32 off, entropy calibration, 500 frames
I/O:      images [1,3,640,640] FLOAT -> output0 [1,300,6] FLOAT   (unchanged)

layer output datatypes:   Int8 147 (77.0%)   Float 44 (23.0%)   Int32 2   Int64 1
engine size:   FP32 12.8 MB -> FP16 7.0 MB -> INT8 4.2 MB   (1.00 / 0.55 / 0.33)
```

**Where INT8 was declined** — the 44 float layers:

```text
attention  /model.10/m/m.0/attn/   24    QKV projection, reshape, softmax path
head       /model.23/              14    detection head + in-engine NMS
other                               6    reformat / cast nodes
```

Both refusals are correct behaviour, not calibration failure. Attention scores have a dynamic range 256 levels handle badly; the NMS block is index arithmetic (`TopK`, `Tile`, `Mod`) with no meaningful activation distribution to calibrate. TensorRT logged these explicitly as *"Missing scale and zero-point ... expect fall back to non-int8 implementation"*.

The INT8 layer count is **reproducible across builds** (147 both times) even though total layer count varied with fusion (193 vs 191) — precision assignment is stable, kernel selection is not.

> **Verification note.** An earlier version of the build script counted precision from a `Precision` key that does not exist in the inspector JSON; every layer read back as `"?"` and a 77%-quantised engine looked like total INT8 failure. Precision is carried by each layer's output-tensor `Format/Datatype`. The check now reads that, and runs **before** the engine is written, so a genuinely failed calibration cannot leave a mislabelled file on disk.

### Parity vs the FP32 ONNX — 2/100

```text
2/100 frames pass     max coord diff 2.932 px     max conf diff 0.072
per-detection: 13 matched pairs across only 8 comparable frames
  coord_diff  median 4.817e-01 px   max 2.932e+00
  conf_diff   median 1.322e-02      max 7.204e-02
  matched IoU min 0.98054
```

**The per-detection statistic does not rescue this one.** For FP16 it did — 88.9% of detections passed against 77% of frames. Here only **8 of 100 frames were numerically comparable at all**; the other 92 failed structurally on box count or matched-IoU below 0.98, never reaching the coord/conf comparison. So the 13-pair pool is self-selected, drawn from the frames that happened to match. Quote it as evidence of *how far* things moved, not as a pass rate.

New failure mode vs FP16: **IoU failures** (0.9743, 0.9739, 0.9135). FP16 kept boxes sub-2-pixel. INT8 moves them enough to fall below the 0.98 bar — geometry is genuinely shifting.

Tolerances were left at their FP32-calibrated values throughout. Loosening them to manufacture a pass would make the gate meaningless.

### Accuracy (pycocotools, val100 @ conf 0.001)

```text
              mAP50    mAP50-95   dets    Δ mAP50   Δ mAP50-95   (vs V2)
V2 FP32      0.9350     0.7572     986    +0.0000      +0.0000
V3 FP16      0.9348     0.7572     989    -0.0002      -0.0000
V4 INT8      0.9236     0.7272    1086    -0.0115      -0.0300
```

**The loss is localisation-dominated: mAP50-95 falls 2.6x harder than mAP50.**

That split is the diagnostic. mAP50 measures whether the object was found; mAP50-95 measures how precisely it was boxed. INT8 is still finding the tools — it is boxing them less exactly. The parity data agrees independently: 92/100 frames failed on IoU or box count, with matched IoUs down to 0.9135.

The likely mechanism is **entropy calibration clipping activations that carry box-coordinate precision**. Entropy deliberately saturates distribution tails, and for regression outputs those tails are semantically real — large boxes, coordinates near frame edges — rather than noise. A min/max A/B would test this directly and was not run before the stage concluded.

### Latency (batch=1, 10x100, GPU 1)

```text
Stage          Mean      Std   Median      Min      P95      P99
Preprocess    1.937    0.736    1.804    0.952    3.328    4.411
Inference     1.506    0.062    1.488    1.459    1.616    1.791
Postprocess   0.022    0.006    0.020    0.018    0.032    0.048
Total         3.466    0.757    3.331    2.446    4.893    6.021
                                                     -> 300.2 FPS
```

```text
stage           V2 FP32   V3 FP16   V4 INT8   V4/V2   V4/V3
Inference        2.318     1.561     1.488    1.56x   1.05x
Total            3.736     2.922     3.331    1.12x   0.88x
```

**INT8 is only 5% faster than FP16 on inference, and slower end-to-end.** Two separate reasons:

1. **Amdahl.** 23% of layers stayed float — and they are attention and the detection head, apparently where the time actually goes. Quantising 77% of layers bought 1.56x over FP32, not the 2-4x the arithmetic alone suggests.
2. **The Total column is contaminated.** V4 ran on GPU 1; V2/V3 on GPU 0, on a different day. Preprocess moved 1.372 -> 1.804 ms, which is host-side CPU variance with nothing to do with precision. **Compare Inference across the three; treat Total as non-comparable.**

A paired re-benchmark of all three in one session was attempted and **could not be completed** — see Caveats.

---

## Open items — must be closed before QAT is judged against V4

These are not rhetorical caveats. QAT is measured against the numbers in this
document, so both gaps below directly limit what any QAT result can claim.

```text
OPEN 1 -- FULL-6449 mAP ON V4 NEVER RAN.
  The PTQ accuracy baseline is val100-only: 100 images, 237 boxes. A -0.0115
  mAP50 delta is inside that set's noise floor, so the magnitude is an estimate,
  not a measurement. Blocked by host commit-limit exhaustion (see Caveats).
  ACTION: run MODE=engine on the full 6,449-image val split when the box frees,
  so QAT compares against a resolved number rather than a 237-box estimate.
  Until then, quote V4 accuracy as "val100-only".

OPEN 2 -- MIN/MAX CALIBRATOR A/B NEVER RAN.
  "Entropy clipping box-coordinate activation tails" is the best-supported
  explanation for the localisation-dominated loss, INFERRED from the
  mAP50-vs-mAP50-95 asymmetry and the parity IoU failures. It is NOT confirmed.
  A min/max build would test it directly: min/max clips nothing, so if the
  hypothesis holds it should recover much of the mAP50-95 gap.
  ACTION: optional. If left unrun, this document must continue to describe the
  mechanism as inferred rather than measured -- do not let it harden into a
  stated cause through repetition.
```

## Caveats

```text
- val100 NEVER RESOLVED THIS DELTA. -0.0115 mAP50 on 237 boxes is inside the
  noise floor of a set that size. FP16 could hand-wave val100 because its delta
  was 0.0002; INT8 cannot. The full 6,449-image run is REQUIRED before the
  accuracy figures here are trusted quantitatively. The direction and the
  mAP50-vs-mAP50-95 asymmetry are solid; the magnitudes are not.

- The paired V2/V3/V4 latency re-run was blocked. The host runs strict
  overcommit (vm.overcommit_memory=2) and reached ~2.4 GB of commit headroom
  against a 493 GB CommitLimit, so TensorRT could not even be imported --
  "failed to map segment from shared object". Not a GPU or code fault. The
  benchmark script aborted the whole set rather than record a partial one.

- No latency run in this project had an exclusive GPU (exclusive_gpu: false on
  all of V2/V3/V4). Dormant foreign CUDA contexts were tolerated via
  GATE_ALLOW_IDLE_MIB and recorded.

- The min/max calibrator A/B was not run, so "entropy clipping caused the
  localisation loss" remains the best-supported explanation rather than a
  measured conclusion.
```

---

## Reproduce

```bash
source ~/venvs/medtronic-trt/bin/activate
ENG=models/yolo26n_sanoscience_full_left/3_int8_ptq/best_int8.engine

# calibration set (images are regenerable; only the manifest is committed)
N_EPISODES=500 SEED=42 python scripts/data/make_calib_set.py

# build (reuses best_int8_entropy.calib_cache if present, skipping calibration)
PRECISION=int8 CALIBRATOR=entropy DEVICE=<idle_gpu> python scripts/tensorrt/build_tensorrt_engine.py

ENGINE_PATH=$ENG DEVICE=<idle_gpu> python scripts/evaluate/validate_engine_parity.py
MODE=engine ENGINE_PATH=$ENG DEVICE=<idle_gpu> python scripts/evaluate/evaluate_engine_map.py
ENGINE_PATH=$ENG DEVICE=<idle_gpu> BENCHMARK_REPEATS=10 python scripts/benchmark/benchmark_latency_trt.py
```

---

## Why this leads to QAT

PTQ quantises a network that was never trained to tolerate it. The weights are optimised for FP32 arithmetic, and calibration can only choose the least-bad scales after the fact — which for box regression means clipping tails that carry real coordinate information.

QAT instead inserts fake-quantisation into training so the network learns weights that survive INT8, and expresses the result as explicit Q/DQ nodes the TensorRT builder must honour. NVIDIA report EfficientNet-B0 at PTQ 33.9% vs QAT 76.8% against an FP32 77.4% baseline — a far larger recovery than our own gap, but the same mechanism.

**V4 is the number QAT must beat: mAP50 0.9236 / mAP50-95 0.7272, and specifically the −0.0300 localisation loss.** Continued in docs/06.
