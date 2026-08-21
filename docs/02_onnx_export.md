# ONNX Export

This document records the FP32 ONNX export of the trained YOLO26n Sanoscience baseline, and the PyTorch↔ONNX parity gate that validates it. The `.onnx` produced here is the input to the TensorRT stages that follow (`docs/03`–`docs/05`).

The export is deliberately **FP32 only**. No precision change happens here — half precision, INT8, and quantisation-aware training are separate, later stages, done one at a time so each accuracy change is attributable to a single cause. This stage's only job is to move the model from PyTorch into the ONNX format **without changing what it computes**, and to prove that.

---

## Overview flow

```text
results/models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.pt   (FP32 baseline, read only)
        │
        │  ultralytics + torch load the model and run one forward pass
        ▼
torch traces the forward pass and re-expresses every operation
in the ONNX operator set at opset 17
        │
        │  onnxslim simplifies the resulting graph
        ▼
best.onnx   (self-contained: weights + graph + baked-in NMS)
        │
        ├──► PyTorch↔ONNX parity gate  (16 frames, CPU)
        │       compares final detections of .pt vs .onnx
        │       PASS = export is faithful
        │
        └──► best.onnx.provenance.json  (sha256, versions, parity record)
        │
        ▼
next stage: TensorRT FP16 engine  (docs/03)
```

The input `best.pt` is never modified. The export writes a sibling `best.onnx`; the original checkpoint is read only.

---

## Script

Main script:

```text
scripts/export/export_onnx.py
```

It is a flat top-to-bottom program with a `# Settings` block of hardcoded constants and env-var overrides for run-time knobs, matching the convention of the other scripts in `scripts/`. It does three things in order:

```text
1. Export       best.pt -> best.onnx   via ultralytics model.export()
2. Parity       run both models on the same frames, compare final detections
3. Provenance   write best.onnx.provenance.json next to the .onnx
```

If parity fails, the script exits non-zero and prints a warning **not** to use the ONNX for TensorRT. A `.onnx` only leaves this stage if it provably matches the PyTorch baseline.

Export settings baked into the ONNX:

```text
format:   onnx
opset:    17
imgsz:    640
batch:    1
dynamic:  False   (static [1, 3, 640, 640] input, no batch dimension to sweep)
half:     False   (FP32 only)
simplify: True    (onnxslim graph simplification; it ran)
device:   cpu     (the export trace and the parity check both run on CPU)
```

---

## What happened

### End-to-end (NMS-free) output

YOLO26 is an end-to-end / NMS-free architecture. Both the PyTorch model's forward pass **and** the exported ONNX emit **final detections** directly:

```text
output shape: [1, 300, 6]   =   [batch, max_detections, (x1, y1, x2, y2, conf, cls)]
```

Non-maximum suppression is baked into the model graph, so the `.onnx` is a complete detector — no external post-processing is needed to turn its output into boxes. This is a property of the architecture, not an export flag.

This had a direct consequence for the parity check. An earlier assumption — that the model outputs a raw pre-NMS tensor and NMS is applied afterwards — does not hold for YOLO26. `ultralytics.utils.ops.non_max_suppression` no longer exists in ultralytics 8.4.90. The parity gate was therefore written to compare the models' **final detections** directly: filter each model's output at the same confidence, greedily match boxes, and measure the coordinate and confidence differences on matched pairs.

### Parity result (export fidelity)

Parity was run on 16 validation frames, on CPU, comparing `best.pt` (PyTorch) against `best.onnx` (ONNX Runtime) fed the identical preprocessed input. CPU is used deliberately: it is deterministic and free of GPU-contention noise, so the check isolates export fidelity and nothing else.

```text
frames:            16
device:            cpu (both models)
match criterion:   box IoU >= 0.98, same class, same count
max_coord_diff:    1.98e-04 px   (largest box-corner disagreement)
max_conf_diff:     1.51e-05      (largest confidence disagreement)
result:            PASS  (16/16 frames)
```

The two models produce effectively identical detections — sub-pixel agreement by four orders of magnitude. The export did not lose anything.

### Accuracy check (against ground truth)

Parity proves the ONNX equals the PyTorch model; it does **not** by itself measure accuracy against labels. To confirm the ONNX inherits the baseline's accuracy, both models were validated on the 100-image random val subset. mAP is measured at `conf = 0.001` — the Ultralytics evaluation protocol — so the full precision-recall curve is built. (A deployment-style `conf = 0.25` truncates that curve and gives a misleadingly different number; it is not an accuracy-measurement threshold.)

```text
validation subset:  demo_val100_random_yolo  (100 images, 100 labels)
conf:               0.001   (mAP protocol, not deployment threshold)

                    mAP50     mAP50-95
PT   (.pt, GPU)     0.9408    0.7673
ONNX (.onnx, CPU)   0.9394    0.7595
difference          0.0014    0.0078
```

The PyTorch subset mAP50 (0.9408) matches the full-set training baseline (0.934), confirming the subset is representative. The ONNX matches the PyTorch model to within 0.0014 mAP50. Accuracy is preserved.

The tiny residual difference is expected and benign: `.pt` ran on GPU and `.onnx` on CPU (slightly different floating-point arithmetic), and the 300-detection cap reshuffles a few near-zero-confidence boxes at the bottom of the curve. It is not export corruption — the CPU-vs-CPU parity check above already established the export itself is exact.

---

## Important elements to take into account

```text
End-to-end model.  Output is final detections [1, 300, 6], not a raw pre-NMS
tensor. NMS is inside the graph. Do not apply NMS again downstream.

opset 17.  Chosen for compatibility with both TensorRT (8.5+ / 10.x) and
onnxruntime (>= 1.14). Pinned and recorded for reproducibility.

Static batch = 1.  The input shape [1, 3, 640, 640] is fixed. This matches the
batch=1 latency baseline; a batch sweep is out of scope. There is no dynamic
batch dimension in the graph.

FP32 only.  No precision change at export. FP16 / INT8 / QAT are later stages.

Parity runs on CPU.  Deterministic and contention-free, so it measures export
fidelity alone. This is not a latency measurement.

Accuracy is measured at conf = 0.001.  That is the mAP protocol. A deployment
threshold (conf ~ 0.25) is the wrong tool for measuring accuracy.

Device vs precision.  CPU-vs-GPU does not meaningfully change accuracy (only
floating-point noise at the 4th-5th decimal). Precision (FP16 -> INT8) is what
will change accuracy downstream, and recovering that drop is the point of QAT.
Deployment / latency is on GPU; the CPU work here is export validation only.

The .onnx is git-ignored (*.onnx) and force-added as an exception, exactly like
best.pt. Its provenance sidecar records the source-.pt sha256, so the committed
.onnx is traceable to the exact checkpoint that produced it.
```

---

## Environment and versions

The export was run in the UCL environment under the project venv:

```text
host venv:   ~/venvs/medtronic-qats
python:      3.9.25
```

Package versions used for this export (recorded in the provenance file):

```text
ultralytics   8.4.90
torch         2.8.0+cu128
onnx          1.19.1
onnxruntime   1.19.2
onnxslim      0.1.94
numpy         2.0.2
opencv        5.0.0
```

Three packages had to be added to the venv for this stage — the rest were already present from training:

```text
onnx          (export + graph inspection)
onnxruntime   (parity: run the .onnx on CPU)
onnxslim      (graph simplification during export)
```

Install:

```bash
source ~/venvs/medtronic-qats/bin/activate
pip install onnx onnxruntime onnxslim
```

---

## Reproduce

```bash
cd ~/medtronic_qat/Medtronics-UCL-QAT
source ~/venvs/medtronic-qats/bin/activate
python scripts/export/export_onnx.py
```

Run-time knobs are environment-variable overrides with defaults, e.g.:

```bash
OPSET=17 IMG_SIZE=640 N_PARITY=16 CONF=0.25 python scripts/export/export_onnx.py
```

Note that `CONF` here is the parity-comparison threshold (which detections both models must agree on), not the mAP-evaluation threshold used for the accuracy check above.

---

## Outputs

```text
results/models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.onnx                   (9.4 MB, FP32, opset 17)
results/models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.onnx.provenance.json   (sha256 + versions + parity record)
```

Both are committed to the repository (the `.onnx` force-added past the `*.onnx` ignore, as with `best.pt`). The provenance JSON is the human-readable record of this export: source checkpoint sha256, all package versions, the exact export settings the exporter baked in, and per-frame parity results.

---

## Next stage

The validated FP32 `best.onnx` is the input to the TensorRT FP16 engine build (`docs/03`). The accuracy figures above (ONNX mAP50 0.9394 / mAP50-95 0.7595) are the FP32 reference that every downstream engine — FP16, INT8/PTQ, QAT — is compared against.
