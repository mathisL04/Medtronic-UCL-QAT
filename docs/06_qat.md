# Quantisation-Aware Training

QAT fine-tunes the converged V1 baseline with fake-quant nodes so the INT8 scales
are *learned*, not just calibrated (PTQ). It is the V5 stage.

## Where QAT runs, where it is measured

```text
train:    PyTorch  -- fake-quant / Q-DQ nodes, optimiser + labelled data
convert:  PyTorch -> ONNX (Q/DQ) -> TensorRT INT8 engine
measure:  TensorRT -- accuracy (mAP) and, above all, latency
```

QAT is a training-time technique: it runs in PyTorch, never on TensorRT. TensorRT
is inference-only -- its job is to fold the learned scales into a real INT8 engine
that we then benchmark. We do **not** "train on TensorRT".

## Fine-tune parameters

Fine-tuning is training with a small learning rate: the weights are already
converged, so a high LR would retrain rather than adapt. Use a low LR that decays
to near-zero -- "small" is the start point, "decay" is the shape (not alternatives).

```text
lr0     ~1e-3 or lower   ~1% of the full-training lr0 (0.01); NOT yet set in the
                         script -- it currently inherits Ultralytics' 0.01
lrf     keep default     decay tail (cosine/linear to near-zero) -- wanted
epochs  by convergence   short fine-tune; choose when mAP stops improving, NOT 2^n
batch   16               power-of-two is a mild GPU-efficiency nicety, not a rule
imgsz   640              LOCKED: multiple of 32 (YOLO stride) and the engine's
                         fixed input -- do not change
amp     False            fake-quant scales are FP32; FP16 autocast corrupts them
                         (deviation from V1, which trained with amp)
workers 0 for the smoke  avoids the fork/overcommit OSError-12 on the shared box
layers  unchanged        QAT wraps existing layers with fake-quant; it never
                         adds/removes any. Architecture is fixed.
```

Power-of-two note: it matters *mildly* for batch (memory/tensor-core alignment)
and not at all for epochs. The only hard sizing rule is imgsz = multiple of 32,
which 640 already satisfies.

## Environment (V5 stack) -- a deliberate migration

QAT training and QAT export run on a **different** venv from the FP32/FP16/PTQ
work, and the split is forced, not incidental:

```text
V2-V4 (TensorRT stages):  ~/venvs/medtronic-trt        Py3.9,  TensorRT 10.16.1.11
V5 QAT train + export:    ~/venvs/medtronic-qat-p311   Py3.11, torch 2.7.0+cu128,
                                                        modelopt 0.33.1
(superseded)              ~/venvs/medtronic-qat         Py3.9,  modelopt 0.29.0
```

Why the migration off Py3.9 / modelopt 0.29:

```text
- modelopt 0.29 CANNOT export this model to Q/DQ ONNX under ANY torch version
  (see root cause below). It is a dead end for deployment, not just slow.
- modelopt >= 0.31 fixes the export path but requires Python >= 3.10, so the
  training venv had to move to Py3.11 as well. 0.29 was the last Py3.9 build.
- train_qat.py runs UNCHANGED under 0.33.1: same 608 quantisers / 253 with
  scales, reload gate passes, losses coherent. The migration touched the
  environment, not the training logic. qat_run.sh takes PY as an env override
  so the venv can move without editing the script.
```

The engine build (Q/DQ -> INT8) still happens in the TensorRT venv; the ONNX is a
portable hand-off between the two.

## Export recipe (QAT fake-quant -> Q/DQ ONNX -> TensorRT INT8)

This was hard-won and is fragile. The exact working path:

```text
export:  scripts/export_qat_onnx.py   (run with the Py3.11 venv)
build:   scripts/build_tensorrt_int8_qdq.py   (run with the TensorRT venv)
```

**Root cause of the original failure -- it is the API, not the torch version.**
The obvious approach, `torch.onnx.export()` under modelopt's `export_torch_mode()`,
FAILS: the quantiser scale `_amax` traces as a graph input where modelopt's INT8
symbolic needs an `onnx::Constant` -> `SymbolicValueError: got 'prim::Param'`.
This fails identically on torch 2.6, 2.7 and 2.8, and on BOTH the legacy and
dynamo exporter backends. **Do not chase a torch downgrade** -- it does nothing.

**The fix is modelopt's blessed API**, which wraps the export in
`torch.inference_mode` with the right graph post-processing:

```text
from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata
payload, _ = get_onnx_bytes_and_metadata(model, dummy, onnx_opset=17)
OnnxBytes.from_bytes(payload).write_to_disk(dir)
```

**The dep set must be pinned exactly -- each pin fixes a real failure:**

```text
modelopt   0.33.1     >=0.31 for the fixed export; needs Py>=3.10
torch      2.7.0                (version is NOT the discriminator; any works with 0.33)
onnx       1.17.0     onnx 1.22 REMOVED onnx.reference.custom_element_types, which
                      modelopt imports -> ImportError at import time
numpy      <2         numpy 2.x breaks modelopt's onnx importer
plus:      onnxruntime, onnx_graphsurgeon, polygraphy   (the modelopt [onnx] extra,
           from --extra-index-url https://pypi.nvidia.com)
```

**Head export mode gives a single deployment output.** Restored raw, YOLO26n's
forward returns 11 tensors ([1,300,6] detections + 10 one2many/one2one auxiliary
heads). Setting `Detect.export=True` (+ eval, requires_grad off) collapses it to
the single `[1,3,640,640] -> [1,300,6]` graph.

**The 255 -> 207 Q/DQ drop is CORRECT, not a regression.** Without head export
mode the graph has ~255 Q/DQ pairs; with it, ~207. The difference is the
one2many auxiliary training heads being pruned -- their quantisers leave the graph
with them. Only the deployment path (one2one + backbone) stays quantised.

**TensorRT INT8 build is calibrator-free.** A Q/DQ ONNX is explicit quantization:
`config.set_flag(trt.BuilderFlag.INT8)` and TensorRT reads scales from the Q/DQ
nodes. (Contrast the PTQ path in build_tensorrt_engine.py, which needs an
IInt8Calibrator + calibration data.) Built cleanly on TRT 10.16.1.11.

## Pipeline status -- structurally proven, accuracy NOT yet measured

The full chain is validated end-to-end, but only on a **throwaway 1-epoch smoke**
model whose scales are not converged. What this proves and does not:

```text
PROVEN (structure):  train (unchanged) -> mto.save/restore (reload gate 608/253)
                     -> Q/DQ ONNX [1,3,640,640]->[1,300,6], 207 Q/DQ pairs
                     -> TensorRT INT8 engine (4.2 MB, deserialises, single output)
NOT YET MEASURED:    whether a REAL fine-tune preserves accuracy through this
                     path. The smoke model's mAP is meaningless; no INT8 accuracy
                     or latency number exists yet.
```

## Planned content (remaining)

- Two code changes before the real run: `lr0` ~1e-3 + `lrf` in the overrides
  (currently inherits 0.01), and an mto-based best-epoch checkpoint callback (the
  smoke keeps only the final EMA because save_model is disabled).
- Real multi-epoch QAT fine-tune, then export (recipe above) + INT8 build.
- TensorRT INT8 accuracy (mAP) + latency comparison vs V2 FP32 / V3 FP16 / V4 PTQ.
