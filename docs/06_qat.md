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

## Results (V5 -- first real fine-tune, 10 epochs)

Run: 10 epochs, lr0 1e-3 / lrf 0.01, warmup 3, amp off, batch 16, ~1h41m on A100
(GPU 1). State: `runs_qat/qat_v5/qat_modelopt_state_best.pt` (best epoch = 10).

**Accuracy** (full 6,449-img val, pycocotools) vs the other precisions:

```text
                mAP50    mAP50-95
V2 FP32         0.9325   0.7747
V3 FP16         0.9327   0.7748
V4 INT8 PTQ     0.9282   0.7571
V5 INT8 QAT     0.9283   0.7437   <- this run
```

QAT did **not** recover accuracy: mAP50-95 (0.7437) is BELOW both PTQ and FP32.
Cause: **undertrained** -- still climbing at epoch 10 (mAP50-95 0.684 ep1 -> 0.743
ep10, best = last epoch). NVIDIA's recipe is ~10% of the original schedule with an
annealing LR; note the 30% warmup on 10 epochs also re-perturbs the converged
model. Calibration is `MaxCalibrator`; histogram/percentile is NVIDIA-preferred
and untried.

**Latency** (batch=1, val100, CUDA-event; engine numbers idle-gated):

```text
                        kernel(GPU) median   total median   FPS(total)
V2 FP32 engine              2.019 ms            3.736 ms       267.7
V3 FP16 engine              1.137 ms            2.922 ms       342.2
V5 INT8 QAT engine          1.429 ms            3.789 ms       263.9
PyTorch QAT fake-quant     81.427 ms           83.958 ms        11.9   <- NOT deployable
```

## Latency analysis -- why INT8 is NOT faster than FP16 here

INT8 kernel (1.429 ms) is **slower** than FP16 (1.137 ms). Investigated and settled:

- **NOT FP32 fallback.** Rebuilding the same Q/DQ ONNX with INT8+FP16 moved 56
  layers FP32->FP16 (`144 INT8 / 113 FP32` -> `144 INT8 / 56 FP16 / 40 FP32`) but
  the kernel did **not** change (1.429 -> 1.440). Disproves the fallback theory.
- **Explicit-quantization semantics** (NVIDIA): in a Q/DQ network TensorRT is
  forbidden from swapping Q/DQ-dictated INT8 for faster FP16 -- so the INT8+FP16
  flag *cannot* help, exactly the observed null result.
- **Real cause: Q/DQ reformat overhead on a launch-bound tiny model.** YOLO26n is
  2.5M params / 5.8 GFLOPs; on A100 each layer is memory/launch-bound, not
  compute-bound, so INT8's arithmetic advantage is ~nil while the reformat kernels
  at INT8<->other boundaries add cost. Uniform-precision FP16 has no reformats.

**Deployment implication:** on A100, **FP16 is the right precision for this model**.
INT8/QAT pays off on INT8-accelerated edge HW (Jetson/DLA) or larger compute-bound
models. The PyTorch fake-quant 81 ms is the eager Q/DQ-in-FP32 simulation (~57x the
deployed engine) -- it is *why* deployment runs on TensorRT, not a number to compare.

## Open items

- **Longer QAT fine-tune to plateau** -- accuracy is the real gap; best-epoch
  checkpoint already keeps the best.
- Try **histogram/percentile calibration** (vs the current MaxCalibrator); reduce warmup.
- INT8+FP16 build engine kept in scratch (does not help latency; not shipped).
- Two-venv ergonomics + the INT8-only build flag are documented on the conversion page.
