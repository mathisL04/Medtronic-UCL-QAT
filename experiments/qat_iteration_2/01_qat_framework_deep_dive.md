# QAT Framework — Deep Dive (engineer reference)

Method = **NVIDIA modelopt QAT** (float fake-quant + STE) → deploy = TensorRT explicit Q/DQ.
Model = YOLO26n, `[1,3,640,640] → [1,300,6]`. Code: `train_qat.py`, `export_qat_onnx.py`,
`build_tensorrt_int8_qdq.py`.

---

## A. Training

**A.1 Insert quantizers** — `train_qat.py:207`
```python
model = mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)
```
Walks the module tree; wraps every `Conv2d`/`Linear` with `TensorQuantizer`s (input + weight) →
**608 quantizers**. `forward_loop` calibrates the initial scales.

**A.2 Fake-quant (forward)** — round to INT8 and back, staying float:
```text
s     = amax / 127
x_int = clamp(round(x/s), −127, 127)
x_fq  = x_int * s            # float, but numerically = INT8
```
Simulating in float keeps autograd working so the loss *feels* the rounding error.

**A.3 STE (backward)** — `round()` has zero gradient, so it's treated as identity inside the range:
```text
∂x_fq/∂x ≈ 1  inside [−127,127],  0 outside
```
This is what makes QAT trainable (📖 STE: arXiv:1308.3432).

**A.4 Fine-tune** — `trainer.train()` (Ultralytics loop on the quantized model):
```text
TRAINED  → weights (optimizer, LR 1e-3)         ← QAT moves these to absorb the rounding
FROZEN   → _amax / scales (a buffer, not a param) ← set at calibration, kept fixed
amp=False (HAZARD 2)  fake-quant scales are FP32 → no FP16 autocast
quantize-before-EMA (HAZARD 1)  get_model():292 before ModelEMA:373
```
Why fine-tune (vs PTQ): weights adapt to the grid → recovers accuracy (PTQ 0.7571 → QAT V6 0.7644).

---

## B. Config — `INT8_DEFAULT_CFG` (verified)
`num_bits=8`, `calibrator="max"`, `fake_quant=True`, matched by fnmatch wildcard:
```python
"*weight_quantizer": {axis: 0}      # WEIGHTS     → per-channel (one scale per output filter)
"*input_quantizer":  {axis: None}   # ACTIVATIONS → per-tensor
```
- **Per-channel weights**: each filter keeps its own range → far better weight fidelity, static/free.
- **Per-tensor activations**: a single runtime scale — what HW/TensorRT want.
- Calibration = 128 episode-diverse frames in `eval/no_grad`; `max` calibrator writes abs-max → `_amax`.
📖 https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html

---

## C. Export — quantizers → ONNX Q/DQ  (`export_qat_onnx.py`)
```python
get_onnx_bytes_and_metadata(model, dummy, onnx_opset=17)   # NOT torch.onnx.export
```
- Each active quantizer → a **`QuantizeLinear`+`DequantizeLinear`** pair carrying its frozen scale.
- Blessed API, not `torch.onnx.export` — the latter fails (`SymbolicValueError` on `_amax`).
- `Detect.export=True` drops the training-only aux heads → single `[1,300,6]`.
📖 https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html

---

## D. Build — TensorRT reads Q/DQ  (`build_tensorrt_int8_qdq.py`)
`config.set_flag(trt.BuilderFlag.INT8)`, **no calibrator** (scales are in the graph = explicit quant).
- Fuses `Q → Conv → DQ` into one **INT8 kernel** (all convs).
- Layers with no Q/DQ (attention softmax/matmul, NMS `TopK/Mod`) → **FP32/FP16**.
- **Reformats** appear at INT8↔float boundaries (pure overhead).
📖 https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html

---

## E. Dependencies — import → role → version → venv

Two venvs: **modelopt needs py≥3.10** (train/export in py3.11), **TensorRT** in py3.9. The Q/DQ ONNX
file is the hand-off between them.

| Import | Role | Version · venv |
|---|---|---|
| `modelopt.torch.quantization` (**mtq**) | `quantize()`, `INT8_DEFAULT_CFG` — **insert + calibrate fake-quant** | 0.33.1 · p311 |
| `modelopt.torch.opt` (**mto**) | `save`/`restore` the quant state (recipe+weights+scales) | 0.33.1 · p311 |
| `modelopt...nn.TensorQuantizer` | the fake-quant unit (forward math + `_amax`) | 0.33.1 · p311 |
| `modelopt...\_deploy...get_onnx_bytes_and_metadata` | **Q/DQ ONNX export** | 0.33.1 · p311 |
| `torch` | tensors, autograd/STE, the model | 2.7.0+cu128 · p311 |
| `ultralytics` (YOLO, DetectionTrainer) | model + **training harness** (loss, aug, LR, EMA) | 8.4.90 · p311 |
| `onnx` | verify Q/DQ graph (counts, output shape) | 1.17.0 · p311 |
| `tensorrt` | **build the INT8 engine** | 10.16.1.11 · trt |
| `pycocotools` · `cuda-python` · `pynvml` | mAP metric · device mem/stream · idle-gating | trt |
| `numpy` · `opencv` | arrays · letterbox preprocess | p311 / trt |

**modelopt submodules:** `mtq` = the algorithm · `mto` = state I/O · `.nn` = the inserted modules
(`TensorQuantizer`, `QuantConv2d`) · `._deploy` = ONNX export.

---

**In one line:** `mtq.quantize` wraps convs in fake-quant (INT8↔float, STE-trainable); fine-tune moves
**weights** while **scales stay frozen**; export → Q/DQ ONNX → TensorRT fuses into INT8 kernels, floats
the attention/NMS ops, and adds reformats at the seams.
