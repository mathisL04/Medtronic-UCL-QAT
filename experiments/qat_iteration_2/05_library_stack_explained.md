# The Library Stack, Explained — modelopt in context (review, run nothing)

An accessible, teachable explanation of what each library the QAT work added actually *does* and
*where* it acts — focused on **modelopt**. Concept first, API second.

---

## 1. modelopt — what it fundamentally IS

**NVIDIA Model Optimizer (`nvidia-modelopt`) is NVIDIA's PyTorch library for compressing models for
deployment** — quantization, pruning, distillation, sparsity. We use exactly one part of it:
**quantization**, and within that, **QAT**.

Think of it as a **layer that sits on top of PyTorch and rewrites your model's arithmetic to lower
precision**, while keeping it a normal PyTorch model you can still train. It does **not** run
inference fast itself and it does **not** deploy anything — it *prepares* a model so that a deployment
engine (TensorRT) can run it in INT8.

- **What it DOES to our model:** inserts simulated-quantization ("fake-quant") into every convolution
  so the model can be *trained* to tolerate INT8, then exports that model in a form TensorRT
  understands (Q/DQ ONNX).
- **WHERE it acts:** the **training and export stages only** — entirely in **PyTorch, on the CPU/GPU
  during training**. It is completely gone by deploy time; the deployed engine is pure TensorRT.

```text
[modelopt acts here]              [modelopt is gone here]
 train (PyTorch) ── export ONNX ──►  build engine (TensorRT) ── run
```

📖 https://nvidia.github.io/Model-Optimizer/ · repo https://github.com/NVIDIA/Model-Optimizer

---

## 2. The submodules we use — what each is responsible for

| Import | Plain-terms job | When it runs |
|---|---|---|
| **`modelopt.torch.quantization` (`mtq`)** | The quantization **algorithm**. `mtq.quantize(model, cfg, forward_loop)` rewrites the model to fake-quant + calibrates it. `mtq.print_quant_summary` reports what it did. | training (once, up front) |
| **`TensorQuantizer`** (`modelopt.torch.quantization.nn`) | The **fake-quant unit** — the small module that actually does "round to INT8 and back to float" in the forward pass and holds the scale (`_amax`). 608 of them get inserted. | every forward pass |
| **`modelopt.torch.opt` (`mto`)** | **Save/restore of the modelopt "state"** — `mto.save` writes the quantization *recipe + trained weights + scales* to a file; `mto.restore` rebuilds that exact quantized model on a fresh baseline. Needed because normal `torch.save` can't pickle modelopt's dynamic classes. | after training / before export |
| **`modelopt.torch._deploy...get_onnx_bytes_and_metadata`** | The **export path** — turns the fake-quant model into **ONNX with Q/DQ nodes** that TensorRT can read. | export stage |

One line each:
```text
mtq            → "quantize my model and calibrate it"
TensorQuantizer→ "the thing that simulates INT8 inside each layer"
mto            → "save/reload the quantized model correctly"
_deploy export → "write it out as Q/DQ ONNX for TensorRT"
```

---

## 3. How modelopt BEHAVES — what it physically changes

When you call `mtq.quantize(model, ...)`, modelopt **modifies the PyTorch module tree in place**:

- **It REPLACES** each quantizable layer with a modelopt **subclass**: `nn.Conv2d` → `QuantConv2d`.
  This subclass is generated **dynamically at runtime** (that's why plain `torch.save` chokes on it —
  and why we use `mto.save`).
- **The subclass WRAPS the original operation** — same weights, same conv math — and **ADDS** two
  small submodules around it: an **input `TensorQuantizer`** (fake-quants the activation) and a
  **weight `TensorQuantizer`** (fake-quants the weights). So a quantized conv is literally:
  `x → input_quantizer → (weight_quantizer→weights) → conv → output`.
- **It PRESERVES the module tree shape** — same layer names, same nesting, same forward signature. That
  is *why the rest of the stack doesn't notice*: Ultralytics still calls `model(x)`, still computes the
  same loss, still backprops — it just happens to be driving a model whose convs now round to INT8 and
  back on the way through.

**What it leaves completely untouched:**
- The **architecture / graph** (no layers added or removed from the network's logic — only quantizers
  wrapped around existing ops).
- **Non-quantized ops** — activations (SiLU), norms, the attention softmax/matmul, the NMS head:
  modelopt doesn't wrap these (they stay float).
- The **training loop, optimizer, data, loss** — all Ultralytics/PyTorch, unchanged.
- The **weights' values** at insertion time — quantize() only *wraps*; the fine-tune is what later
  moves the weights.

**Coexistence in one picture:**
```text
PyTorch          provides autograd + tensors + nn.Module   (the engine)
   └─ Ultralytics drives the training loop on an nn.Module  (the driver)
        └─ modelopt has swapped some nn.Modules for QuantConv2d underneath  (new parts in the car)
             └─ TensorQuantizer does fake-quant inside each, transparently to the driver
```
Ultralytics is **unaware** it's training a quantized model — it sees a normal `DetectionModel`.
modelopt changed the *parts*; PyTorch still runs them; Ultralytics still drives.

---

## 4. The other added libraries — what each contributes and where

| Library | What it contributes | Where it acts |
|---|---|---|
| **ONNX** (`onnx`) | The **interchange format** — a framework-neutral graph. modelopt writes the quantized model as ONNX with **Q/DQ nodes**; we read it back to *verify* (count Q/DQ, check the single `[1,300,6]` output). It's the **bridge** between the PyTorch world and the TensorRT world. | export → build boundary |
| **TensorRT** (`tensorrt`) | The **deployment engine builder + runtime** — reads the Q/DQ ONNX, **fuses Q→conv→DQ into real INT8 kernels**, picks the fastest kernels for the GPU, and produces the `.engine` that actually runs fast at inference. This is where INT8 becomes *real* (not simulated). | build + run |
| (supporting: `cuda-python`, `pycocotools`, `pynvml`) | device memory/streams for measurement; mAP metric; GPU idle-gating | measurement only |

**Why ONNX in the middle at all?** PyTorch and TensorRT don't speak directly for quantized graphs.
ONNX's `QuantizeLinear`/`DequantizeLinear` are the **standard, portable way** to say "this tensor is
INT8 at this scale" — modelopt emits them, TensorRT consumes them.

📖 ONNX Q/DQ: https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html
📖 TensorRT explicit Q/DQ: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html

---

## 5. The "who does what" map — the whole stack in one line each

```text
PyTorch      → THE FOUNDATION: tensors, autograd (incl. the STE gradient), nn.Module. Everything runs on it.
Ultralytics  → THE TRAINING HARNESS: YOLO model definition + the fine-tune loop (loss, augmentation, LR, EMA, val).
modelopt     → THE QUANTIZER: rewrites the model to fake-quant (QAT), calibrates scales, exports Q/DQ ONNX.
ONNX         → THE BRIDGE: portable graph with Q/DQ nodes carrying the INT8 scales — PyTorch → TensorRT.
TensorRT     → THE ENGINE: reads Q/DQ, fuses into real INT8 kernels, builds + runs the fast deployed model.
```

Or as a sentence you can teach from:
> **PyTorch** holds the model and does the math; **Ultralytics** trains it; **modelopt** makes that
> training quantization-aware and writes the result as **ONNX** Q/DQ; **TensorRT** turns that ONNX
> into the real INT8 engine that ships.

---

## 6. The single most important mental correction
Adding modelopt did **not** replace PyTorch or Ultralytics or fork the training — it **inserted a thin
quantization layer inside the existing PyTorch model**. The model is still a normal PyTorch/Ultralytics
model that trains normally; modelopt just made its convolutions *simulate INT8* during training and
gave us a way to *export* that to TensorRT. Everything else in the stack is unchanged.

📖 Docs to teach from: modelopt QAT guide
(https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#quantization-aware-training-qat)
· `cnn_qat` example (https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/cnn_qat)
· PyTorch autograd/STE background (arXiv:1308.3432)
· Ultralytics trainer (https://docs.ultralytics.com/reference/engine/trainer/)
· TensorRT Q/DQ (https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html)
