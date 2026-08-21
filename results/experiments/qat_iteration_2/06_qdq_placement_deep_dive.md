# Q/DQ Placement — Complete Reference (review, run nothing)

How quantizers are positioned throughout YOLO26n, from our code + modelopt behavior, with counts
verified from our recorded provenance (608 → 253 → 207) and rules verified against the modelopt docs.

Verified numbers (v6 provenance):
```text
quantizers_inserted      608     quantizers_with_scales   253
QuantizeLinear nodes     207     DequantizeLinear nodes   207   (= 207 Q/DQ pairs)
onnx_nodes_total        1477     num_outputs                1     opset 17     cfg INT8_DEFAULT_CFG
```

---

# PART 1 — Every import, grouped by role

### Foundation
| Import | Is | Use in our pipeline | Targets | Key symbols | Version · venv |
|---|---|---|---|---|---|
| `torch` | PyTorch DL framework | tensors, autograd (incl. STE), the model, `.half()`, dummy inputs | all stages | `torch.no_grad`, `torch.zeros`, `torch.onnx` (unused for QAT) | 2.7.0+cu128 · p311 |
| `numpy` | array math | preprocess arrays, letterbox | preprocess | `np.ascontiguousarray`, `np.full` | 1.26.4 · p311 |
| `cv2` (opencv) | image I/O | read + letterbox calib/eval frames | calibration input | `cv2.imread`, `cv2.resize` | 4.11 · p311 |
| stdlib | — | paths, json, hashing, timing | provenance | `pathlib`, `json`, `hashlib` | — |

### Training harness
| Import | Is | Use | Targets | Key symbols | Version · venv |
|---|---|---|---|---|---|
| `ultralytics.YOLO` | YOLO wrapper | load `best.pt`, hold the `DetectionModel` | training/export | `YOLO(path).model` | 8.4.90 · p311 |
| `ultralytics.models.yolo.detect.DetectionTrainer` | YOLO training loop | subclassed as `QATTrainer` — loss, aug, LR, EMA, val | training | `get_model()`, `train()`, `add_callback()` | 8.4.90 · p311 |
| `ultralytics.nn.modules.head.Detect` | detection head module | set `export=True` → single `[1,300,6]` | export | `Detect.export/format/dynamic` | 8.4.90 · p311 |

### Quantization (modelopt)
| Import | Is | Use | Targets | Key symbols | Version · venv |
|---|---|---|---|---|---|
| `modelopt.torch.quantization as mtq` | the QAT algorithm | **insert + calibrate** fake-quant | training — **weights + activations of Conv/Linear** | `mtq.quantize`, `mtq.INT8_DEFAULT_CFG`, `mtq.print_quant_summary` | 0.33.1 · p311 |
| `modelopt.torch.quantization.nn.TensorQuantizer` | the fake-quant module | count/inspect quantizers; the unit that rounds INT8↔float | training — each quantizer | `isinstance(m, TensorQuantizer)`, `m._amax` | 0.33.1 · p311 |
| `modelopt.torch.opt as mto` | modelopt state I/O | **save/restore** recipe+weights+scales | after-train / pre-export | `mto.save`, `mto.restore` | 0.33.1 · p311 |

### Export
| Import | Is | Use | Targets | Key symbols | Version · venv |
|---|---|---|---|---|---|
| `modelopt.torch._deploy.utils.torch_onnx` | modelopt ONNX deploy path | **TensorQuantizers → Q/DQ ONNX** | export — the graph | `get_onnx_bytes_and_metadata`, `OnnxBytes` | 0.33.1 · p311 |
| `onnx` | ONNX graph lib | read back + **verify** Q/DQ counts, output shape | export verify | `onnx.load`, `graph.node`, op_type counts | 1.17.0 · p311 |

### Build
| Import | Is | Use | Targets | Key symbols | Version · venv |
|---|---|---|---|---|---|
| `tensorrt as trt` | TensorRT builder/runtime | **build INT8 engine** from Q/DQ ONNX | build — the engine | `trt.Builder`, `OnnxParser`, `BuilderFlag.INT8`, `Runtime` | 10.16.1.11 · trt |

### Eval / measure (separate scripts, listed for completeness)
| Import | Is | Use | venv |
|---|---|---|---|
| `pycocotools` | COCO mAP metric | `COCOeval` mAP50/50-95 | trt |
| `cuda-python`/`cuda-bindings` | CUDA runtime bindings | device malloc/memcpy/stream for latency | trt |
| `pynvml` | NVML bindings | idle-gating the benchmark | trt |

---

# PART 2 — How Q/DQ gets placed

## 2.1 The placement rule (`INT8_DEFAULT_CFG`)
`mtq.quantize(model, INT8_DEFAULT_CFG, forward_loop)` walks `model.named_modules()` and matches each
against the config's `quant_cfg` **wildcard (`fnmatch`) patterns**. INT8_DEFAULT_CFG targets the
standard **quantizable module TYPES — `nn.Conv2d` (+ Conv1d/3d, ConvTranspose), `nn.Linear`** — and
for each match inserts **two `TensorQuantizer`s**:
```python
{"*weight_quantizer": {"num_bits": 8, "axis": 0}},      # WEIGHTS  → per-channel (verified)
{"*input_quantizer":  {"num_bits": 8, "axis": None}},   # ACTIVATION → per-tensor (verified)
{"default": {"num_bits": 8, "axis": None, "calibrator": "max", "fake_quant": True, "unsigned": False}}
```
📖 https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html
**Key consequence:** placement is decided by **module type**. Functional ops (`torch.matmul`,
`F.softmax`, `TopK`/`Mod`/`gather`) and non-Conv/Linear layers are **not modules modelopt wraps**, so
they **never get quantizers** — this alone explains the whole INT8-vs-float map (Part 3).

## 2.2 Where the two quantizers sit — walk one conv
```text
BEFORE  (plain):     x ─────────────► Conv2d(W) ─────► y
AFTER   (QuantConv2d):
                     x ─► [input_quantizer] ─►┐
                                              ├─► Conv(·, ·) ─► y
                        W ─► [weight_quantizer]┘
   input_quantizer  : PER-TENSOR  INT8 fake-quant on the activation feeding the conv
   weight_quantizer : PER-CHANNEL INT8 fake-quant on the conv's weights (axis=0)
   OUTPUT is NOT quantized here — the next layer's input_quantizer handles it (avoids double-quant)
```
So modelopt **replaces** `Conv2d` with a dynamic subclass `QuantConv2d`, **wraps** the same conv math,
and **adds** exactly those two `TensorQuantizer` submodules. Confirmed both-quantizers behavior in the
config docs (`*weight_quantizer` and `*input_quantizer` matched independently).
📖 TensorQuantizer: https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.nn.modules.tensor_quantizer.html

## 2.3 Why 608 — and 608 → 253 → 207
```text
608 TensorQuantizers inserted   = input_quantizer + weight_quantizer across ALL quantizable
                                  Conv/Linear layers (~2 per layer; incl. BOTH head branches —
                                  one2many[train] + one2one[deploy] duplicate the head convs)
        │ calibration (forward_loop) sets _amax where data/weights flow
        ▼
253 quantizers WITH scales      = the ACTIVE quantizers that carry a calibrated _amax
        │ EXPORT: (a) aux-head pruning — Detect.export=True drops the training-only one2many
        │         head, removing its quantizers;  (b) only active quantizers emit Q/DQ
        ▼
207 QuantizeLinear + 207 DequantizeLinear  = 207 Q/DQ PAIRS in the deployment ONNX
```
(The exact per-layer split is printable via `mtq.print_quant_summary(model)` — not run here.)

## 2.4 How each quantizer becomes a Q/DQ pair (export)
`get_onnx_bytes_and_metadata` traces the fake-quant model; each **active** `TensorQuantizer` emits a
**`QuantizeLinear` → `DequantizeLinear`** pair carrying its frozen scale as a constant:
```text
per deployed conv:  x → [Q → DQ]activation → Conv( ·, [Q → DQ]weights ) → y
   activation Q/DQ : per-tensor scale (axis=None)
   weight     Q/DQ : per-channel scale (axis=0), one scale per output filter
```
≈2 Q/DQ pairs per deployed conv → ~103 convs → **207 pairs**. `Detect.export=True` also collapses the
11 raw heads to the single `[1,300,6]` output (verified `num_outputs=1`).
📖 ONNX Q/DQ: https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html

## 2.5 How TensorRT reads placement (build)
`config.set_flag(trt.BuilderFlag.INT8)`, **no calibrator** — TensorRT reads the scales from the Q/DQ
nodes (explicit quantization):
- A conv **with Q/DQ on both its activation input and weights** → **fused into a single INT8 kernel**
  (Q→conv→DQ folds into the kernel's requantization).
- A layer **without Q/DQ** (attention matmul/softmax, NMS ops) → runs **FP32** (or FP16 if that flag
  helps).
- **Reformats** ("Reformatting CopyNode") are inserted at every **INT8↔float boundary** — where a
  quantized conv's output feeds an un-quantized region (e.g. backbone conv → attention block, or
  neck → head index ops). They convert precision/layout and are pure overhead.
📖 https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html

---

# PART 3 — What each element targets in OUR YOLO26n

## 3.1 Gets quantizers (→ INT8 in engine)
| Region | Layers | Why they're quantized |
|---|---|---|
| Backbone convs (model.0–9) | 60 | `nn.Conv2d` modules → matched by `INT8_DEFAULT_CFG` |
| Neck/PAN convs (model.11–21) | 50 | same |
| Attention **convs** (qkv/pe in model.10, model.22) | 4 | they ARE `nn.Conv2d` → quantized |
| Head conv branches (model.23 cls/box) | ~17 | `nn.Conv2d` |

## 3.2 Does NOT get quantizers (→ stays float) — and why
| Region | Ops | Why NOT quantized |
|---|---|---|
| Attention core | `softmax`, Q·K / A·V **matmul**, muls | **Functional ops** (`F.softmax`, `torch.matmul`), **not `nn.Module`s** → modelopt never wraps them. (Also numerically hostile to INT8 — we measured a −0.30 mAP collapse when forced.) |
| NMS-free head | `TopK`, `Tile`, `Mod`, `gather` | **Integer / index ops**, not Conv/Linear and not floating-point math → no quantizer applies; inherently int32/int64 |
| Activations / norms | SiLU, BN | not separately quantized (BN folds into conv; SiLU is functional) |

**The rule in one line:** modelopt quantizes **Conv/Linear *modules*** → everything that isn't one
(functional matmul/softmax, index ops, activations) is left float **by construction**, not by choice.

## 3.3 Levers to REPOSITION Q/DQ (for the sensitivity experiments)
Verified from the modelopt docs — placement is changed via the **`quant_cfg` list**, not a standalone
function:
```python
# copy the default and DISABLE quantizers on matched layers (they then run float)
import copy
cfg = copy.deepcopy(mtq.INT8_DEFAULT_CFG)
cfg["quant_cfg"].append({"quantizer_name": "*model.23*", "enable": False})     # e.g. keep head float
cfg["quant_cfg"].append({"quantizer_name": "*input_quantizer",
                         "parent_class": "SomeModule", "enable": False})        # class-scoped
model = mtq.quantize(model, cfg, forward_loop)
```
- **`"enable": False`** + a **`quantizer_name`** wildcard (fnmatch) disables specific quantizers →
  those layers run float. `**parent_class**` scopes by module type. API: **`set_quantizer_by_cfg`**.
  (There is **no** `mtq.disable_quantizer()` — config-based only.)
- **What we CAN do for experiments:** disable quantizers per layer/region (e.g. "head float," "first N
  backbone convs float"), change `axis` (per-channel↔per-tensor), `num_bits`, or `calibrator` per
  pattern — all via a custom config. This is exactly the lever for Sweeps 3–4 and any
  selective-quantization sensitivity study.
- **What we CANNOT do via config:** *add* quantizers to functional ops (softmax/matmul/index) — they
  aren't modules; quantizing them would require custom module replacement (out of scope).
📖 https://nvidia.github.io/Model-Optimizer/guides/_quant_cfg.html · `set_quantizer_by_cfg`
(https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.conversion.html)

---

## One-paragraph summary
`mtq.quantize` matches every **`nn.Conv2d`/`nn.Linear`** by wildcard and wraps each in a `QuantConv2d`
holding a **per-tensor input quantizer + per-channel weight quantizer** — **608** total across both
head branches. Calibration sets scales on **253** active quantizers; export prunes the training-only
one2many head and emits **207 Q/DQ pairs** (each ≈ activation-Q/DQ + weight-Q/DQ around a conv).
TensorRT fuses Q→conv→DQ into INT8 kernels, leaves the **functional** attention/NMS ops float (they're
not modules and can't take a quantizer), and inserts **reformats** at the INT8↔float seams. Placement
is fully controllable via the **`quant_cfg` `"enable": False` / `axis` / `parent_class`** levers — the
handle for the sensitivity sweeps.
