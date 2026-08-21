# QAT Method Reference — our pipeline, dependencies, hyperparameters, docs

Definitive reference for the QAT model in this repo, drawn directly from the code
(`scripts/train/train_qat.py`, `scripts/train/qat_run.sh`, `scripts/export/export_qat_onnx.py`,
`scripts/tensorrt/build_tensorrt_int8_qdq.py`), the recorded provenance, and the installed venvs.
Read-only; nothing here changes the code.

---

## 1. Method — definitively

**NVIDIA Model Optimizer (modelopt) QAT.** Not `pytorch_quantization` (superseded predecessor),
not `torchao`/`torch.ao.quantization` (PyTorch-native, targets ExecuTorch/torch.compile — wrong
deployment target for us; we deploy on **TensorRT**).

- **Quantization is Quantization-Aware Training (QAT), not PTQ** — the distinguishing step is the
  **fine-tune** (`trainer.train()`). PTQ would stop after calibration; we continue training.
- **Fake-quant mechanism:** modelopt's `TensorQuantizer` does float-domain fake quantization
  (`x_fq = (round(x/scale + zp) − zp) * scale`, staying float32) with a **straight-through
  estimator (STE)** for gradients.
- **Scales:** `_amax` is a **buffer, not a parameter** → calibrated once, then **frozen**; the
  fine-tune adapts the **weights** to the fixed quantization grid.
- **Deployment:** exported to **ONNX with Q/DQ** (QuantizeLinear/DequantizeLinear) nodes →
  **explicit-quantization** TensorRT engine (scales read from Q/DQ, **no calibrator**).

---

## 2. Pipeline — end to end (functions + files)

```text
STEP              FUNCTION / CALL                                          FILE                          OWNER
1 LOAD            YOLO("baseline/best.pt")                                 train_qat.py                  ultralytics
2 INSERT+CALIB    mtq.quantize(model, mtq.INT8_DEFAULT_CFG, forward_loop)  train_qat.py:207              modelopt
                  → inserts 608 TensorQuantizers; forward_loop runs
                    N_CALIB train frames to set initial scales (amax).
                    Runs inside QATTrainer.get_model(), BEFORE ModelEMA.
3 FINE-TUNE       trainer.train()   (QATTrainer(DetectionTrainer))         train_qat.py:363              ultralytics
                  → fake-quant ON, amp=False, STE backprop; weights          + modelopt STE
                    fine-tuned (scales frozen). Early-stop on mAP50-95.
4 SAVE STATE      mto.save(model, "qat_modelopt_state.pt")                 train_qat.py:245              modelopt
                  + best-epoch callback save_best_qat → *_best.pt          train_qat.py:290
5 VERIFY RELOAD   mto.restore(fresh, state) → assert 608 quantizers        train_qat.py:260              modelopt
6 EXPORT ONNX     YOLO(base) → mto.restore(state) → Detect.export=True     export_qat_onnx.py            modelopt
                  → get_onnx_bytes_and_metadata(model, dummy, opset=17)    export_qat_onnx.py:120
                  → ONNX with QuantizeLinear/DequantizeLinear (Q/DQ)
7 BUILD ENGINE    trt.Builder → parse Q/DQ ONNX →                          build_tensorrt_int8_qdq.py    TensorRT
                  config.set_flag(trt.BuilderFlag.INT8), NO calibrator     build_tensorrt_int8_qdq.py:72
                  → INT8 engine (scales from Q/DQ)
```

**Orchestration:** `scripts/train/qat_run.sh` runs step 1–5 on local scratch (`/tmp/zcemml1_qat`,
off the NFS quota) and copies the durable artifacts (`qat_modelopt_state.pt`, `*_best.pt`,
`qat_provenance.json`, `results.csv`, `args.yaml`) back to `runs_qat/<RUN_NAME>` on exit.

**Durable artifact:** the modelopt **state file** (`qat_modelopt_state_best.pt`), NOT a `best.pt` —
Ultralytics' `torch.save` cannot pickle modelopt's dynamic `QuantConv2d`, so `save=False` and the
state is written by `mto.save` and reloaded via `mto.restore` onto the committed baseline.

---

## 3. Dependencies — exact installed versions (read from the venvs)

### Environment A — `~/venvs/medtronic-qat-p311` (steps 1–6: train + ONNX export)
```text
python                3.11.13
nvidia-modelopt       0.33.1        (+ nvidia-modelopt-core 0.33.1)   ← the QAT toolkit
torch                 2.7.0+cu128   torchvision 0.22.0+cu128
ultralytics           8.4.90        (+ ultralytics-thop 2.0.20)       ← training harness
onnx                  1.17.0        ← pinned; 1.22 removed onnx.reference.custom_element_types
onnxruntime-gpu       1.22.0
onnxslim/onnxsim      onnxsim 0.6.5 · onnxscript 0.7.1 · onnx_graphsurgeon 0.6.1 · onnx-ir 0.2.1
numpy                 1.26.4        ← pinned <2 (numpy 2.x breaks modelopt importer)
opencv-python         4.11.0.86
CUDA runtime          nvidia-cuda-runtime-cu12 12.8.57 · cudnn 9.7.1.26
```

### Environment B — `~/venvs/medtronic-trt` (step 7 + measurement)
```text
python                3.9.25
tensorrt              10.16.1.11    (cu13 build)                      ← engine build
cuda-python           13.0.3        cuda-bindings 13.0.3
numpy                 2.0.2
onnx                  1.19.1        onnxruntime 1.19.2
pycocotools           2.0.11        ← mAP metric
pynvml                13.0.1        ← idle-gating in benchmarks
opencv-python-headless 5.0.0.93
```

> Two separate venvs by design: modelopt 0.31+ needs Python ≥3.10 (→ 3.11), while the TensorRT
> stack lives in the 3.9 image. The ONNX Q/DQ file is the hand-off between them.

**Key module imports (from the code):**
```python
import modelopt.torch.quantization as mtq                 # mtq.quantize, INT8_DEFAULT_CFG
import modelopt.torch.opt as mto                           # mto.save / mto.restore
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch._deploy.utils.torch_onnx import get_onnx_bytes_and_metadata, OnnxBytes
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import tensorrt as trt                                      # build venv
```

---

## 4. Hyperparameters & variables

### 4a. `train_qat.py` — env-var knobs (runtime; each has a default)
| Var | Default | Meaning |
|---|---|---|
| `DEVICE` | **required** (no default) | GPU index; script aborts if unset |
| `EPOCHS` | 1 | training epochs (ceiling if early-stopping on) |
| `PATIENCE` | 0 = OFF | early-stop after N epochs with no mAP50-95 improvement |
| `LR0` | 1e-3 | initial LR (~1% of baseline's 0.01 — model already converged) |
| `LRF` | 0.01 | final-LR fraction (scheduler decay tail) |
| `BATCH` | 16 | batch size |
| `WORKERS` | 4 | dataloader workers (**use 0** on this host — fork OOM under strict overcommit) |
| `IMG_SIZE` | 640 | input resolution |
| `N_CALIB` | 128 | warm-start calibration frames (1 per episode, seeded) |
| `CALIB_SEED` | 42 | calibration sampling seed |
| `RUN_NAME` | qat_smoke | run/output name |
| `PROJECT` | /tmp/zcemml1_qat | output dir (local scratch via qat_run.sh) |

### 4b. `train_qat.py` — fixed constants
```text
MODEL_PATH   models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.pt
DATA_YAML    .../sanoscience_yolo_full_nonexpert_stereo/sanoscience_yolo.yaml
TRAIN_IMAGES .../images/train                    (warm-start source)
QUANT CONFIG mtq.INT8_DEFAULT_CFG
```

### 4c. `train_qat.py` — Ultralytics overrides baked in (the 3 HAZARDS)
```text
amp   = False   HAZARD 2 — fake-quant + FP16 autocast clash (deviation from baseline recipe)
save  = False   HAZARD 3 — Ultralytics torch.save can't pickle modelopt QuantConv2d
val   = True    validate each epoch (drives early-stopping + best-epoch callback)
plots = False   exist_ok = True   single-GPU (CUDA_VISIBLE_DEVICES = DEVICE)
[structural]    quantize BEFORE ModelEMA — get_model():292 vs ModelEMA:373  (HAZARD 1)
```

### 4d. The V6 (final) run — actual values used
```text
EPOCHS=50  PATIENCE=10  DEVICE=1  WORKERS=0  RUN_NAME=qat_v6  BATCH=16
LR0=1e-3  LRF=0.01  N_CALIB=128  CALIB_SEED=42  quant=INT8_DEFAULT_CFG  amp=False
→ stopped by PLATEAU (not the 50-epoch ceiling); 608 quantizers (253 with scales)
→ result: mAP50 0.9321 / mAP50-95 0.7644  (full 6,449 val)
```

### 4e. `export_qat_onnx.py` — variables
```text
BASE_MODEL   baseline/best.pt              (architecture frame)
QAT_STATE    qat/v6_final/qat_modelopt_state_best.pt   (env QAT_STATE)
OUT_ONNX     qat/v6_final/best_qat.onnx    (env OUT_ONNX)
IMG_SIZE 640   BATCH 1   OPSET 17   DEVICE (required: cpu or GPU idx)
head mode:   Detect.export=True, format="onnx", dynamic=False → single [1,300,6]
verify gate: QuantizeLinear>0 AND DequantizeLinear>0 AND exactly 1 output
```

### 4f. `build_tensorrt_int8_qdq.py` — variables
```text
ONNX_PATH    qat/v6_final/best_qat.onnx    (env ONNX_PATH)
ENGINE_PATH  qat/v6_final/best_qat_int8.engine  (env ENGINE_PATH)
WORKSPACE_GB 8   DEVICE (required)
flags:       config.set_flag(trt.BuilderFlag.INT8)   — NO calibrator (explicit Q/DQ)
```

### 4g. Fixed I/O contract (all stages)
```text
input  [1,3,640,640]   output [1,300,6] = x1,y1,x2,y2,conf,cls   class {0: surgical_tool}
imgsz 640   opset 17   conf 0.001 (mAP)   maxDets 300
```

---

## 5. Documentation sources — QAT (verified Aug 2026; docs rebranded to `Model-Optimizer`)

### ✅ Our method (modelopt QAT) — read these
| Resource | URL | Covers step |
|---|---|---|
| QAT guide section | https://nvidia.github.io/Model-Optimizer/guides/_pytorch_quantization.html#quantization-aware-training-qat | 2–3 (mtq.quantize + fine-tune; "~10% of original epochs") |
| **CNN/vision QAT example** (`torchvision_qat.py`) | https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/cnn_qat | 2–6 — the closest template to our pipeline |
| Save / Restore guide | https://nvidia.github.io/Model-Optimizer/guides/2_save_load.html | 4–5 (mto.save/restore) |
| Docs home | https://nvidia.github.io/Model-Optimizer/ | overview |
| Repo | https://github.com/NVIDIA/Model-Optimizer | all |

### ✅ TensorRT (explicit Q/DQ build) — read these
| Resource | URL | Covers step |
|---|---|---|
| Working with Quantized Types (explicit Q/DQ, no calibrator) | https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html | 7 |
| TensorRT Python API (Builder, BuilderFlag.INT8, OnnxParser) | https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/python-api/ | 7 |

### ✅ Ultralytics (training harness)
| Resource | URL | Covers step |
|---|---|---|
| Train mode (args: amp, patience, lr0/lrf) | https://docs.ultralytics.com/modes/train/ | 1, 3 |
| Trainer internals (DetectionTrainer / get_model / ModelEMA) | https://docs.ultralytics.com/reference/engine/trainer/ | 2–3 |

### ⛔ Do NOT use for our method
| Resource | Why |
|---|---|
| `examples/onnx_ptq`, `examples/hf_ptq`, `examples/llm_ptq` | **PTQ**, not QAT |
| https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/ | **old `pytorch_quantization` toolkit** — superseded by modelopt; different API |
| https://docs.pytorch.org/ao/ (torchao) & https://pytorch.org/docs/stable/quantization.html | **PyTorch-native** quantization — targets ExecuTorch/torch.compile, not TensorRT |

> Names like `TensorQuantizer`/`QuantConv2d`/Q-DQ appear in several of these — read the **modelopt**
> pages (our API is `mtq.quantize` + `mto.save/restore`), and the **QAT** example (`cnn_qat`), not
> anything named `*_ptq`.
