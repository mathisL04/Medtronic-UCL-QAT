# Medtronic x UCL QAT Research

This repository contains the code, configs, baseline model, and experiment outputs for the Medtronic x UCL summer research project on computer vision model optimisation for surgical tool detection.

The project pipeline is:

```text
Open-H Sanoscience surgical videos
        ↓
automatic segmentation-based YOLO labelling
        ↓
YOLO26n baseline training
        ↓
ONNX export
        ↓
TensorRT FP32 inference (V2 baseline)
        ↓
TensorRT FP16 inference
        ↓
TensorRT INT8 / PTQ
        ↓
QAT and accuracy-latency comparison
```

## Current Status

Completed:

- Sanoscience dataset prelabelling
- YOLO-format dataset generation
- YOLO26n baseline training on UCL Cork GPU cluster
- Baseline model checkpoint saved
- Training metrics and plots saved
- FP32 latency benchmark on verified-idle GPU
- FP32 ONNX export with PyTorch↔ONNX parity gate
- ONNX accuracy validated against the PyTorch baseline
- TensorRT FP32 engine (V2 baseline): built, parity-validated, mAP measured
- TensorRT FP16 engine (V3): built, mAP measured (-0.0002 mAP50 vs V2)
- Paired V2/V3 latency benchmark (1.49x inference, 1.28x end-to-end)
- Engine mAP tooling (pycocotools, ONNX and engine through identical metric code)

Current baseline:

```text
Model: YOLO26n
Task: single-class surgical tool detection
Class: surgical_tool
Validation images: 6,449
Validation instances: 14,517
Precision: 0.937
Recall: 0.866
mAP50: 0.934
mAP50-95: 0.782
```

FP32 ONNX export (validated, see docs/02):

```text
File: models/yolo26n_sanoscience_full_left/best.onnx
Format: ONNX opset 17, static [1, 3, 640, 640], FP32
Parity vs PyTorch (16 frames, CPU): PASS, max coord diff 1.98e-04 px
Accuracy (val100 subset, conf 0.001):
  PT    mAP50 0.9408   mAP50-95 0.7673
  ONNX  mAP50 0.9394   mAP50-95 0.7595
```

TensorRT FP32 engine "V2 baseline" (validated, see docs/03):

```text
File: models/yolo26n_sanoscience_full_left/best_fp32.engine  (GPU-specific, gitignored)
Build: TensorRT 10.16.1.11, A100, true FP32 (TF32 disabled)
Parity vs ONNX (val100): 100/100 PASS, max coord diff 1.831e-04 px
Accuracy (measured, pycocotools @ conf 0.001): mAP50 0.9350 / mAP50-95 0.7572
Latency (batch=1, GPU 0): 3.736 ms median (267.7 FPS), inference 2.318 ms
  vs PyTorch 8.642 ms -> 2.31x faster end-to-end
```

TensorRT FP16 engine "V3" (validated, see docs/04):

```text
File: models/yolo26n_sanoscience_full_left/best_fp16.engine  (GPU-specific, gitignored)
Build: TensorRT 10.16.1.11, A100, FP16 (TF32 disabled -- precision is the only
       difference from V2; every other build field is identical)
Size: 7.0 MB  vs  V2 12.8 MB
Parity vs FP32 ONNX: 77/100 FAIL at FP32 tolerances -- expected for half precision,
       and NOT the accuracy claim (see docs/04)
Accuracy (measured, pycocotools @ conf 0.001): mAP50 0.9348 / mAP50-95 0.7572
       -> vs V2: -0.0002 mAP50, 0.0000 mAP50-95
Latency (batch=1, GPU 0, same conditions as V2): 2.922 ms median (342.2 FPS)
       inference 1.561 ms -> 1.49x vs V2 inference; 1.28x end-to-end
Note: both latency runs are non-exclusive (2 dormant contexts tolerated,
       exclusive_gpu: false). See docs/03 "Gate relaxation".
```

Accuracy note: V2/V3 figures are measured with pycocotools, the docs/02 PT/ONNX
figures with Ultralytics. On the same ONNX the two metrics differ by 0.0044 mAP50,
so precision comparisons are made within the pycocotools column only.

## Repository Structure

```text
scripts/      active and archived project scripts
configs/      YAML configuration files
models/       small saved checkpoints and model cards
reports/      training metrics, plots, and summaries
docs/         detailed project documentation
```

## Main Documentation

```text
docs/01_dataset_labelling_training.md   dataset creation and YOLO26n training
docs/02_onnx_export.md                  ONNX export + PyTorch/ONNX parity
docs/03_tensorrt_fp32.md                TensorRT FP32 engine (V2 baseline)
docs/04_tensorrt_fp16.md                TensorRT FP16 engine (V3)
docs/05_tensorrt_int8_ptq.md            INT8 post-training quantisation notes
docs/06_qat.md                          quantisation-aware training notes
```

## Current Important Files

```text
scripts/build_sanoscience_yolo_full_cork.py
scripts/train_sanoscience_yolo_full_cork.py
scripts/export_onnx.py
scripts/build_tensorrt_engine.py
scripts/validate_engine_parity.py
scripts/benchmark_latency_trt.py
scripts/evaluate_engine_map.py
configs/sanoscience_yolo_cork.yaml
models/yolo26n_sanoscience_full_left/best.pt
models/yolo26n_sanoscience_full_left/best.onnx
models/yolo26n_sanoscience_full_left/best.onnx.provenance.json
models/yolo26n_sanoscience_full_left/model_card.md
reports/yolo26n_sanoscience_full_left/results.csv
reports/yolo26n_sanoscience_full_left/results.png
reports/yolo26n_sanoscience_full_left/confusion_matrix.png
```

## TensorRT tooling (general, all precisions)

The three `*_tensorrt_*` / `*_engine_*` scripts above are **not** FP32-specific — they are the shared TensorRT tooling for every precision stage (FP32, FP16, INT8). Precision is a run-time knob passed at the command line, never a code change:

```text
scripts/build_tensorrt_engine.py     ONNX -> .engine.  PRECISION={fp32|fp16} selects precision.
scripts/validate_engine_parity.py    engine vs ONNX detection parity.  ENGINE_PATH selects engine.
scripts/benchmark_latency_trt.py     batch=1 idle-gated latency.        ENGINE_PATH selects engine.
scripts/evaluate_engine_map.py       engine mAP (pycocotools) on val100. ENGINE_PATH / MODE.
```

### Choosing FP32 vs FP16

The precision is set by the `PRECISION` variable in front of the **same** build command — same script, same `best.onnx`, only the value changes. Each run writes a precision-named engine (`best_fp32.engine`, `best_fp16.engine`), so both can coexist:

```bash
# FP32 engine (V2 baseline)
PRECISION=fp32 DEVICE=<idle_gpu> python scripts/build_tensorrt_engine.py

# FP16 engine
PRECISION=fp16 DEVICE=<idle_gpu> python scripts/build_tensorrt_engine.py
```

Then point the parity / latency / accuracy scripts at whichever engine with `ENGINE_PATH`:

```bash
ENG=models/yolo26n_sanoscience_full_left/best_fp16.engine   # or best_fp32.engine

ENGINE_PATH=$ENG DEVICE=<idle_gpu> python scripts/validate_engine_parity.py
ENGINE_PATH=$ENG DEVICE=<idle_gpu> python scripts/evaluate_engine_map.py
ENGINE_PATH=$ENG DEVICE=<idle_gpu> BENCHMARK_REPEATS=10 python scripts/benchmark_latency_trt.py
```

INT8 reuses these too, plus a calibration step added in its own stage.

## Next Steps

Done:

```text
1. PyTorch baseline validation
2. PyTorch latency benchmark
3. ONNX export
4. ONNX validation
5. TensorRT FP32 engine (V2 baseline): build + parity + accuracy + latency
6. TensorRT FP16 engine (V3): build + parity + accuracy + latency
```

Remaining:

```text
7. TensorRT INT8 post-training quantisation
8. Quantisation-aware training
9. Accuracy / latency / memory comparison
```
