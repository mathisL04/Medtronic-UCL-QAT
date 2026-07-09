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
docs/02_onnx_export.md                  ONNX export notes
docs/03_tensorrt_fp16.md                TensorRT FP16 notes
docs/04_tensorrt_int8_ptq.md            INT8 post-training quantisation notes
docs/05_qat.md                          quantisation-aware training notes
```

## Current Important Files

```text
scripts/build_sanoscience_yolo_full_cork.py
scripts/train_sanoscience_yolo_full_cork.py
configs/sanoscience_yolo_cork.yaml
models/yolo26n_sanoscience_full_left/best.pt
models/yolo26n_sanoscience_full_left/model_card.md
reports/yolo26n_sanoscience_full_left/results.csv
reports/yolo26n_sanoscience_full_left/results.png
reports/yolo26n_sanoscience_full_left/confusion_matrix.png
```

## Next Steps

```text
1. PyTorch baseline validation
2. PyTorch latency benchmark
3. ONNX export
4. ONNX validation
5. TensorRT FP16 engine
6. TensorRT INT8 post-training quantisation
7. Quantisation-aware training
8. Accuracy / latency / memory comparison
```
