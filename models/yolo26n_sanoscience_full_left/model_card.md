# YOLO26n Sanoscience Surgical Tool Detector

This checkpoint is a YOLO26n model fine-tuned on the Open-H Embodiment Sanoscience surgical dataset.

## Model

- Architecture: YOLO26n
- Task: object detection
- Number of classes: 1
- Class name: `surgical_tool`
- Input size: 640
- Checkpoint: `baseline/best.pt`

## Dataset

- Source: Open-H Embodiment Sanoscience `nonexpert_stereo`
- View: left stereo camera only
- Train images: 20,756
- Validation images: 6,449
- Validation instances: 14,517

The generated YOLO dataset is not stored in GitHub. It remains on UCL Cork storage.

## Training

- Initial checkpoint: `yolo26n.pt`
- Epochs: 50
- Batch size: 16
- Workers: 8
- Hardware: UCL Cork, Tesla V100-PCIE-16GB
- Runtime: 3.816 hours
- AMP: enabled

## Best Validation Result

- Precision: 0.937
- Recall: 0.866
- mAP50: 0.934
- mAP50-95: 0.782

## Purpose

This trained model is the baseline for later optimisation:

- ONNX export
- TensorRT FP16 inference
- TensorRT INT8 post-training quantisation
- Quantisation-aware training
- Embedded GPU deployment benchmarking
