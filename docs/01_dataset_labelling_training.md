# Medtronic x UCL QAT Research

This repository contains the working code and results for the Medtronic x UCL summer research project on model optimisation, quantisation-aware training, and TensorRT deployment for surgical computer vision models.

Current baseline:

- Model: YOLO26n
- Task: single-class surgical tool detection
- Dataset: Open-H Embodiment Sanoscience surgical dataset
- Later target: ONNX export, TensorRT FP16, TensorRT INT8/PTQ, QAT, embedded GPU deployment

---

## Repository Structure

```text
Medtronics-UCL-QAT/
├── scripts/
│   ├── build_sanoscience_yolo_full_cork.py
│   ├── train_sanoscience_yolo_full_cork.py
│   ├── build_sanoscience_yolo_full.py
│   ├── train_sanoscience_yolo_full.py
│   ├── evaluate_sanoscience_predictions.py
│   └── archive/
│       ├── exploration/
│       ├── prototypes/
│       └── utenn/
├── configs/
│   └── sanoscience_yolo_cork.yaml
├── models/
│   └── yolo26n_sanoscience_full_left/
│       ├── best.pt
│       └── model_card.md
├── reports/
│   └── yolo26n_sanoscience_full_left/
│       ├── results.csv
│       ├── results.png
│       └── confusion_matrix.png
└── README.md
```

The generated YOLO dataset is not stored in GitHub because it is several GB. It remains on UCL Cork storage.

---

## UCL Cork GPU Cluster Access

The full dataset creation and training were run on the UCL EEE Cork GPU server.

From the local Ubuntu terminal, first connect to the UCL EEE gateway:

```bash
ssh -4 \
  -o GSSAPIAuthentication=no \
  -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password \
  zcemml1@ssh.ee.ucl.ac.uk
```

Then connect to Cork:

```bash
ssh -4 \
  -o GSSAPIAuthentication=no \
  -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password \
  zcemml1@cork.ee.ucl.ac.uk
```

Activate the project environment:

```bash
cd ~/medtronic_qat/Medtronics-UCL-QAT
source ~/venvs/medtronic-qats/bin/activate
```

GPU used:

```text
NVIDIA Tesla V100-PCIE-16GB
CUDA device 0
```

GPU check:

```bash
nvidia-smi
```

PyTorch CUDA check:

```bash
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU count:", torch.cuda.device_count())
print("GPU 0:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

---

## Dataset

Dataset source:

```text
nvidia/PhysicalAI-Robotics-Open-H-Embodiment
```

Subset used:

```text
Surgical/sanoscience/sanoscience_v1_2_merged/nonexpert_stereo
```

The videos are stereo side-by-side. This baseline uses the left stereo view only.

Final YOLO dataset location on Cork:

```text
/home/zcemml1/medtronic_qat_data/datasets/sanoscience_yolo_full_nonexpert_stereo
```

Final generated dataset:

```text
Train images: 20,756
Train labels: 20,756
Val images:   6,449
Val labels:   6,449
Total images: 27,205
Dataset size: approximately 4.0 GB
Class: surgical_tool
```

---

## Automatic Labelling / Prelabelling Process

Main script:

```text
scripts/build_sanoscience_yolo_full_cork.py
```

The dataset contains colour videos and corresponding segmentation videos. The segmentation videos highlight surgical tools in green. The labelling script converts these segmentation masks into YOLO bounding-box labels.

Pipeline:

```text
Hugging Face colour video
+
Hugging Face segmentation video
        ↓
ffmpeg frame extraction
        ↓
left stereo view crop
        ↓
detect tool segmentation colours
        ↓
binary tool mask
        ↓
connected components
        ↓
bounding boxes
        ↓
YOLO labels
```

Segmentation colours used:

```text
(96, 240, 0)
(96, 240, 112)
```

Generated YOLO structure:

```text
images/train/
images/val/
labels/train/
labels/val/
sanoscience_yolo.yaml
```

YOLO label format:

```text
class_id x_center y_center width height
```

For this dataset:

```text
class_id = 0 = surgical_tool
```

---

## Parallel Dataset-Building Workers

Prelabelling is mostly CPU, ffmpeg, disk, and network I/O bound. It does not rely on the GPU.

The 1602 episodes were split across multiple independent workers. Each worker processed a non-overlapping episode range.

Worker launch pattern:

```bash
nohup nice -n 10 env START_EPISODE_INDEX=<START> END_EPISODE_INDEX=<END> SHARD_NAME=<NAME> \
python -u scripts/build_sanoscience_yolo_full_cork.py \
> /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs/build_cork_<NAME>.log 2>&1 &
```

Meaning:

```text
nohup       keeps the process running after SSH disconnect
nice -n 10  lowers priority on the shared server
env         passes worker-specific variables
START       first episode index
END         final episode index, exclusive
SHARD_NAME  unique worker/temp/log name
&           runs in background
```

Workers used:

```text
s0:  episodes 0-399
s1:  episodes 400-799
s2:  episodes 800-999
s2b: episodes 1000-1199
s3b: episodes 1200-1601
```

Worker monitoring:

```bash
pgrep -af build_sanoscience_yolo_full_cork.py
```

---

## Training Process

Main training script:

```text
scripts/train_sanoscience_yolo_full_cork.py
```

Training command:

```bash
mkdir -p /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs

nohup python -u scripts/train_sanoscience_yolo_full_cork.py \
> /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs/train_yolo26n_full_cork.log 2>&1 &
```

Training config:

```text
Model: YOLO26n
Initial checkpoint: yolo26n.pt
Epochs: 50
Batch size: 16
Image size: 640
Workers: 8
Device: CUDA GPU 0
AMP: enabled
Hardware: Tesla V100-PCIE-16GB
Runtime: 3.816 hours
```

The `WORKERS = 8` setting refers to CPU dataloader workers, not 8 GPU jobs. These CPU workers load images, resize/augment them, prepare labels, and feed batches to the GPU.

```text
CPU dataloader workers → prepared batch → GPU training step
```

This reduces GPU idle time during training.

Training monitoring:

```bash
tail -f /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs/train_yolo26n_full_cork.log
```

GPU monitoring:

```bash
watch -n 1 nvidia-smi
```

CPU process monitoring:

```bash
ps -u $USER -o pid,ppid,pcpu,pmem,cmd | grep train_sanoscience_yolo_full_cork.py | grep -v grep
```

---

## Final Training Results

Training completed successfully:

```text
50 epochs completed in 3.816 hours.
```

Final Cork checkpoints:

```text
/home/zcemml1/medtronic_qat_data/runs_sanoscience/yolo26n_sanoscience_full_left/weights/best.pt
/home/zcemml1/medtronic_qat_data/runs_sanoscience/yolo26n_sanoscience_full_left/weights/last.pt
```

The best checkpoint was copied into this repository:

```text
models/yolo26n_sanoscience_full_left/best.pt
```

Validation set:

```text
Images:    6,449
Instances: 14,517
```

Best model metrics:

```text
Precision: 0.937
Recall:    0.866
mAP50:     0.934
mAP50-95:  0.782
```

Ultralytics validation speed on V100:

```text
0.1 ms preprocess
0.8 ms inference
0.1 ms postprocess
per image
```

---

## Next Steps

This trained YOLO26n model is the baseline for optimisation.

Planned stages:

```text
1. PyTorch baseline validation
2. PyTorch latency benchmark
3. ONNX export
4. ONNX validation
5. TensorRT FP16 engine
6. TensorRT INT8 post-training quantisation
7. Quantisation-aware training
8. Accuracy/latency comparison
```

Comparison metrics:

```text
Precision
Recall
mAP50
mAP50-95
Latency
FPS
Model size
GPU memory
Deployment compatibility
```
