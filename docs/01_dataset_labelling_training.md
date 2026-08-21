# Stage 1: Dataset, Auto-Labelling and Baseline Training

How the Open-H Sanoscience surgical videos became a YOLO detection dataset, and how
the YOLO26n baseline that every later stage is measured against was trained.

Machine and environment: **Cork**, `~/venvs/medtronic-qats`. See
[`00_environment_and_access.md`](00_environment_and_access.md).

The PyTorch latency baseline that used to live at the end of this document has moved
to [`07_pytorch_latency.md`](07_pytorch_latency.md), alongside the rest of the
PyTorch-side timing work.

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
---

## Automatic Labelling / Prelabelling Process

Main script:

```text
scripts/data/build_sanoscience_yolo_full_cork.py
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
---

## Parallel Dataset-Building Workers

Prelabelling is mostly CPU, ffmpeg, disk, and network I/O bound. It does not rely on the GPU.

The 1602 episodes were split across multiple independent workers. Each worker processed a non-overlapping episode range.

Worker launch pattern:

```bash
nohup nice -n 10 env START_EPISODE_INDEX=<START> END_EPISODE_INDEX=<END> SHARD_NAME=<NAME> \
python -u scripts/data/build_sanoscience_yolo_full_cork.py \
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
---

## Training Process

Main training script:

```text
scripts/train/train_sanoscience_yolo_full_cork.py
```

Training command:

```bash
mkdir -p /home/zcemml1/medtronic_qat_data/runs_sanoscience/logs

nohup python -u scripts/train/train_sanoscience_yolo_full_cork.py \
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
models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.pt
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

**Validation confidence threshold.** mAP must be measured at `conf=0.001`, not at a deployment threshold. `conf=0.25` truncates the precision-recall curve before its low-confidence tail and under-reports mAP50 by 4.9 points on the 100-image subset and 5.4 points on the full set. Precision and recall are unaffected, because both are reported at max-F1, which sits at high confidence. Measured across both evaluation sets and both thresholds:

```text
eval                       P        R      mAP50   mAP50-95
subset-100   conf=0.25   0.9535   0.8657   0.8915    0.7340
subset-100   conf=0.001  0.9535   0.8657   0.9408    0.7673
full-6449    conf=0.25   0.9385   0.8644   0.8797    0.7406
full-6449    conf=0.001  0.9385   0.8644   0.9340    0.7815
```

The reported baseline is the last row. At matched `conf=0.001` the subset and the full set agree to 0.7 points of mAP50 and 1.4 points of mAP50-95 — so the earlier assumption that the 100-image subset carried roughly 4 points of sampling noise was mostly this threshold artefact, not sampling.

**Validation throughput (batched, Cork V100).** Ultralytics' own `model.val()` speed printout from the baseline validation pass — per-image time amortised over the validation batch, from Ultralytics' internal speed counter, on the Cork V100:

```text
0.1 ms preprocess
0.8 ms inference
0.1 ms postprocess
per image
```

This is a **throughput** number, **not** the deployment latency, and is not comparable to the single-frame figures below. For the reported latency baseline see **Latency benchmarking**.

---
