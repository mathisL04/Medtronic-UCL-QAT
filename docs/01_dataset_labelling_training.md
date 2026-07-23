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

## Latency benchmarking

**Deployment latency (batch=1, Geneva A100) — reported baseline.** Single-frame inference timed with `scripts/benchmark_latency.py`: one image per `model.predict()` call (batch=1), wall-clock around each call, the 100-frame validation subset preloaded to RAM, warmup, then pooled per-stage **medians** over 10 repeats × 100 frames (1000 samples) on a verified-idle A100.

```text
preprocess    1.552 ms   (median)
inference     6.844 ms   (median)
postprocess   0.238 ms   (median)
total         8.642 ms   (median)  ->  115.7 FPS
```

(Stage figures are each the median of their own column, so they need not sum exactly to the median total. Here they differ by 0.008 ms: 1.552 + 6.844 + 0.238 = 8.634 against a reported total of 8.642.)

Run conditions:

```text
Date:        16 July 2026
Host:        Geneva
GPU:         A100-SXM4-80GB, GPU 3, explicitly pinned
Model:       YOLO26n, batch=1
Samples:     10 repeats x 100 frames = 1000
Seed:        42
Image size:  640
conf:        0.25
Warmup:      30 images
Priority:    nice -n 10
Contention:  none on any repeat (OtherProc=0, memory flat at 1407 MiB)
GPU peak:    30-33%, which is our own batch=1 load
```

Full distribution, in milliseconds:

```text
stage          mean     std   median     min     p95     p99     max
preprocess    1.674   0.325    1.552   1.493   2.457   2.490   4.488
inference     6.884   0.260    6.844   6.705   7.040   8.236  10.704
postprocess   0.245   0.085    0.238   0.229   0.259   0.295   2.738
total         8.803   0.446    8.642   8.442   9.600  10.390  12.574
```

Contention only ever adds time, so the mean is dragged upward while the median and the min estimate the clean machine. FPS is quoted median-based at **115.7**; the mean-based figure is 113.6, giving a range of 113.6-115.7. Every downstream comparison in this project is median-based, so 115.7 is the number the TensorRT stages are measured against.

The maximum matters for a surgical detector, but a max over 1000 samples is hostage to one bad frame, which is why p95 and p99 are reported alongside it — they separate a real tail from a single artefact.

**The tails are CPU-side.** Preprocess has p95 2.457 and max 4.488 against a 1.552 median, which is letterbox jitter. Postprocess carries one 2.738 ms outlier against a 0.238 median. Inference is tight by comparison — p95 7.040 against a 6.844 median. Any tail quoted for this baseline is a CPU artefact, not the network.

Batch=1 is the right unit for this project: surgical frames arrive one at a time, so the cost that matters is what a single live frame takes end to end — not throughput amortised over a batch. That is also why this total is ~8× the *Validation throughput* figure above: that one batches, this one does not. Hardware rules out the alternative explanation — the throughput figure was measured on the **slower V100 yet reads lower**, so the gap is batch size, not the GPU.

**The idle-GPU gate.** Before claiming the GPU, the benchmark refuses to run unless the target device is idle for *compute* — checked via `nvmlDeviceGetComputeRunningProcesses`, ignoring graphics contexts such as Xorg — and it samples per-repeat GPU state to flag mid-run contention.

Why the gate matters. The 14 July FP32 baseline measured 16.928 ms median total; re-run on a verified-idle GPU it measured 8.642 ms — a 49% correction. Both runs used identical timing methodology (RAM preload, warmup, single-thread, `cudnn.benchmark`), so this correction is entirely environmental: the machine changed, not the code. (Caveat: the box became idle and the NVIDIA driver was reinstalled the same afternoon, so idle-versus-driver cannot be separated from the available data.) The code contribution is a separate, far smaller effect — adding warmup, single-threading, and `cudnn.benchmark` to the earlier un-instrumented benchmark moved median total by only −0.92 ms, against the −8.29 ms environmental swing. That such contention is routine rather than exceptional was confirmed on 2026-07-17, when a re-run was refused by the gate: another user's 4-GPU DDP training run held ~47 GB and 100% util across all four A100s. A latency figure is only comparable when measured on a verified-idle GPU.

**An earlier run, archived for context.** A first exploratory benchmark measured 16.103 ms median total (62.2 FPS), with preprocess 2.637, inference 10.155 and postprocess 5.005. It is recorded here for provenance only and is **not** a controlled comparison against the figures above: it predates device pinning, the idle gate, warmup and single-threading, and it ran in a different and noisier machine state, so no single cause can be attributed to the difference. Its postprocess figure is the clearest measure of how noisy that environment was — single-class NMS over roughly two boxes costs a fraction of a millisecond, so 5.005 ms was never NMS cost. The controlled before/after is the 14 July to 16 July pair described above, not this run.

**What changed in the benchmark itself.** The current `scripts/benchmark_latency.py` differs from the earliest version in five ways, all of which exist to make a number attributable rather than to make it faster: `DEVICE` is strict and recorded, so a missing environment variable can no longer silently fall back to GPU 0; the GPU is gated via pynvml before CUDA initialises; utilisation, memory and process count are snapshotted around every repeat and written to the CSV, so a contended run is no longer indistinguishable from a clean one; median, p95 and p99 are reported alongside mean, std, min and max; and both the script and the subset generator are tracked in git rather than living only on Geneva's disk. Timing methodology is otherwise unchanged — same `result.speed[]` stage timings, same 100 RAM-preloaded frames, same warmup, same three stages.

Threading is **not** among the changes. `torch.set_num_threads(1)` and `torch.backends.cudnn.benchmark = True` are still set (`scripts/benchmark_latency.py:230` and `:235`), exactly as in the archived `benchmark_fp32_geneva_ram_stable.py`. The hypothesis that single-threading was throttling preprocess and postprocess was tested and rejected: the entire code-side contribution is −0.92 ms, an order of magnitude too small to account for the gap that mattered. That gap was environmental.

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
