import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

import numpy as np


# -----------------------------
# Settings
# -----------------------------
# PyTorch-side latency of the QAT (fake-quant) model, as a counterpart to the
# TensorRT-engine latency in benchmark_latency_trt.py. Kept deliberately parallel
# to that script -- same val100 images, batch=1, RAM preload, warmup, pooled
# repeats, per-frame CUDA-event GPU timing + wall-clock -- so the two are
# comparable.
#
# IMPORTANT CAVEAT. This times the FAKE-QUANT forward: modelopt's TensorQuantizer
# simulates INT8 by quantise->dequantise in FP32, so this is NOT real INT8 and NOT
# a deployment number. It is expected to be SLOWER than plain FP32 PyTorch (extra
# Q/DQ ops) and much slower than the real INT8 TensorRT engine. The deployable
# latency is the TensorRT engine; this exists for completeness of the PyTorch page.
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
MODEL_PATH = REPO / "models/yolo26n_sanoscience_full_left/best.pt"
QAT_STATE = Path(os.environ.get(
    "QAT_STATE", str(REPO / "runs_qat/qat_v5/qat_modelopt_state_best.pt")))
IMG_DIR = Path(os.environ.get(
    "IMG_DIR", "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/images/val"))
OUT_DIR = Path("/home/zcemml1/medtronic_qat_data/runs_sanoscience")

SEED = int(os.environ.get("SEED", 42))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
BENCHMARK_REPEATS = int(os.environ.get("BENCHMARK_REPEATS", 10))
WARMUP_IMAGES = int(os.environ.get("WARMUP_IMAGES", 50))
GATE_ALLOW_IDLE_MIB = int(os.environ.get("GATE_ALLOW_IDLE_MIB", 0))

# DEVICE has no default -- same strict discipline as benchmark_latency_trt.py.
DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=2). No GPU specified, no run.")

# -----------------------------
# Idle-GPU gate (physical index, before CUDA_VISIBLE_DEVICES remaps it)
# -----------------------------
import pynvml  # noqa: E402
pynvml.nvmlInit()
_h = pynvml.nvmlDeviceGetHandleByIndex(int(DEVICE))
_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(_h)
_foreign = [(p.pid, (p.usedGpuMemory or 0) // (1 << 20)) for p in _procs]
_blocking = [pp for pp in _foreign if pp[1] > GATE_ALLOW_IDLE_MIB]
exclusive = len(_foreign) == 0
print(f"GPU gate: device {DEVICE}, foreign compute procs {_foreign}, "
      f"blocking {_blocking}")
if _blocking:
    sys.exit(f"Gate FAILED: other compute processes on GPU {DEVICE}: {_blocking}")
print("Gate passed." + ("" if exclusive else " (non-exclusive: dormant contexts tolerated)"))

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

import cv2                                                # noqa: E402
import torch                                              # noqa: E402
from ultralytics import YOLO                              # noqa: E402
import modelopt.torch.opt as mto                          # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


def letterbox(img, new_shape):
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(img_bgr, imgsz):
    lb = letterbox(img_bgr, imgsz)
    im = lb[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0


# -----------------------------
# Restore the QAT fake-quant model
# -----------------------------
print("=" * 60)
print("PyTorch QAT (fake-quant) latency  --  NOT a deployment number")
print("=" * 60)
print("model:", MODEL_PATH, "\nstate:", QAT_STATE, "\nimages:", IMG_DIR)
torch.manual_seed(SEED)
y = YOLO(str(MODEL_PATH))
mto.restore(y.model, str(QAT_STATE))
model = y.model.eval().cuda()
for p in model.parameters():
    p.requires_grad = False
print("torch", torch.__version__, "| cuda", torch.cuda.get_device_name(0))

# -----------------------------
# Preload val100 into RAM
# -----------------------------
image_paths = sorted(IMG_DIR.glob("*.jpg"))
if not image_paths:
    sys.exit(f"No images found in {IMG_DIR}")
ram = [(p.name, cv2.imread(str(p))) for p in image_paths]
print(f"preloaded {len(ram)} images")

# -----------------------------
# Warmup
# -----------------------------
with torch.no_grad():
    for _, img in ram[:min(WARMUP_IMAGES, len(ram))]:
        x = torch.from_numpy(preprocess(img, IMG_SIZE)).cuda()
        model(x)
torch.cuda.synchronize()
print("warmup complete")

# -----------------------------
# Timed repeats -- per-frame CUDA-event forward + wall-clock total
# -----------------------------
pool = {"preprocess_ms": [], "forward_gpu_ms": [], "total_ms": []}
ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
for run in range(1, BENCHMARK_REPEATS + 1):
    for name, img in ram:
        t0 = perf_counter()
        arr = preprocess(img, IMG_SIZE)                  # CPU
        t1 = perf_counter()
        x = torch.from_numpy(arr).cuda()                 # H2D
        torch.cuda.synchronize()
        ev_s.record()
        with torch.no_grad():
            model(x)                                     # fake-quant forward (GPU)
        ev_e.record()
        torch.cuda.synchronize()
        t2 = perf_counter()
        pool["preprocess_ms"].append((t1 - t0) * 1e3)
        pool["forward_gpu_ms"].append(ev_s.elapsed_time(ev_e))
        pool["total_ms"].append((t2 - t0) * 1e3)
    print(f"  run {run}/{BENCHMARK_REPEATS}: "
          f"forward(GPU) median so far "
          f"{np.median(pool['forward_gpu_ms']):.3f} ms")

# -----------------------------
# Report
# -----------------------------
order = [("preprocess_ms", "Preprocess(CPU)"), ("forward_gpu_ms", "Forward(GPU)"),
         ("total_ms", "Total")]
print(f"\n{'Stage':16}{'Mean':>9}{'Std':>9}{'Median':>9}{'Min':>9}{'P95':>9}{'P99':>9}")
rows = {}
for key, label in order:
    s = summarize(pool[key])
    rows[label] = s
    print(f"{label:16}{s['mean_ms']:9.3f}{s['std_ms']:9.3f}{s['median_ms']:9.3f}"
          f"{s['min_ms']:9.3f}{s['p95_ms']:9.3f}{s['p99_ms']:9.3f}")
fwd_med = rows["Forward(GPU)"]["median_ms"]
tot_med = rows["Total"]["median_ms"]
print(f"\nForward(GPU) median {fwd_med:.3f} ms  ->  {1000.0/fwd_med:.1f} FPS")
print(f"Total median        {tot_med:.3f} ms  ->  {1000.0/tot_med:.1f} FPS")

OUT_DIR.mkdir(parents=True, exist_ok=True)
prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "pytorch_qat_fakequant_latency",
    "note": "fake-quant (Q/DQ simulated in FP32) forward; NOT real INT8, NOT deployment",
    "device": DEVICE, "exclusive_gpu": exclusive,
    "model": str(MODEL_PATH), "qat_state": str(QAT_STATE),
    "qat_state_sha256": sha256(QAT_STATE),
    "img_dir": str(IMG_DIR), "n_images": len(ram),
    "repeats": BENCHMARK_REPEATS, "pooled_samples": len(pool["total_ms"]),
    "img_size": IMG_SIZE, "seed": SEED,
    "torch": torch.__version__,
    "summary": {label: rows[label] for _, label in order},
    "fps_median_forward": 1000.0 / fwd_med,
    "fps_median_total": 1000.0 / tot_med,
}
out = OUT_DIR / f"benchmark_latency_pytorch_qat_seed{SEED}_pooled_summary.provenance.json"
out.write_text(json.dumps(prov, indent=2))
print("Provenance:", out)
