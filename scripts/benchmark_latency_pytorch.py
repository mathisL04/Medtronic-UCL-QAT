import os
import sys
import json
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter

import numpy as np


# -----------------------------
# Settings
# -----------------------------
# Raw-PyTorch latency study (NO TensorRT, no export). Measures the eager models
# themselves, two ways, to contrast the pre-conversion PyTorch cost against the
# TensorRT engines measured elsewhere.
#
#   MODEL_MODE = fp32 | fp16 | qat
#     fp32 : the V1 model (best.pt), FP32
#     fp16 : best.pt with model.half()  -- built here; see FP16 cleanliness note
#     qat  : the QAT FAKE-QUANT model (Q/DQ simulated in FP32). NOT real INT8.
#
# Column A -- pure-GPU compute: CUDA events around the forward pass only
#            (trtexec-method), val100, batch=1, pooled repeats, idle-gated, median.
# Column B -- Ultralytics pipeline: model.predict() speed counters
#            (preprocess + inference + postprocess), same val100.
#
# HONESTY: eager-mode carries PyTorch per-op launch overhead -- NOT clean engine
# numbers. The qat row is a fake-quant SIMULATION, not 8-bit. FP16 cleanliness
# (pure half vs autocast/FP32 islands) is detected and reported.
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
MODEL_PATH = REPO / "models/yolo26n_sanoscience_full_left/best.pt"
QAT_STATE = Path(os.environ.get(
    "QAT_STATE", str(REPO / "runs_qat/qat_v5/qat_modelopt_state_best.pt")))
IMG_DIR = Path(os.environ.get(
    "IMG_DIR", "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/images/val"))
OUT_DIR = Path("/home/zcemml1/medtronic_qat_data/runs_sanoscience")

MODEL_MODE = os.environ.get("MODEL_MODE", "").lower()
if MODEL_MODE not in ("fp32", "fp16", "qat"):
    sys.exit("MODEL_MODE must be fp32 | fp16 | qat")

SEED = int(os.environ.get("SEED", 42))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
BENCHMARK_REPEATS = int(os.environ.get("BENCHMARK_REPEATS", 10))
WARMUP_IMAGES = int(os.environ.get("WARMUP_IMAGES", 50))
GATE_ALLOW_IDLE_MIB = int(os.environ.get("GATE_ALLOW_IDLE_MIB", 0))

DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=2). No GPU specified, no run.")

# Idle-GPU gate (physical index, before CUDA_VISIBLE_DEVICES remaps it)
import pynvml  # noqa: E402
pynvml.nvmlInit()
_procs = pynvml.nvmlDeviceGetComputeRunningProcesses(
    pynvml.nvmlDeviceGetHandleByIndex(int(DEVICE)))
_foreign = [(p.pid, (p.usedGpuMemory or 0) // (1 << 20)) for p in _procs]
_blocking = [pp for pp in _foreign if pp[1] > GATE_ALLOW_IDLE_MIB]
exclusive = len(_foreign) == 0
if _blocking:
    sys.exit(f"Gate FAILED: other compute processes on GPU {DEVICE}: {_blocking}")
print(f"Gate passed (device {DEVICE}, foreign {_foreign}, exclusive={exclusive})")

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

import cv2                                                # noqa: E402
import torch                                              # noqa: E402
from ultralytics import YOLO                              # noqa: E402


def summarize(v):
    a = np.asarray(v, dtype=float)
    return {"n": int(a.size), "mean_ms": float(a.mean()),
            "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "median_ms": float(np.median(a)), "min_ms": float(a.min()),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99)),
            "max_ms": float(a.max())}


def letterbox(img, s):
    h0, w0 = img.shape[:2]
    r = min(s / h0, s / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    canvas = np.full((s, s, 3), 114, dtype=np.uint8)
    top, left = (s - nh) // 2, (s - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def preprocess(img, s):
    im = letterbox(img, s)[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0


print("=" * 60)
print(f"PyTorch latency study  --  MODE={MODEL_MODE}  (raw eager, no conversion)")
print("=" * 60)

# -----------------------------
# Build the model for column A + detect FP16 cleanliness
# -----------------------------
def load_base():
    y = YOLO(str(MODEL_PATH))
    if MODEL_MODE == "qat":
        import modelopt.torch.opt as mto
        mto.restore(y.model, str(QAT_STATE))
    return y


yA = load_base()
model = yA.model.eval().cuda()
for p in model.parameters():
    p.requires_grad = False

half_input = False
fp16_mode = None
use_autocast = False
if MODEL_MODE == "fp16":
    test = torch.from_numpy(preprocess(np.zeros((640, 640, 3), np.uint8), IMG_SIZE))
    try:
        model = model.half()
        with torch.no_grad():
            model(test.half().cuda())
        fp16_mode, half_input = "pure model.half()", True
        print("[fp16] pure model.half() forward OK -> clean FP16 (no manual FP32 islands)")
    except Exception as e:
        model = yA.model.eval().float().cuda()
        fp16_mode, use_autocast = "autocast (FP32 islands)", True
        print(f"[fp16] model.half() forward FAILED ({type(e).__name__}: {str(e)[:120]})")
        print("[fp16] falling back to torch.autocast(fp16) -> NOT pure FP16 (FP32 islands)")

print("torch", torch.__version__, "| cuda", torch.cuda.get_device_name(0))


def forward(x):
    with torch.no_grad():
        if use_autocast:
            with torch.autocast("cuda", dtype=torch.float16):
                return model(x)
        return model(x)


# -----------------------------
# Preload val100
# -----------------------------
paths = sorted(IMG_DIR.glob("*.jpg"))
if not paths:
    sys.exit(f"No images in {IMG_DIR}")
ram = [(p.name, cv2.imread(str(p))) for p in paths]
print(f"preloaded {len(ram)} images")

# -----------------------------
# Column A -- pure-GPU CUDA-event forward
# -----------------------------
for _, img in ram[:min(WARMUP_IMAGES, len(ram))]:
    x = torch.from_numpy(preprocess(img, IMG_SIZE)).cuda()
    forward(x.half() if half_input else x)
torch.cuda.synchronize()

ev_s, ev_e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
fwd_ms = []
for run in range(BENCHMARK_REPEATS):
    for _, img in ram:
        x = torch.from_numpy(preprocess(img, IMG_SIZE)).cuda()
        if half_input:
            x = x.half()
        torch.cuda.synchronize()
        ev_s.record()
        forward(x)
        ev_e.record()
        torch.cuda.synchronize()
        fwd_ms.append(ev_s.elapsed_time(ev_e))
A = summarize(fwd_ms)
print(f"\n[A] Forward(GPU) CUDA-event: median {A['median_ms']:.3f} ms "
      f"(mean {A['mean_ms']:.3f}) -> {1000.0/A['median_ms']:.1f} FPS")

# -----------------------------
# Column B -- Ultralytics predict speed counters (pre+inf+post)
# -----------------------------
yB = load_base()
half_flag = (MODEL_MODE == "fp16")
# warmup predict
yB.predict(ram[0][1], device=0, half=half_flag, verbose=False, imgsz=IMG_SIZE)
pre, inf, post = [], [], []
for run in range(BENCHMARK_REPEATS):
    for _, img in ram:
        r = yB.predict(img, device=0, half=half_flag, verbose=False, imgsz=IMG_SIZE)
        sp = r[0].speed
        pre.append(sp["preprocess"]); inf.append(sp["inference"]); post.append(sp["postprocess"])
tot = [a + b + c for a, b, c in zip(pre, inf, post)]
Bpre, Binf, Bpost, Btot = map(summarize, (pre, inf, post, tot))
print(f"[B] Ultralytics pipeline: pre {Bpre['median_ms']:.3f} + inf {Binf['median_ms']:.3f} "
      f"+ post {Bpost['median_ms']:.3f} = total {Btot['median_ms']:.3f} ms "
      f"(half={half_flag}) -> {1000.0/Btot['median_ms']:.1f} FPS")

# -----------------------------
# Record
# -----------------------------
prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "pytorch_raw_latency", "model_mode": MODEL_MODE,
    "note": ("eager PyTorch, no conversion. qat=fake-quant sim (not real INT8). "
             "fp16 cleanliness: " + str(fp16_mode)),
    "fp16_mode": fp16_mode, "device": DEVICE, "exclusive_gpu": exclusive,
    "img_dir": str(IMG_DIR), "n_images": len(ram), "repeats": BENCHMARK_REPEATS,
    "img_size": IMG_SIZE, "seed": SEED, "torch": torch.__version__,
    "A_forward_gpu": A,
    "B_pipeline": {"preprocess": Bpre, "inference": Binf, "postprocess": Bpost, "total": Btot,
                   "half": half_flag},
    "fps_A_forward": 1000.0 / A["median_ms"], "fps_B_total": 1000.0 / Btot["median_ms"],
}
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / f"benchmark_latency_pytorch_{MODEL_MODE}_seed{SEED}_summary.provenance.json"
out.write_text(json.dumps(prov, indent=2))
print("Provenance:", out)
