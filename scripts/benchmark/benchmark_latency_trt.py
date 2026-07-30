from pathlib import Path
import csv
import gc
import hashlib
import json
import os
import socket
import sys
import time
from time import perf_counter
from datetime import datetime, timezone

import numpy as np
import cv2
import pynvml


# -----------------------------
# Settings
# -----------------------------
# DEVICE is strict: this benchmark refuses to guess a GPU (same rule as
# benchmark_latency.py -- a mis-attributed GPU is how a latency number becomes
# unusable).
if "DEVICE" not in os.environ:
    sys.exit(
        "DEVICE is not set. Refusing to guess a GPU.\n"
        "Set it explicitly, e.g.  DEVICE=2 python scripts/benchmark/benchmark_latency_trt.py"
    )

DEVICE = int(os.environ["DEVICE"])
SEED = int(os.environ.get("SEED", 42))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
CONF = float(os.environ.get("CONF", 0.25))
BENCHMARK_REPEATS = int(os.environ.get("BENCHMARK_REPEATS", 10))
WARMUP_IMAGES = int(os.environ.get("WARMUP_IMAGES", 30))

# Gate thresholds (identical semantics to benchmark_latency.py).
GATE_UTIL_THRESHOLD = int(os.environ.get("GATE_UTIL_THRESHOLD", 10))

# Per-process memory below which a foreign compute process is tolerated on the
# target GPU. Default 0 = the strict original rule: any other process refuses.
# Raise it only to get past DORMANT contexts (abandoned interpreters holding a
# few hundred MiB at 0% util) that would otherwise make a shared server
# permanently unbenchmarkable. A tolerated run is NOT an exclusive-GPU run and
# is recorded as such in the provenance sidecar.
GATE_ALLOW_IDLE_MIB = int(os.environ.get("GATE_ALLOW_IDLE_MIB", 0))
GATE_SAMPLES = int(os.environ.get("GATE_SAMPLES", 5))
GATE_INTERVAL_S = float(os.environ.get("GATE_INTERVAL_S", 0.2))

ENGINE_PATH = Path(os.environ.get(
    "ENGINE_PATH",
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/baseline/fp32/best_fp32.engine",
))

IMG_DIR = Path(
    "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/images/val"
)

OUT_DIR = Path("/home/zcemml1/medtronic_qat_data/runs_sanoscience")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output filenames are derived from the engine, never hardcoded. An earlier
# version baked "fp32" into these names, so benchmarking a different engine
# silently overwrote the FP32 results with FP16 numbers under an fp32 filename.
ENGINE_TAG = ENGINE_PATH.stem.replace("best_", "", 1)   # best_fp16.engine -> fp16


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

HOST = socket.gethostname()
OWN_PID = os.getpid()


# -----------------------------
# GPU gate (copied from benchmark_latency.py -- same discipline, no torch here)
# -----------------------------
def resolve_physical_index(dev):
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None or cvd.strip() == "":
        return dev
    visible = [x.strip() for x in cvd.split(",") if x.strip() != ""]
    if dev >= len(visible):
        sys.exit(f"DEVICE={dev} is out of range for CUDA_VISIBLE_DEVICES={cvd!r}.")
    return int(visible[dev])


def _compute_pids(handle):
    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    return [(p.pid, p.usedGpuMemory) for p in procs]


def _print_procs(procs):
    for pid, mem in procs:
        mib = "unknown" if mem is None else f"{mem / (1024 ** 2):.0f} MiB"
        print(f"  pid {pid}: {mib}")


def _blocking_procs(procs):
    """Compute processes that disqualify the GPU.

    Default GATE_ALLOW_IDLE_MIB=0 means ANY other process blocks -- the original,
    strictest rule. Raising it tolerates small DORMANT contexts: an abandoned
    interpreter holding a few hundred MiB at 0% utilisation consumes no SM time
    and no meaningful bandwidth, so it cannot distort a latency measurement,
    but under the strict rule it makes a shared server unbenchmarkable forever.

    This tolerance is deliberately memory-based only. Utilisation is still gated
    separately by GATE_UTIL_THRESHOLD, and every repeat records its own
    compute_procs_other / gpu_util_peak snapshot -- so if a tolerated process
    wakes up mid-run it shows up in the per-repeat table and the run can be
    discarded. Nothing is hidden by allowing it through the gate.
    """
    blocking = []
    for pid, mem in procs:
        # Unknown memory is treated as blocking -- never assume it is small.
        if mem is None or mem / (1024 ** 2) > GATE_ALLOW_IDLE_MIB:
            blocking.append((pid, mem))
    return blocking


def gate_gpu_or_exit(physical_index):
    pynvml.nvmlInit()
    try:
        count = pynvml.nvmlDeviceGetCount()
        if physical_index >= count:
            sys.exit(f"Physical GPU index {physical_index} does not exist "
                     f"(NVML sees {count} device(s)).")
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        print("------------------------------------------------------------")
        print("GPU gate")
        print("------------------------------------------------------------")
        print(f"Target physical GPU {physical_index}: {name}")

        present = _compute_pids(handle)
        busy = _blocking_procs(present)
        if busy:
            print("Refusing to run. Compute processes already on this GPU:")
            _print_procs(busy)
            if GATE_ALLOW_IDLE_MIB:
                print(f"(tolerance GATE_ALLOW_IDLE_MIB={GATE_ALLOW_IDLE_MIB} MiB "
                      f"was not enough to clear these)")
            sys.exit(1)
        tolerated = present            # everything here passed the size test

        util_samples = []
        for i in range(GATE_SAMPLES):
            util_samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
            busy = _blocking_procs(_compute_pids(handle))
            if busy:
                print("Refusing to run. A compute process appeared during the gate:")
                _print_procs(busy)
                sys.exit(1)
            if i < GATE_SAMPLES - 1:
                time.sleep(GATE_INTERVAL_S)

        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        peak_util = max(util_samples)
        if tolerated:
            print(f"Compute processes: {len(tolerated)} tolerated "
                  f"(each <= GATE_ALLOW_IDLE_MIB={GATE_ALLOW_IDLE_MIB} MiB, dormant)")
            _print_procs(tolerated)
            print("NOTE: this run did NOT have the GPU exclusively. Per-repeat "
                  "contention snapshots below are the audit trail.")
        else:
            print("Compute processes: none")
        print(f"Utilisation samples (%): {util_samples}  peak={peak_util}")
        print(f"Memory used: {mem.used / (1024 ** 2):.0f} MiB "
              f"of {mem.total / (1024 ** 2):.0f} MiB")
        if peak_util > GATE_UTIL_THRESHOLD:
            sys.exit(f"Refusing to run. Peak utilisation {peak_util}% exceeds "
                     f"GATE_UTIL_THRESHOLD={GATE_UTIL_THRESHOLD}%.")
        print("Gate passed: GPU is idle"
              f"{' (exclusive)' if not tolerated else ' apart from tolerated dormant contexts'}.\n")
        return {"tolerated_procs": [{"pid": p, "mib": None if m is None
                                     else int(round(m / (1024 ** 2)))}
                                    for p, m in tolerated],
                "gate_allow_idle_mib": GATE_ALLOW_IDLE_MIB,
                "gate_util_peak": peak_util}
    finally:
        pynvml.nvmlShutdown()


def gpu_snapshot(physical_index, own_pid):
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        others = [p.pid for p in procs if p.pid != own_pid]
        return {"gpu_util_pct": int(util.gpu),
                "mem_used_mib": int(round(mem.used / (1024 ** 2))),
                "compute_procs_other": len(others)}
    finally:
        pynvml.nvmlShutdown()


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "median_ms": float(np.median(arr)),
        "min_ms": float(arr.min()),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "max_ms": float(arr.max()),
    }


# -----------------------------
# Gate BEFORE creating any CUDA context
# -----------------------------
PHYSICAL_INDEX = resolve_physical_index(DEVICE)
gate_info = gate_gpu_or_exit(PHYSICAL_INDEX)


# -----------------------------
# CUDA / TensorRT setup (after the gate passes)
# -----------------------------
import tensorrt as trt                          # noqa: E402
from cuda.bindings import runtime as cudart     # noqa: E402

H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


def CHK(ret):
    err, rest = (ret[0], ret[1:]) if isinstance(ret, (tuple, list)) else (ret, ())
    if int(err) != 0:
        raise RuntimeError(f"CUDA error code {int(err)}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


# Pin the target GPU for this thread before any allocation / engine context.
CHK(cudart.cudaSetDevice(DEVICE))


class TRTEngine:
    def __init__(self, path, logger):
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {path}")
        self.context = self.engine.create_execution_context()
        self.inp_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inp_name, self.inp_shape = n, tuple(self.engine.get_tensor_shape(n))
            else:
                self.out_name, self.out_shape = n, tuple(self.engine.get_tensor_shape(n))
        self.out_host = np.empty(self.out_shape, dtype=np.float32)
        self.d_in = CHK(cudart.cudaMalloc(int(np.prod(self.inp_shape)) * 4))
        self.d_out = CHK(cudart.cudaMalloc(int(np.prod(self.out_shape)) * 4))
        self.stream = CHK(cudart.cudaStreamCreate())
        self.context.set_tensor_address(self.inp_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))
        # CUDA events for GPU-side phase timing, all on the same stream. Each
        # phase is bracketed by its own pair of events, so H2D, compute, and D2H
        # are measured separately and in GPU time -- the same method (and the
        # same three quantities) trtexec reports. Event timing is immune to CPU
        # scheduling jitter, unlike wall-clock around cudaStreamSynchronize.
        self.ev = {k: (CHK(cudart.cudaEventCreate()), CHK(cudart.cudaEventCreate()))
                   for k in ("h2d", "compute", "d2h")}

    def infer(self, im):
        """Returns (output, {h2d_ms, kernel_ms, d2h_ms}). All three are GPU-time
        via CUDA events (trtexec-style). GPU latency = their sum."""
        im = np.ascontiguousarray(im, dtype=np.float32)
        (h2d_s, h2d_e), (c_s, c_e), (d2h_s, d2h_e) = (
            self.ev["h2d"], self.ev["compute"], self.ev["d2h"])
        CHK(cudart.cudaEventRecord(h2d_s, self.stream))
        CHK(cudart.cudaMemcpyAsync(self.d_in, im.ctypes.data, im.nbytes, H2D, self.stream))
        CHK(cudart.cudaEventRecord(h2d_e, self.stream))
        # compute: nothing but execute_async_v3 sits between these two records
        CHK(cudart.cudaEventRecord(c_s, self.stream))
        self.context.execute_async_v3(int(self.stream))
        CHK(cudart.cudaEventRecord(c_e, self.stream))
        CHK(cudart.cudaEventRecord(d2h_s, self.stream))
        CHK(cudart.cudaMemcpyAsync(self.out_host.ctypes.data, self.d_out,
                                   self.out_host.nbytes, D2H, self.stream))
        CHK(cudart.cudaEventRecord(d2h_e, self.stream))
        CHK(cudart.cudaStreamSynchronize(self.stream))
        return self.out_host, {
            "h2d_ms": CHK(cudart.cudaEventElapsedTime(h2d_s, h2d_e)),
            "kernel_ms": CHK(cudart.cudaEventElapsedTime(c_s, c_e)),
            "d2h_ms": CHK(cudart.cudaEventElapsedTime(d2h_s, d2h_e)),
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
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return im[None]


def postprocess(raw, conf):
    d = raw[0]
    return d[d[:, 4] >= conf]


# -----------------------------
# Report header
# -----------------------------
print("============================================================")
print("TensorRT FP32 engine latency benchmark (pooled percentiles)")
print("============================================================")
print("Host:", HOST)
print("Engine:", ENGINE_PATH)
print("Physical NVML index:", PHYSICAL_INDEX)
print("Image size:", IMG_SIZE, " Confidence:", CONF)
print("Repeats:", BENCHMARK_REPEATS, " Warmup images:", WARMUP_IMAGES)
print("TensorRT:", trt.__version__)

if not ENGINE_PATH.exists():
    sys.exit(f"Engine not found: {ENGINE_PATH}")

# -----------------------------
# Preload images into RAM (disk I/O outside the timed loop)
# -----------------------------
image_paths = sorted(IMG_DIR.glob("*.jpg"))
if not image_paths:
    raise ValueError(f"No images found in {IMG_DIR}")

print("\nPreloading images into RAM...")
ram_images = []
for path in image_paths:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    ram_images.append((path.name, img))
print(f"Loaded {len(ram_images)} images into RAM.")

# -----------------------------
# Load engine once
# -----------------------------
print("\nLoading engine once...")
logger = trt.Logger(trt.Logger.WARNING)
eng = TRTEngine(ENGINE_PATH, logger)
gc.collect()
print("Engine loaded.")

# -----------------------------
# Warmup (stabilise GPU clocks; TRT kernels are already fixed from build)
# -----------------------------
print("\nWarmup...")
for _, img in ram_images[: min(WARMUP_IMAGES, len(ram_images))]:
    _ = eng.infer(preprocess(img, IMG_SIZE))
print("Warmup complete.")

# -----------------------------
# Benchmark repeats -- pool every per-image reading across all runs
# -----------------------------
pool = {"preprocess_ms": [], "inference_ms": [], "postprocess_ms": [],
        "h2d_ms": [], "kernel_ms": [], "d2h_ms": [], "gpu_latency_ms": [],
        "total_ms": []}
per_repeat = []
snapshot_at = {len(ram_images) // 4, len(ram_images) // 2, (3 * len(ram_images)) // 4}

for run_idx in range(1, BENCHMARK_REPEATS + 1):
    print(f"\nRun {run_idx}/{BENCHMARK_REPEATS}")
    gc.collect()
    rows, snaps = [], []

    for i, (image_name, img) in enumerate(ram_images):
        t0 = perf_counter()
        im = preprocess(img, IMG_SIZE)          # CPU: letterbox + normalise
        t1 = perf_counter()
        raw, gpu = eng.infer(im)                 # GPU phases event-timed inside
        t2 = perf_counter()
        dets = postprocess(raw, CONF)            # CPU: confidence filter (NMS is in-engine)
        t3 = perf_counter()

        # A = total (pipeline, wall-clock). B = inference (memcpy-incl, wall-clock,
        # t1->t2). GPU phases (h2d/kernel/d2h) are CUDA-event GPU time; their sum
        # is the trtexec-style GPU latency, all immune to CPU scheduling jitter.
        pre, inf, post = (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3
        total = pre + inf + post
        gpu_lat = gpu["h2d_ms"] + gpu["kernel_ms"] + gpu["d2h_ms"]

        rows.append({"run": run_idx, "image": image_name,
                     "preprocess_ms": pre, "inference_ms": inf,
                     "postprocess_ms": post, "h2d_ms": gpu["h2d_ms"],
                     "kernel_ms": gpu["kernel_ms"], "d2h_ms": gpu["d2h_ms"],
                     "gpu_latency_ms": gpu_lat, "total_ms": total,
                     "num_boxes": int(len(dets))})
        pool["preprocess_ms"].append(pre)
        pool["inference_ms"].append(inf)
        pool["postprocess_ms"].append(post)
        pool["h2d_ms"].append(gpu["h2d_ms"])
        pool["kernel_ms"].append(gpu["kernel_ms"])
        pool["d2h_ms"].append(gpu["d2h_ms"])
        pool["gpu_latency_ms"].append(gpu_lat)
        pool["total_ms"].append(total)

        if i in snapshot_at:
            snaps.append(gpu_snapshot(PHYSICAL_INDEX, OWN_PID))

    run_total = np.asarray([r["total_ms"] for r in rows], dtype=float)
    util_peak = max((s["gpu_util_pct"] for s in snaps), default=-1)
    mem_peak = max((s["mem_used_mib"] for s in snaps), default=-1)
    others_peak = max((s["compute_procs_other"] for s in snaps), default=0)

    per_repeat.append({"run": run_idx,
                       "mean_total_ms": float(run_total.mean()),
                       "min_total_ms": float(run_total.min()),
                       "max_total_ms": float(run_total.max()),
                       "gpu_util_peak_pct": util_peak,
                       "mem_used_peak_mib": mem_peak,
                       "compute_procs_other_peak": others_peak})

    flag = "" if others_peak == 0 else f"  <-- CONTENDED ({others_peak} other proc)"
    print(f"  images: {len(rows)}   mean total: {run_total.mean():.3f} ms   "
          f"min: {run_total.min():.3f}   max: {run_total.max():.3f}")
    print(f"  gpu snapshot: util_peak={util_peak}%  mem_peak={mem_peak} MiB  "
          f"other_compute_procs={others_peak}{flag}")

    out_csv = OUT_DIR / f"benchmark_latency_trt_{ENGINE_TAG}_seed{SEED}_run{run_idx}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  saved per-image CSV: {out_csv}")

# -----------------------------
# Pooled summary
# -----------------------------
n_pooled = len(pool["total_ms"])
# GPU phases (H2D / Kernel / D2H, all CUDA-event) grouped, then the wall-clock
# CPU stages and totals.
stage_order = [("preprocess_ms", "Preprocess"), ("inference_ms", "Inference"),
               ("h2d_ms", "H2D(GPU)"), ("kernel_ms", "Kernel(GPU)"),
               ("d2h_ms", "D2H(GPU)"), ("gpu_latency_ms", "GPUlatency"),
               ("postprocess_ms", "Postprocess"), ("total_ms", "Total")]
summary_rows = []
for key, label in stage_order:
    s = {"stage": label, "n": n_pooled}
    s.update(summarize(pool[key]))
    summary_rows.append(s)

total_median = next(r["median_ms"] for r in summary_rows if r["stage"] == "Total")
kernel_median = next(r["median_ms"] for r in summary_rows if r["stage"] == "Kernel(GPU)")
fps_median = 1000.0 / total_median
fps_kernel = 1000.0 / kernel_median
noncompute_gap = total_median - kernel_median

summary_csv = OUT_DIR / f"benchmark_latency_trt_{ENGINE_TAG}_seed{SEED}_pooled_summary.csv"
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

# Sidecar tying these numbers to the exact engine that produced them -- same
# reasoning as the sha256 in the parity record: a latency figure with no engine
# hash cannot be re-attached to its artifact once the engine is rebuilt.
meta_path = summary_csv.with_suffix(".provenance.json")
meta_path.write_text(json.dumps({
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "host": HOST,
    "device": DEVICE,
    "engine": str(ENGINE_PATH),
    "engine_sha256": sha256(ENGINE_PATH),
    "engine_bytes": ENGINE_PATH.stat().st_size,
    "seed": SEED,
    "repeats": BENCHMARK_REPEATS,
    "images_per_run": len(ram_images),
    "pooled_samples": n_pooled,
    "warmup_images": WARMUP_IMAGES,
    "conf": CONF,
    "img_size": IMG_SIZE,
    # Was the GPU exclusively ours? A run with tolerated dormant contexts is
    # still a valid measurement but is NOT the same condition as an exclusive
    # run, and must not be silently compared against one.
    "gate": gate_info,
    "exclusive_gpu": not gate_info["tolerated_procs"],
    "max_compute_procs_other": max(r["compute_procs_other_peak"] for r in per_repeat),
    "max_gpu_util_peak_pct": max(r["gpu_util_peak_pct"] for r in per_repeat),
    "summary": {r["stage"]: r for r in summary_rows},
    "fps_median_total": fps_median,
    "fps_median_kernel": fps_kernel,
    "noncompute_gap_ms": noncompute_gap,   # total - kernel = CPU preprocess + memcpy
}, indent=2))
print("Provenance:", meta_path)

print("\n============================================================")
print(f"Pooled latency summary (TensorRT {ENGINE_TAG.upper()} engine)")
print("============================================================")
print(f"Runs: {BENCHMARK_REPEATS}   Images/run: {len(ram_images)}   Pooled samples: {n_pooled}")
print("CSV summary:", summary_csv)
print()
header = (f"{'Stage':<12} {'Mean':>9} {'Std':>9} {'Median':>9} "
         f"{'Min':>9} {'P95':>9} {'P99':>9} {'Max':>9}")
print(header)
print("-" * len(header))
for row in summary_rows:
    print(f"{row['stage']:<12} {row['mean_ms']:>9.3f} {row['std_ms']:>9.3f} "
          f"{row['median_ms']:>9.3f} {row['min_ms']:>9.3f} {row['p95_ms']:>9.3f} "
          f"{row['p99_ms']:>9.3f} {row['max_ms']:>9.3f}")
print("-" * len(header))
print(f"Approx FPS (from pooled Total median): {fps_median:.2f}")

# -----------------------------
# Per-repeat contention table
# -----------------------------
repeats_csv = OUT_DIR / f"benchmark_latency_trt_{ENGINE_TAG}_seed{SEED}_per_repeat.csv"
with open(repeats_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=per_repeat[0].keys())
    writer.writeheader()
    writer.writerows(per_repeat)

print("\n------------------------------------------------------------")
print("Per-repeat GPU snapshot (contention check)")
print("------------------------------------------------------------")
contended = [r["run"] for r in per_repeat if r["compute_procs_other_peak"] > 0]
if contended:
    print(f"WARNING: contention detected on repeat(s): {contended}")
else:
    print("No other compute processes seen on any repeat: all clean.")
print("Done.")
