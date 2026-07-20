import os
import sys
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone


# -----------------------------
# Settings
# -----------------------------
# Source ONNX -- the validated FP32 export from docs/02. We do not modify it;
# the build writes a sibling .engine named for its precision.
ONNX_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/best.onnx"
)

# Precision baked into the engine. This is a BUILD-time choice -- the engine
# cannot be re-precisioned afterwards. Each precision is a separate build/file.
#   fp32 -> the V2 baseline (TensorRT optimisation, no precision change)
#   fp16 -> half precision (BuilderFlag.FP16)
# INT8 is deliberately NOT handled here: it needs a calibration pass, which
# belongs in its own script (docs/04).
PRECISION = os.environ.get("PRECISION", "fp32").lower()

# DEVICE is REQUIRED and has no default -- same discipline as benchmark_latency.py.
# TensorRT times candidate kernels on the live GPU during the build to pick the
# fastest, so a contended GPU can bias kernel selection. No GPU specified, no build.
DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=2). No GPU specified, no build.")
# Pin the GPU BEFORE tensorrt/CUDA initialises. After this, CUDA device 0 is the
# one we selected.
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

# TF32 for the FP32 path. On Ampere+ TensorRT runs FP32 conv/matmul in TF32
# (10-bit mantissa) BY DEFAULT -- a mild precision reduction. For a clean FP32
# baseline that isolates TensorRT optimisation with zero precision change, we
# turn it OFF (true FP32, matching PyTorch/ONNX). Set ALLOW_TF32=1 to keep the
# faster default TF32 behaviour. Ignored for fp16.
ALLOW_TF32 = os.environ.get("ALLOW_TF32", "0") == "1"

# Builder workspace (scratch) memory ceiling, in GiB.
WORKSPACE_GB = float(os.environ.get("WORKSPACE_GB", 8))

ENGINE_PATH = ONNX_PATH.with_name(f"best_{PRECISION}.engine")

import tensorrt as trt  # noqa: E402 -- imported after CUDA_VISIBLE_DEVICES is set


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_name(index):
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", str(index)],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:
        return None


# -----------------------------
# Build
# -----------------------------
print("============================================================")
print(f"ONNX -> TensorRT engine  ({PRECISION.upper()})")
print("============================================================")
print("ONNX:      ", ONNX_PATH)
print("Engine:    ", ENGINE_PATH)
print("DEVICE:    ", DEVICE, f"({gpu_name(DEVICE)})")
print("TensorRT:  ", trt.__version__)
print("Workspace: ", WORKSPACE_GB, "GiB")

if not ONNX_PATH.exists():
    sys.exit(f"ONNX not found: {ONNX_PATH}")
if PRECISION not in ("fp32", "fp16"):
    sys.exit(f"PRECISION must be fp32 or fp16 (got {PRECISION!r}); INT8 has its own script.")

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)

print("\nParsing ONNX...")
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("  ", parser.get_error(i))
        sys.exit("ONNX parse failed.")
print(f"  parsed OK: {network.num_layers} layers")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(WORKSPACE_GB * (1 << 30)))

fp16_flag = False
tf32_enabled = True
if PRECISION == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)
    fp16_flag = True
elif PRECISION == "fp32" and not ALLOW_TF32:
    # Force true FP32: disable the default-on TF32 tensor-core path.
    config.clear_flag(trt.BuilderFlag.TF32)
    tf32_enabled = False
print(f"  precision={PRECISION}  fp16_flag={fp16_flag}  tf32_enabled={tf32_enabled}")

# The ONNX is static [1, 3, 640, 640], so no optimisation profile is needed.
print(f"\nBuilding {PRECISION.upper()} engine (kernel autotuning on GPU {DEVICE})...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    sys.exit("Engine build failed (build_serialized_network returned None).")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)
print("Saved engine:", ENGINE_PATH, f"({ENGINE_PATH.stat().st_size / 1e6:.1f} MB)")


# -----------------------------
# Sanity: deserialize + report bindings  (proves the engine file is valid)
# -----------------------------
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(ENGINE_PATH.read_bytes())
if engine is None:
    sys.exit("Deserialization failed -- engine file is not usable.")

io = []
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    io.append({
        "name": name,
        "mode": str(engine.get_tensor_mode(name)).split(".")[-1],   # INPUT / OUTPUT
        "shape": list(engine.get_tensor_shape(name)),
        "dtype": str(engine.get_tensor_dtype(name)).split(".")[-1],
    })
print("\nEngine I/O tensors:")
for t in io:
    print(f"  {t['mode']:<6} {t['name']:<10} {t['shape']}  {t['dtype']}")


# -----------------------------
# Provenance  (next to the .engine; mirrors the ONNX stage)
# -----------------------------
prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "host": platform.node(),
    "device_index": DEVICE,
    "gpu_name": gpu_name(DEVICE),
    "precision": PRECISION,
    "fp16_flag": fp16_flag,
    "tf32_enabled": tf32_enabled,
    "tensorrt_version": trt.__version__,
    "workspace_gib": WORKSPACE_GB,
    "num_layers": network.num_layers,
    "source_onnx": {
        "path": str(ONNX_PATH),
        "sha256": sha256(ONNX_PATH),
        "bytes": ONNX_PATH.stat().st_size,
    },
    "output_engine": {
        "path": str(ENGINE_PATH),
        "sha256": sha256(ENGINE_PATH),
        "bytes": ENGINE_PATH.stat().st_size,
    },
    "io_tensors": io,
}
prov_path = ENGINE_PATH.with_name(ENGINE_PATH.name + ".provenance.json")
prov_path.write_text(json.dumps(prov, indent=2))
print("\nProvenance:", prov_path)
print(f"\nDONE -- {PRECISION.upper()} engine built and deserialised cleanly on GPU {DEVICE}.")
print("NOTE: this is a build + validity check, not an accuracy or latency measurement.")
