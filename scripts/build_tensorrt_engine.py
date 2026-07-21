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
#   fp16 -> V3, half precision (BuilderFlag.FP16; mixed precision)
#   int8 -> V4, PTQ. Needs a calibration pass over real frames, so unlike the
#           other two it also reads CALIB_DIR and writes a calibration cache.
PRECISION = os.environ.get("PRECISION", "fp32").lower()

# INT8 only. Calibration frames must be DISJOINT from the evaluation data --
# see scripts/make_calib_set.py, which enforces that at episode level and refuses
# to build an overlapping set. Calibrating on eval frames biases accuracy upward.
CALIB_DIR = Path(os.environ.get(
    "CALIB_DIR", "/home/zcemml1/medtronic_qat_data/calib_int8_train_yolo"))
CALIBRATOR = os.environ.get("CALIBRATOR", "entropy").lower()   # entropy | minmax

# INT8 alone leaves non-INT8 layers in FP32. Setting INT8_ALLOW_FP16=1 also sets
# the FP16 flag so they fall back to FP16 instead -- the usual deployment config,
# but it makes the engine differ from V2 by TWO precision changes rather than
# one. Default off, so V4 stays a clean single-variable ablation against V2.
INT8_ALLOW_FP16 = os.environ.get("INT8_ALLOW_FP16", "0") == "1"

# DEVICE is REQUIRED and has no default -- same discipline as benchmark_latency.py.
# TensorRT times candidate kernels on the live GPU during the build to pick the
# fastest, so a contended GPU can bias kernel selection. No GPU specified, no build.
DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=2). No GPU specified, no build.")
# Pin the GPU BEFORE tensorrt/CUDA initialises. After this, CUDA device 0 is the
# one we selected.
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

# TF32. On Ampere+ TensorRT runs FP32 conv/matmul in TF32 (10-bit mantissa) BY
# DEFAULT -- a mild precision reduction. We turn it OFF for EVERY precision, not
# just fp32: TF32 governs how layers that run in FP32 are computed, which in a
# mixed-precision FP16 engine includes the FP32 fallback layers. Leaving it at
# its default for fp16 would mean an FP32-vs-FP16 comparison measured two
# changes at once. Set ALLOW_TF32=1 for the faster default behaviour -- that is
# a different precision and a separate build, not the baseline.
ALLOW_TF32 = os.environ.get("ALLOW_TF32", "0") == "1"

# Builder workspace (scratch) memory ceiling, in GiB.
WORKSPACE_GB = float(os.environ.get("WORKSPACE_GB", 8))

# Calibration input size. Must match the ONNX's static input [1,3,640,640] --
# the calibration batch shape has to match the network input shape exactly.
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))

ENGINE_PATH = ONNX_PATH.with_name(f"best_{PRECISION}.engine")

sys.path.insert(0, str(Path(__file__).resolve().parent))   # for int8_calibrator

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
if PRECISION not in ("fp32", "fp16", "int8"):
    sys.exit(f"PRECISION must be fp32, fp16 or int8 (got {PRECISION!r}).")
if PRECISION == "int8":
    if CALIBRATOR not in ("entropy", "minmax"):
        sys.exit(f"CALIBRATOR must be entropy or minmax (got {CALIBRATOR!r}).")
    calib_images = sorted((CALIB_DIR / "images").glob("*.jpg"))
    if not calib_images:
        sys.exit(
            f"No calibration frames in {CALIB_DIR / 'images'}.\n"
            "Build the set first:  python scripts/make_calib_set.py"
        )
    calib_manifest = next(iter(sorted(CALIB_DIR.glob("calib*_manifest.txt"))), None)
    if calib_manifest is None:
        sys.exit(f"No calibration manifest in {CALIB_DIR} -- refusing to build "
                 "an INT8 engine whose calibration data cannot be identified.")
    print("Calib dir: ", CALIB_DIR, f"({len(calib_images)} frames, {CALIBRATOR})")

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

# Precision is a builder flag (TensorRT 10.x). FP16 lets TensorRT run layers in
# half precision where it is faster, keeping numerically sensitive layers in
# higher precision automatically (mixed precision) -- so I/O stays FP32 and the
# inference wrapper is unchanged across precisions.
if PRECISION == "fp16":
    config.set_flag(trt.BuilderFlag.FP16)

# INT8 is the only precision that needs data: TensorRT runs the calibration set
# through the network to observe activation ranges, then derives per-tensor
# scales. The calibrator object is what feeds it those frames.
calibrator = None
calib_cache = None
if PRECISION == "int8":
    from int8_calibrator import CALIBRATORS      # noqa: E402 -- needs CUDA pinned first

    config.set_flag(trt.BuilderFlag.INT8)
    if INT8_ALLOW_FP16:
        config.set_flag(trt.BuilderFlag.FP16)

    calib_cache = ENGINE_PATH.with_name(f"best_int8_{CALIBRATOR}.calib_cache")
    calibrator = CALIBRATORS[CALIBRATOR](calib_images, calib_cache, IMG_SIZE)
    config.int8_calibrator = calibrator

# Unconditional, not an elif on PRECISION -- see the ALLOW_TF32 note above.
if not ALLOW_TF32:
    config.clear_flag(trt.BuilderFlag.TF32)

# DETAILED keeps per-layer metadata in the engine so the precision each layer
# actually got can be read back after the build. That is the only way to catch a
# silent fallback -- an INT8 build whose calibration failed still produces a
# working engine, it is just FP32 wearing an INT8 label. Does not affect kernel
# selection or numerics.
config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

# Read the flags back off the config instead of tracking them in variables:
# provenance should record what the builder was actually configured with, not
# what we assume we asked for.
fp16_flag = config.get_flag(trt.BuilderFlag.FP16)
int8_flag = config.get_flag(trt.BuilderFlag.INT8)
tf32_enabled = config.get_flag(trt.BuilderFlag.TF32)
print(f"  precision={PRECISION}  fp16_flag={fp16_flag}  int8_flag={int8_flag}  "
      f"tf32_enabled={tf32_enabled}")

# The ONNX is static [1, 3, 640, 640], so no optimisation profile is needed.
print(f"\nBuilding {PRECISION.upper()} engine (kernel autotuning on GPU {DEVICE})...")
serialized = builder.build_serialized_network(network, config)
if serialized is None:
    sys.exit("Engine build failed (build_serialized_network returned None).")


# -----------------------------
# Sanity + per-layer precision  -- BEFORE saving
# -----------------------------
# Deliberately ahead of the write. An INT8 build whose calibration failed still
# produces a working engine, it just runs in float under an INT8 name; if that
# check ran after the save we would leave a mislabelled engine on disk with no
# provenance beside it.
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(bytes(serialized))
if engine is None:
    sys.exit("Deserialization failed -- engine is not usable, nothing written.")

# There is NO "Precision" field on a layer in the inspector JSON -- an earlier
# version of this script assumed one and every layer read back as "?", which
# looked exactly like a total INT8 failure on an engine that was in fact 76%
# quantised. The precision a layer ran at is carried by its output tensors'
# Format/Datatype. Read that.
layer_dtypes = {}
int8_layers, float_layers = [], []
try:
    parsed = json.loads(engine.create_engine_inspector().get_engine_information(
        trt.LayerInformationFormat.JSON))
    layers = parsed["Layers"] if isinstance(parsed, dict) else parsed
    for L in layers:
        if not isinstance(L, dict):
            continue
        dts = {o.get("Format/Datatype", "?") for o in (L.get("Outputs") or [])}
        for t in dts:
            layer_dtypes[t] = layer_dtypes.get(t, 0) + 1
        (int8_layers if any("Int8" in t for t in dts)
         else float_layers).append(L.get("Name", ""))
    n_layers_total = len(layers)
except Exception as e:                      # inspection is diagnostic, not fatal
    layer_dtypes = {"unavailable": str(e)}
    n_layers_total = 0

n_int8 = len(int8_layers)
print("\nLayer output-datatype breakdown:")
for k, v in sorted(layer_dtypes.items(),
                   key=lambda x: -x[1] if isinstance(x[1], int) else 0):
    if isinstance(v, int) and n_layers_total:
        print(f"  {k:<10} {v:>4}  ({100 * v / n_layers_total:.1f}%)")
    else:
        print(f"  {k}: {v}")

if PRECISION == "int8":
    if n_int8 == 0:
        sys.exit(
            "INT8 build produced ZERO INT8 layers -- calibration failed and this "
            "engine is not actually quantised. Nothing written."
        )
    print(f"  -> {n_int8}/{n_layers_total} layers emit INT8 "
          f"({100 * n_int8 / n_layers_total:.1f}%)")
    # Which layers declined INT8 is the useful diagnostic, not just how many.
    print(f"  -> {len(float_layers)} layers stayed float "
          f"(attention / detection head / reformat nodes)")

with open(ENGINE_PATH, "wb") as f:
    f.write(serialized)
print("\nSaved engine:", ENGINE_PATH, f"({ENGINE_PATH.stat().st_size / 1e6:.1f} MB)")

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
    "int8_flag": int8_flag,
    "tf32_enabled": tf32_enabled,
    # Read back from the engine, not asserted: layer output datatypes are how
    # you tell a real INT8 engine from a float one wearing an INT8 label.
    "layer_output_dtypes": layer_dtypes,
    "int8_layer_count": n_int8,
    "float_layer_count": len(float_layers),
    "engine_layer_count": n_layers_total,
    "float_layer_names": float_layers,
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

# INT8 only: record WHAT this engine was calibrated on. Without this the engine
# is unfalsifiable -- an accuracy number with no way to check whether the
# calibration data overlapped the evaluation set.
if PRECISION == "int8":
    manifest_head = {}
    for line in calib_manifest.read_text().splitlines():
        if not line.startswith("#"):
            break
        if ":" in line:
            k, _, v = line.lstrip("# ").partition(":")
            manifest_head[k.strip()] = v.strip()
    prov["calibration"] = {
        "calibrator": CALIBRATOR,
        "calibrator_class": type(calibrator).__name__,
        "deprecated_api": True,       # implicit quantisation, deprecated in TRT 10.1
        "n_frames": len(calib_images),
        "calib_dir": str(CALIB_DIR),
        "manifest": str(calib_manifest),
        "manifest_sha256": sha256(calib_manifest),
        "calib_set_sha256": manifest_head.get("set_sha256"),
        "source_split": manifest_head.get("source_split"),
        "seed": manifest_head.get("seed"),
        "cache": str(calib_cache),
        "cache_sha256": sha256(calib_cache) if calib_cache.exists() else None,
        "int8_allow_fp16": INT8_ALLOW_FP16,
    }
prov_path = ENGINE_PATH.with_name(ENGINE_PATH.name + ".provenance.json")
prov_path.write_text(json.dumps(prov, indent=2))
print("\nProvenance:", prov_path)
print(f"\nDONE -- {PRECISION.upper()} engine built and deserialised cleanly on GPU {DEVICE}.")
print("NOTE: this is a build + validity check, not an accuracy or latency measurement.")
