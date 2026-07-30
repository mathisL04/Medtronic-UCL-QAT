import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter


# -----------------------------
# Settings
# -----------------------------
# Build a TensorRT INT8 engine from a Q/DQ (QuantizeLinear/DequantizeLinear) ONNX
# produced by the QAT export (scripts/export_qat_onnx.py). This is EXPLICIT
# quantization: the scales live in the Q/DQ nodes, so there is NO calibrator and
# no calibration dataset -- TensorRT reads the scales straight from the graph.
#
# This is the QAT counterpart to build_tensorrt_engine.py's PTQ int8 path, which
# uses an IInt8Calibrator to derive scales from unlabelled data. Do not confuse
# them: PTQ = calibrator + FP32 ONNX; QAT = no calibrator + Q/DQ ONNX.
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
ONNX_PATH = Path(os.environ.get(
    "ONNX_PATH", str(REPO / "models/yolo26n_sanoscience_full_left/best_qat.onnx")))
ENGINE_PATH = Path(os.environ.get(
    "ENGINE_PATH", str(REPO / "models/yolo26n_sanoscience_full_left/best_qat_int8.engine")))
WORKSPACE_GB = float(os.environ.get("WORKSPACE_GB", 8))

# DEVICE has no default and the script aborts if unset -- same discipline as the
# benchmarks. A build autotunes by timing kernels on the live GPU, so a
# mis-attributed GPU corrupts the autotune just as it would a latency number.
DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=1). No GPU specified, no run.")
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

import tensorrt as trt                                    # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


print("=" * 60)
print("TensorRT INT8 engine from Q/DQ ONNX (QAT, no calibrator)")
print("=" * 60)
print("tensorrt:", trt.__version__, " device:", DEVICE)
print("onnx:   ", ONNX_PATH)
print("engine: ", ENGINE_PATH)
if not ONNX_PATH.exists():
    sys.exit(f"ONNX not found: {ONNX_PATH}")

logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(0)
parser = trt.OnnxParser(network, logger)
with open(ONNX_PATH, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("PARSE ERROR:", parser.get_error(i))
        sys.exit("ONNX parse failed")
print(f"parsed: {network.num_layers} layers, "
      f"{network.num_inputs} inputs, {network.num_outputs} outputs")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(WORKSPACE_GB * (1 << 30)))
# Explicit Q/DQ -> INT8. TensorRT reads per-tensor/per-channel scales from the
# QuantizeLinear/DequantizeLinear nodes; layers without Q/DQ run in FP32.
config.set_flag(trt.BuilderFlag.INT8)

print("building INT8 engine (reading scales from Q/DQ nodes) ...")
_t0 = perf_counter()
serialized = builder.build_serialized_network(network, config)
build_seconds = perf_counter() - _t0
if serialized is None:
    sys.exit("BUILD FAILED: builder returned None")
data = bytes(serialized)
ENGINE_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(ENGINE_PATH, "wb") as f:
    f.write(data)
print(f"engine built + saved: {ENGINE_PATH} ({len(data)/1e6:.1f} MB, {build_seconds:.1f} s)")

# Deserialise to prove the engine is loadable, and report the I/O contract.
runtime = trt.Runtime(logger)
engine = runtime.deserialize_cuda_engine(data)
if engine is None:
    sys.exit("DESERIALIZE FAILED")
io = []
print("deserialised OK. I/O tensors:")
for i in range(engine.num_io_tensors):
    n = engine.get_tensor_name(i)
    mode = engine.get_tensor_mode(n).name
    shape = list(engine.get_tensor_shape(n))
    dtype = engine.get_tensor_dtype(n).name
    print(f"  {mode:6} {n:10} {shape} {dtype}")
    io.append({"name": n, "mode": mode, "shape": shape, "dtype": dtype})

prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "tensorrt_int8_qdq_build",
    "tensorrt_version": trt.__version__,
    "device": DEVICE,
    "onnx_path": str(ONNX_PATH),
    "onnx_sha256": sha256(ONNX_PATH),
    "engine_path": str(ENGINE_PATH),
    "engine_sha256": sha256(ENGINE_PATH),
    "engine_bytes": len(data),
    "workspace_gb": WORKSPACE_GB,
    "quantization": "explicit Q/DQ (no calibrator)",
    "num_layers": network.num_layers,
    "io": io,
    "build_seconds": build_seconds,
}
prov_path = Path(str(ENGINE_PATH) + ".provenance.json")
prov_path.write_text(json.dumps(prov, indent=2))
print(f"provenance: {prov_path}")
print("DONE -- INT8 engine built from Q/DQ ONNX and deserialised cleanly.")
