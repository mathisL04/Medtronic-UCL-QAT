import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter


# ==========================================================
# USER CONFIGURATION
# Modify these values when changing the QAT model,
# deployment location or target hardware.
# ==========================================================

# Project root.
# CHANGE if the repository is moved.
REPO = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT"
)


# QAT ONNX containing explicit QuantizeLinear /
# DequantizeLinear nodes.
# CHANGE when deploying another QAT-trained model.
ONNX_PATH = Path(
    os.environ.get(
        "ONNX_PATH",
        str(
            REPO
            / "models/yolo26n_sanoscience_full_left/"
              "qat/v6_final/best_qat.onnx"
        ),
    )
)


# TensorRT engine output.
# CHANGE when using another model/run/output directory.
ENGINE_PATH = Path(
    os.environ.get(
        "ENGINE_PATH",
        str(
            REPO
            / "models/yolo26n_sanoscience_full_left/"
              "qat/v6_final/best_qat_int8.engine"
        ),
    )
)


# TensorRT builder workspace.
# CHANGE according to model size and available GPU memory.
WORKSPACE_GB = float(
    os.environ.get(
        "WORKSPACE_GB",
        8,
    )
)


# Explicit target GPU.
# CHANGE according to available deployment/build hardware.
DEVICE = os.environ.get("DEVICE")

if DEVICE is None:
    sys.exit(
        "DEVICE is required "
        "(e.g. DEVICE=0)."
    )

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE


import tensorrt as trt


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


# ==========================================================
# INPUT CHECK
# ==========================================================

print("=" * 60)
print(
    "TensorRT INT8 engine "
    "from explicit Q/DQ ONNX"
)
print("=" * 60)

print(
    "TensorRT:",
    trt.__version__,
)

print(
    "Device:",
    DEVICE,
)

print(
    "ONNX:",
    ONNX_PATH,
)

print(
    "Engine:",
    ENGINE_PATH,
)


if not ONNX_PATH.exists():
    sys.exit(
        f"ONNX not found: "
        f"{ONNX_PATH}"
    )


# ==========================================================
# PARSE Q/DQ ONNX
# ==========================================================

logger = trt.Logger(
    trt.Logger.WARNING
)

builder = trt.Builder(
    logger
)

network = builder.create_network(
    0
)

parser = trt.OnnxParser(
    network,
    logger,
)


with open(
    ONNX_PATH,
    "rb",
) as f:

    if not parser.parse(
        f.read()
    ):

        for i in range(
            parser.num_errors
        ):
            print(
                "PARSE ERROR:",
                parser.get_error(i),
            )

        sys.exit(
            "ONNX parse failed."
        )


print(
    f"Parsed: "
    f"{network.num_layers} layers, "
    f"{network.num_inputs} inputs, "
    f"{network.num_outputs} outputs"
)


# ==========================================================
# TENSORRT CONFIGURATION
# ==========================================================

config = (
    builder
    .create_builder_config()
)


config.set_memory_pool_limit(
    trt.MemoryPoolType.WORKSPACE,
    int(
        WORKSPACE_GB
        * (1 << 30)
    ),
)


# Explicit Q/DQ INT8 build.
#
# IMPORTANT:
# No calibration dataset is required here.
# Quantization scales are already encoded inside the
# QuantizeLinear / DequantizeLinear nodes of the ONNX graph.
config.set_flag(
    trt.BuilderFlag.INT8
)


# ==========================================================
# BUILD ENGINE
# ==========================================================

print(
    "\nBuilding TensorRT INT8 engine "
    "from Q/DQ graph..."
)


build_start = perf_counter()


serialized = (
    builder
    .build_serialized_network(
        network,
        config,
    )
)


build_seconds = (
    perf_counter()
    - build_start
)


if serialized is None:
    sys.exit(
        "TensorRT build failed."
    )


data = bytes(
    serialized
)


ENGINE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with open(
    ENGINE_PATH,
    "wb",
) as f:
    f.write(
        data
    )


print(
    f"Engine saved: "
    f"{ENGINE_PATH}"
)

print(
    f"Size: "
    f"{len(data) / 1e6:.1f} MB"
)

print(
    f"Build time: "
    f"{build_seconds:.1f} s"
)


# ==========================================================
# ENGINE VALIDITY / I-O CHECK
# ==========================================================

runtime = trt.Runtime(
    logger
)

engine = (
    runtime
    .deserialize_cuda_engine(
        data
    )
)


if engine is None:
    sys.exit(
        "Engine deserialization failed."
    )


io = []


print(
    "\nEngine I/O:"
)


for i in range(
    engine.num_io_tensors
):

    name = (
        engine
        .get_tensor_name(i)
    )

    mode = (
        engine
        .get_tensor_mode(name)
        .name
    )

    shape = list(
        engine
        .get_tensor_shape(name)
    )

    dtype = (
        engine
        .get_tensor_dtype(name)
        .name
    )


    print(
        f"  {mode:<6} "
        f"{name:<12} "
        f"{shape} "
        f"{dtype}"
    )


    io.append(
        {
            "name": name,
            "mode": mode,
            "shape": shape,
            "dtype": dtype,
        }
    )


# ==========================================================
# PROVENANCE
# ==========================================================

prov = {

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "stage":
        "tensorrt_int8_qdq_build",

    "tensorrt_version":
        trt.__version__,

    "device":
        DEVICE,

    "onnx_path":
        str(
            ONNX_PATH
        ),

    "onnx_sha256":
        sha256(
            ONNX_PATH
        ),

    "engine_path":
        str(
            ENGINE_PATH
        ),

    "engine_sha256":
        sha256(
            ENGINE_PATH
        ),

    "engine_bytes":
        len(
            data
        ),

    "workspace_gb":
        WORKSPACE_GB,

    "quantization":
        "explicit Q/DQ",

    "calibration":
        "none",

    "num_layers":
        network.num_layers,

    "io":
        io,

    "build_seconds":
        build_seconds,
}


prov_path = Path(
    str(
        ENGINE_PATH
    )
    + ".provenance.json"
)


prov_path.write_text(
    json.dumps(
        prov,
        indent=2,
    )
)


print(
    "\nProvenance:",
    prov_path,
)


print(
    "\nDONE -- QAT Q/DQ ONNX "
    "compiled to TensorRT INT8."
)
