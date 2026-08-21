import os
import sys
import json
import hashlib
import platform
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing the model, precision,
# calibration data, input size, output engine or target GPU.
# ==========================================================

# Validated ONNX model to compile.
# CHANGE when using another architecture/checkpoint/export.
ONNX_PATH = Path(
    os.environ.get(
        "ONNX_PATH",
        "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
        "models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.onnx",
    )
)

# TensorRT precision:
# fp32 | fp16 | int8
PRECISION = os.environ.get("PRECISION", "fp32").lower()


# ----------------------------------------------------------
# PTQ INT8 SETTINGS
# Only required when PRECISION=int8.
# CHANGE when using another calibration dataset.
# ----------------------------------------------------------

CALIB_DIR = Path(
    os.environ.get(
        "CALIB_DIR",
        "/home/zcemml1/medtronic_qat_data/calib_int8_train_yolo",
    )
)

CALIBRATOR = os.environ.get(
    "CALIBRATOR",
    "entropy",
).lower()

# Allow FP16 fallback for layers not executed in INT8.
INT8_ALLOW_FP16 = (
    os.environ.get(
        "INT8_ALLOW_FP16",
        "0",
    )
    == "1"
)


# ----------------------------------------------------------
# HARDWARE / BUILD SETTINGS
# ----------------------------------------------------------

# Explicit target GPU.
# CHANGE according to available NVIDIA device.
DEVICE = os.environ.get("DEVICE")

if DEVICE is None:
    sys.exit(
        "DEVICE is required "
        "(e.g. DEVICE=0)."
    )

os.environ[
    "CUDA_VISIBLE_DEVICES"
] = DEVICE


# TensorRT normally permits TF32 on supported GPUs.
# Keep disabled for strict FP32/FP16/INT8 comparisons.
ALLOW_TF32 = (
    os.environ.get(
        "ALLOW_TF32",
        "0",
    )
    == "1"
)


# TensorRT builder scratch workspace.
# CHANGE according to model size and available GPU memory.
WORKSPACE_GB = float(
    os.environ.get(
        "WORKSPACE_GB",
        8,
    )
)


# Must match the ONNX input resolution.
# CHANGE if using another model input size.
IMG_SIZE = int(
    os.environ.get(
        "IMG_SIZE",
        640,
    )
)


# Output engine path.
# CHANGE naming if another project structure is used.
ENGINE_PATH = ONNX_PATH.with_name(
    f"best_{PRECISION}.engine"
)


sys.path.insert(
    0,
    str(
        Path(__file__)
        .resolve()
        .parent
    ),
)


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


def gpu_name(index):
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
                "-i",
                str(index),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return out.stdout.strip()

    except Exception:
        return None


# ==========================================================
# INPUT CHECKS
# ==========================================================

print("=" * 60)
print(
    f"ONNX -> TensorRT "
    f"{PRECISION.upper()} engine"
)
print("=" * 60)

print("ONNX:", ONNX_PATH)
print("Engine:", ENGINE_PATH)
print(
    "Device:",
    DEVICE,
    f"({gpu_name(DEVICE)})",
)
print(
    "TensorRT:",
    trt.__version__,
)
print(
    "Workspace:",
    WORKSPACE_GB,
    "GiB",
)


if not ONNX_PATH.exists():
    sys.exit(
        f"ONNX not found: "
        f"{ONNX_PATH}"
    )


if PRECISION not in (
    "fp32",
    "fp16",
    "int8",
):
    sys.exit(
        "PRECISION must be "
        "fp32, fp16 or int8."
    )


calib_images = []
calib_manifest = None


if PRECISION == "int8":

    if CALIBRATOR not in (
        "entropy",
        "minmax",
    ):
        sys.exit(
            "CALIBRATOR must be "
            "entropy or minmax."
        )

    calib_images = sorted(
        (
            CALIB_DIR
            / "images"
        ).glob(
            "*.jpg"
        )
    )

    if not calib_images:
        sys.exit(
            "No calibration frames found "
            f"in {CALIB_DIR / 'images'}."
        )

    calib_manifest = next(
        iter(
            sorted(
                CALIB_DIR.glob(
                    "calib*_manifest.txt"
                )
            )
        ),
        None,
    )

    if calib_manifest is None:
        sys.exit(
            "No calibration manifest found."
        )

    print(
        "Calibration:",
        CALIB_DIR,
        f"({len(calib_images)} frames, "
        f"{CALIBRATOR})",
    )


# ==========================================================
# PARSE ONNX
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


print("\nParsing ONNX...")


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
                parser.get_error(i)
            )

        sys.exit(
            "ONNX parse failed."
        )


print(
    f"Parsed OK: "
    f"{network.num_layers} layers"
)


# ==========================================================
# BUILDER CONFIGURATION
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


# FP16 enables TensorRT mixed-precision execution.
if PRECISION == "fp16":
    config.set_flag(
        trt.BuilderFlag.FP16
    )


# PTQ INT8 path.
# Requires int8_calibrator.py and a representative
# calibration dataset.
calibrator = None
calib_cache = None


if PRECISION == "int8":

    from int8_calibrator import (
        CALIBRATORS,
    )

    config.set_flag(
        trt.BuilderFlag.INT8
    )

    if INT8_ALLOW_FP16:
        config.set_flag(
            trt.BuilderFlag.FP16
        )

    calib_cache = (
        ENGINE_PATH.with_name(
            f"best_int8_"
            f"{CALIBRATOR}"
            ".calib_cache"
        )
    )

    calibrator = (
        CALIBRATORS[
            CALIBRATOR
        ](
            calib_images,
            calib_cache,
            IMG_SIZE,
        )
    )

    config.int8_calibrator = (
        calibrator
    )


if not ALLOW_TF32:
    config.clear_flag(
        trt.BuilderFlag.TF32
    )


# Keep detailed layer information for
# post-build precision inspection.
config.profiling_verbosity = (
    trt.ProfilingVerbosity.DETAILED
)


fp16_flag = config.get_flag(
    trt.BuilderFlag.FP16
)

int8_flag = config.get_flag(
    trt.BuilderFlag.INT8
)

tf32_enabled = config.get_flag(
    trt.BuilderFlag.TF32
)


print(
    f"precision={PRECISION} "
    f"fp16={fp16_flag} "
    f"int8={int8_flag} "
    f"tf32={tf32_enabled}"
)


# ==========================================================
# BUILD ENGINE
# ==========================================================

print(
    f"\nBuilding "
    f"{PRECISION.upper()} engine "
    f"on GPU {DEVICE}..."
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
        "TensorRT engine build failed."
    )


print(
    f"Build completed in "
    f"{build_seconds:.1f} s"
)


# ==========================================================
# ENGINE VALIDITY + PRECISION INSPECTION
# ==========================================================

runtime = trt.Runtime(
    logger
)

engine = (
    runtime
    .deserialize_cuda_engine(
        bytes(serialized)
    )
)


if engine is None:
    sys.exit(
        "Engine deserialization failed."
    )


layer_dtypes = {}
int8_layers = []
float_layers = []


try:

    inspector = (
        engine
        .create_engine_inspector()
    )

    parsed = json.loads(
        inspector
        .get_engine_information(
            trt.LayerInformationFormat.JSON
        )
    )

    layers = (
        parsed["Layers"]
        if isinstance(
            parsed,
            dict,
        )
        else parsed
    )

    for layer in layers:

        if not isinstance(
            layer,
            dict,
        ):
            continue

        dtypes = {
            output.get(
                "Format/Datatype",
                "?",
            )
            for output in (
                layer.get(
                    "Outputs"
                )
                or []
            )
        }

        for dtype in dtypes:
            layer_dtypes[
                dtype
            ] = (
                layer_dtypes.get(
                    dtype,
                    0,
                )
                + 1
            )

        if any(
            "Int8" in dtype
            for dtype in dtypes
        ):
            int8_layers.append(
                layer.get(
                    "Name",
                    "",
                )
            )
        else:
            float_layers.append(
                layer.get(
                    "Name",
                    "",
                )
            )

    n_layers_total = len(
        layers
    )


except Exception as e:

    layer_dtypes = {
        "unavailable": str(e)
    }

    n_layers_total = 0


n_int8 = len(
    int8_layers
)


print(
    "\nLayer output datatypes:"
)


for key, value in (
    layer_dtypes.items()
):

    if (
        isinstance(
            value,
            int,
        )
        and n_layers_total
    ):

        print(
            f"  {key:<20}"
            f"{value:>4} "
            f"({100 * value / n_layers_total:.1f}%)"
        )

    else:
        print(
            f"  {key}: {value}"
        )


# INT8-specific sanity gate.
# For another architecture, the percentage of INT8 layers
# may differ substantially; only zero INT8 layers is treated
# as an invalid INT8 build here.
if PRECISION == "int8":

    if n_int8 == 0:
        sys.exit(
            "INT8 build produced "
            "zero INT8 layers."
        )

    print(
        f"{n_int8}/"
        f"{n_layers_total} "
        "layers emit INT8"
    )

    print(
        f"{len(float_layers)} "
        "layers remain non-INT8"
    )


# ==========================================================
# SAVE ENGINE
# ==========================================================

with open(
    ENGINE_PATH,
    "wb",
) as f:
    f.write(
        serialized
    )


print(
    "\nSaved engine:",
    ENGINE_PATH,
    f"({ENGINE_PATH.stat().st_size / 1e6:.1f} MB)",
)


# ==========================================================
# ENGINE I/O
# ==========================================================

io = []


for i in range(
    engine.num_io_tensors
):

    name = (
        engine
        .get_tensor_name(i)
    )

    tensor = {
        "name":
            name,

        "mode":
            str(
                engine
                .get_tensor_mode(
                    name
                )
            ).split(".")[-1],

        "shape":
            list(
                engine
                .get_tensor_shape(
                    name
                )
            ),

        "dtype":
            str(
                engine
                .get_tensor_dtype(
                    name
                )
            ).split(".")[-1],
    }

    io.append(
        tensor
    )


print(
    "\nEngine I/O:"
)


for tensor in io:

    print(
        f"  {tensor['mode']:<6} "
        f"{tensor['name']:<12} "
        f"{tensor['shape']} "
        f"{tensor['dtype']}"
    )


# ==========================================================
# PROVENANCE
# ==========================================================

prov = {

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "build_seconds":
        round(
            build_seconds,
            2,
        ),

    "host":
        platform.node(),

    "device_index":
        DEVICE,

    "gpu_name":
        gpu_name(
            DEVICE
        ),

    "precision":
        PRECISION,

    "fp16_flag":
        fp16_flag,

    "int8_flag":
        int8_flag,

    "tf32_enabled":
        tf32_enabled,

    "layer_output_dtypes":
        layer_dtypes,

    "int8_layer_count":
        n_int8,

    "float_layer_count":
        len(
            float_layers
        ),

    "engine_layer_count":
        n_layers_total,

    "float_layer_names":
        float_layers,

    "tensorrt_version":
        trt.__version__,

    "workspace_gib":
        WORKSPACE_GB,

    "num_layers":
        network.num_layers,

    "source_onnx": {
        "path":
            str(
                ONNX_PATH
            ),

        "sha256":
            sha256(
                ONNX_PATH
            ),

        "bytes":
            ONNX_PATH.stat().st_size,
    },

    "output_engine": {
        "path":
            str(
                ENGINE_PATH
            ),

        "sha256":
            sha256(
                ENGINE_PATH
            ),

        "bytes":
            ENGINE_PATH.stat().st_size,
    },

    "io_tensors":
        io,
}


# Extra provenance required for PTQ INT8.
if PRECISION == "int8":

    manifest_head = {}

    for line in (
        calib_manifest
        .read_text()
        .splitlines()
    ):

        if not line.startswith(
            "#"
        ):
            break

        if ":" in line:

            key, _, value = (
                line
                .lstrip("# ")
                .partition(":")
            )

            manifest_head[
                key.strip()
            ] = value.strip()


    prov[
        "calibration"
    ] = {

        "calibrator":
            CALIBRATOR,

        "calibrator_class":
            type(
                calibrator
            ).__name__,

        "n_frames":
            len(
                calib_images
            ),

        "calib_dir":
            str(
                CALIB_DIR
            ),

        "manifest":
            str(
                calib_manifest
            ),

        "manifest_sha256":
            sha256(
                calib_manifest
            ),

        "calib_set_sha256":
            manifest_head.get(
                "set_sha256"
            ),

        "source_split":
            manifest_head.get(
                "source_split"
            ),

        "seed":
            manifest_head.get(
                "seed"
            ),

        "cache":
            str(
                calib_cache
            ),

        "cache_sha256":
            (
                sha256(
                    calib_cache
                )
                if calib_cache.exists()
                else None
            ),

        "int8_allow_fp16":
            INT8_ALLOW_FP16,
    }


prov_path = (
    ENGINE_PATH.with_name(
        ENGINE_PATH.name
        + ".provenance.json"
    )
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
    f"\nDONE -- "
    f"{PRECISION.upper()} "
    "engine built successfully."
)
