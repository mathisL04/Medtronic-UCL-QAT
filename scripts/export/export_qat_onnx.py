import os
import sys
import json
import hashlib
import glob
from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing the model, QAT state,
# input dimensions, output location or ONNX export settings.
# ==========================================================

# Project repository.
REPO = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT"
)

# Baseline model used to reconstruct the architecture before
# replaying the ModelOpt QAT state.
# CHANGE when using another model or baseline checkpoint.
BASE_MODEL = Path(
    os.environ.get(
        "BASE_MODEL",
        str(
            REPO
            / "results/models/yolo26n_sanoscience_full_left"
            / "baseline/best.pt"
        ),
    )
)

# Trained ModelOpt QAT state.
# CHANGE when exporting another QAT experiment.
QAT_STATE = Path(
    os.environ.get(
        "QAT_STATE",
        str(
            REPO
            / "results/models/yolo26n_sanoscience_full_left"
            / "qat/v6_final/qat_modelopt_state_best.pt"
        ),
    )
)

# Destination ONNX model.
OUT_ONNX = Path(
    os.environ.get(
        "OUT_ONNX",
        str(
            REPO
            / "results/models/yolo26n_sanoscience_full_left"
            / "qat/v6_final/best_qat.onnx"
        ),
    )
)

# Model input dimensions.
# CHANGE if the replacement model uses another input resolution.
IMG_SIZE = int(
    os.environ.get("IMG_SIZE", 640)
)

# Export batch size.
BATCH = int(
    os.environ.get("BATCH", 1)
)

# ONNX operator set.
# Verify compatibility with ModelOpt and TensorRT before changing.
OPSET = int(
    os.environ.get("OPSET", 17)
)

# Explicit device selection.
DEVICE = os.environ.get("DEVICE")

if DEVICE is None:
    sys.exit(
        "DEVICE is required "
        "(e.g. DEVICE=cpu or DEVICE=1)."
    )

os.environ["CUDA_VISIBLE_DEVICES"] = (
    ""
    if DEVICE == "cpu"
    else DEVICE
)


import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

import modelopt
import modelopt.torch.opt as mto

from modelopt.torch.quantization.nn import (
    TensorQuantizer,
)

from modelopt.torch._deploy.utils.torch_onnx import (
    get_onnx_bytes_and_metadata,
    OnnxBytes,
)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def count_quantizers(model):
    total = sum(
        1
        for m in model.modules()
        if isinstance(
            m,
            TensorQuantizer,
        )
    )

    with_amax = sum(
        1
        for m in model.modules()
        if isinstance(
            m,
            TensorQuantizer,
        )
        and getattr(
            m,
            "_amax",
            None,
        )
        is not None
    )

    return total, with_amax


print("=" * 60)
print("QAT model -> ONNX Q/DQ export")
print("=" * 60)

print("ModelOpt:", modelopt.__version__)
print("PyTorch:", torch.__version__)
print("Base model:", BASE_MODEL)
print("QAT state:", QAT_STATE)
print("Output ONNX:", OUT_ONNX)
print("Image size:", IMG_SIZE)
print("Batch:", BATCH)
print("Opset:", OPSET)
print("Device:", DEVICE)


if not BASE_MODEL.exists():
    sys.exit(
        f"Baseline checkpoint not found: "
        f"{BASE_MODEL}"
    )

if not QAT_STATE.exists():
    sys.exit(
        f"QAT state not found: "
        f"{QAT_STATE}"
    )


# ==========================================================
# RESTORE QAT MODEL
# ==========================================================

dev = (
    "cpu"
    if DEVICE == "cpu"
    else "cuda"
)

# Architecture-specific loader.
# CHANGE if the new model is not loaded through Ultralytics YOLO.
y = YOLO(
    str(BASE_MODEL)
)

# Replay the ModelOpt quantization structure, trained weights
# and quantizer state onto the base architecture.
mto.restore(
    y.model,
    str(QAT_STATE),
)

model = (
    y.model
    .to(dev)
    .eval()
)


n_quant, with_amax = count_quantizers(
    model
)

print(
    f"\n[export] restored "
    f"{n_quant} TensorQuantizers "
    f"({with_amax} with scales)"
)

if n_quant == 0:
    sys.exit(
        "[export] no quantizers restored."
    )


# ==========================================================
# MODEL-SPECIFIC DEPLOYMENT OUTPUT
# ==========================================================

for p in model.parameters():
    p.requires_grad = False


# YOLO26-specific export configuration.
# YOLO26 contains training-time auxiliary detection branches.
# Detect.export=True keeps only the deployment output.
#
# CHANGE OR REMOVE this block if another model has a different
# detection head or already exposes a deployment-ready output.
n_heads = 0

for m in model.modules():

    if isinstance(
        m,
        Detect,
    ):
        m.export = True
        m.format = "onnx"
        m.dynamic = False
        n_heads += 1


print(
    f"[export] configured "
    f"{n_heads} detection head(s) "
    "for deployment output"
)


# ==========================================================
# QAT -> Q/DQ ONNX
# ==========================================================

print(
    "\n[export] exporting through "
    "ModelOpt ONNX deployment API..."
)


# Input tensor shape.
# CHANGE if the new architecture uses a different number of
# channels, spatial dimensions or tensor layout.
dummy = torch.zeros(
    BATCH,
    3,
    IMG_SIZE,
    IMG_SIZE,
    device=dev,
)


t0 = perf_counter()

payload, metadata = (
    get_onnx_bytes_and_metadata(
        model,
        dummy,
        onnx_opset=OPSET,
    )
)

export_seconds = (
    perf_counter()
    - t0
)


ob = (
    OnnxBytes.from_bytes(payload)
    if isinstance(
        payload,
        (bytes, bytearray),
    )
    else payload
)


tmp_dir = (
    OUT_ONNX.parent
    / "_qat_onnx_tmp"
)

ob.write_to_disk(
    str(tmp_dir)
)


produced = sorted(
    glob.glob(
        str(
            tmp_dir
            / "**"
            / "*.onnx"
        ),
        recursive=True,
    )
)


if not produced:
    sys.exit(
        "[export] no ONNX file produced."
    )


src = Path(
    produced[0]
)

OUT_ONNX.parent.mkdir(
    parents=True,
    exist_ok=True,
)

src.replace(
    OUT_ONNX
)


print(
    f"[export] saved: "
    f"{OUT_ONNX} "
    f"({export_seconds:.1f} s)"
)


# ==========================================================
# VERIFY EXPLICIT QUANTIZATION GRAPH
# ==========================================================

import onnx


onnx_model = onnx.load(
    str(OUT_ONNX)
)


op_counts = {}

for node in onnx_model.graph.node:
    op_counts[node.op_type] = (
        op_counts.get(
            node.op_type,
            0,
        )
        + 1
    )


n_q = op_counts.get(
    "QuantizeLinear",
    0,
)

n_dq = op_counts.get(
    "DequantizeLinear",
    0,
)


def shape_of(value):
    return [
        (
            d.dim_value
            if d.dim_value
            else d.dim_param
        )
        for d in (
            value
            .type
            .tensor_type
            .shape
            .dim
        )
    ]


inputs = [
    (
        i.name,
        shape_of(i),
    )
    for i in (
        onnx_model.graph.input
    )
]

outputs = [
    (
        o.name,
        shape_of(o),
    )
    for o in (
        onnx_model.graph.output
    )
]


print(
    f"\n[export] ONNX nodes: "
    f"{len(onnx_model.graph.node)}"
)

print(
    f"[export] QuantizeLinear: {n_q}"
)

print(
    f"[export] DequantizeLinear: {n_dq}"
)

print(
    f"[export] inputs: {inputs}"
)

print(
    f"[export] outputs: {outputs}"
)


# A QAT deployment graph must contain explicit Q/DQ nodes.
if n_q == 0 or n_dq == 0:
    sys.exit(
        "[export] FAIL: "
        "no Q/DQ nodes found."
    )


# YOLO26-specific expectation.
# CHANGE if another model legitimately exports multiple outputs.
if len(
    onnx_model.graph.output
) != 1:
    print(
        "[export] WARNING: "
        "expected one deployment output, "
        f"found {len(onnx_model.graph.output)}."
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
        "qat_onnx_export",

    "modelopt":
        modelopt.__version__,

    "torch":
        torch.__version__,

    "base_model":
        str(
            BASE_MODEL
        ),

    "base_model_sha256":
        sha256(
            BASE_MODEL
        ),

    "qat_state":
        str(
            QAT_STATE
        ),

    "qat_state_sha256":
        sha256(
            QAT_STATE
        ),

    "out_onnx":
        str(
            OUT_ONNX
        ),

    "out_onnx_sha256":
        sha256(
            OUT_ONNX
        ),

    "imgsz":
        IMG_SIZE,

    "batch":
        BATCH,

    "opset":
        OPSET,

    "device":
        DEVICE,

    "quantizers_restored":
        n_quant,

    "quantizers_with_scales":
        with_amax,

    "onnx_nodes_total":
        len(
            onnx_model.graph.node
        ),

    "quantize_linear_nodes":
        n_q,

    "dequantize_linear_nodes":
        n_dq,

    "num_outputs":
        len(
            onnx_model.graph.output
        ),

    "inputs":
        inputs,

    "outputs":
        outputs,

    "export_seconds":
        export_seconds,
}


prov_path = (
    OUT_ONNX.with_suffix(
        ".onnx.provenance.json"
    )
)


prov_path.write_text(
    json.dumps(
        prov,
        indent=2,
    )
)


print(
    f"[export] provenance: "
    f"{prov_path}"
)


print(
    f"\n[export] DONE -- "
    f"{n_q} Q / {n_dq} DQ nodes, "
    f"{len(onnx_model.graph.output)} output(s)."
)

print(
    "Ready for TensorRT INT8 build."
)
