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
# Export a QAT fake-quant model (modelopt state from train_qat.py) to ONNX with
# Q/DQ (QuantizeLinear / DequantizeLinear) nodes, so TensorRT can fold them into
# a real INT8 engine.
#
# WHY THIS SHAPE OF SCRIPT. The obvious approach -- torch.onnx.export() under
# modelopt's export_torch_mode() -- FAILS on this model: the quantiser scale
# (_amax) traces as a graph input where modelopt's INT8 symbolic needs an
# onnx::Constant (SymbolicValueError, "got prim::Param"). That failure is
# independent of the exporter backend (legacy and dynamo both fail) and of the
# torch version. modelopt's OWN blessed path, get_onnx_bytes_and_metadata(),
# handles it -- it wraps the export in torch.inference_mode with the right graph
# post-processing. So this script calls that, NOT torch.onnx.export directly.
# Requires modelopt >= 0.31 (Python >= 3.10); the project's Py3.11 export venv.
#
# HEAD EXPORT MODE. Restored raw, YOLO26n's forward returns 11 tensors (the
# [1,300,6] detections plus 10 one2many/one2one auxiliary heads). Ultralytics'
# own exporter sets Detect.export=True so forward returns only the deployment
# output; we replicate that (eval + requires_grad off + Detect.export/format)
# to get a clean single [1,300,6] graph -- faster and simpler for TensorRT.
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
BASE_MODEL = REPO / "models/yolo26n_sanoscience_full_left/best.pt"
QAT_STATE = Path(os.environ.get(
    "QAT_STATE", str(REPO / "runs_qat/qat_smoke_v2/qat_modelopt_state.pt")))
OUT_ONNX = Path(os.environ.get(
    "OUT_ONNX", str(REPO / "models/yolo26n_sanoscience_full_left/best_qat.onnx")))

IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
BATCH = int(os.environ.get("BATCH", 1))
OPSET = int(os.environ.get("OPSET", 17))

DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=cpu or DEVICE=1). No GPU specified, no run.")
os.environ["CUDA_VISIBLE_DEVICES"] = "" if DEVICE == "cpu" else DEVICE

import torch                                              # noqa: E402
from ultralytics import YOLO                              # noqa: E402
from ultralytics.nn.modules.head import Detect            # noqa: E402
import modelopt                                           # noqa: E402
import modelopt.torch.opt as mto                          # noqa: E402
from modelopt.torch.quantization.nn import TensorQuantizer  # noqa: E402
from modelopt.torch._deploy.utils.torch_onnx import (     # noqa: E402
    get_onnx_bytes_and_metadata, OnnxBytes)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count_quantizers(model):
    total = sum(1 for m in model.modules() if isinstance(m, TensorQuantizer))
    with_amax = sum(1 for m in model.modules()
                    if isinstance(m, TensorQuantizer)
                    and getattr(m, "_amax", None) is not None)
    return total, with_amax


print("=" * 60)
print("QAT model -> ONNX (Q/DQ) export")
print("=" * 60)
print("modelopt:", modelopt.__version__, " torch:", torch.__version__)
print("base model:", BASE_MODEL)
print("qat state: ", QAT_STATE)
print("out onnx:  ", OUT_ONNX)
print("imgsz:", IMG_SIZE, " batch:", BATCH, " opset:", OPSET, " device:", DEVICE)

if not BASE_MODEL.exists():
    sys.exit(f"Baseline checkpoint not found: {BASE_MODEL}")
if not QAT_STATE.exists():
    sys.exit(f"QAT modelopt state not found: {QAT_STATE}")

# -----------------------------
# Restore the quantised model (same path as train_qat.py verify_reload)
# -----------------------------
dev = "cpu" if DEVICE == "cpu" else "cuda"
y = YOLO(str(BASE_MODEL))
mto.restore(y.model, str(QAT_STATE))
model = y.model.to(dev).eval()
n_quant, with_amax = count_quantizers(model)
print(f"\n[export] restored {n_quant} TensorQuantizers ({with_amax} with scales)")
if n_quant == 0:
    sys.exit("[export] no quantizers after restore -- state did not apply.")

# -----------------------------
# Put the Detect head into export mode (single [1,300,6] output)
# -----------------------------
for p in model.parameters():
    p.requires_grad = False
n_heads = 0
for m in model.modules():
    if isinstance(m, Detect):
        m.export = True
        m.format = "onnx"
        m.dynamic = False
        n_heads += 1
print(f"[export] set export mode on {n_heads} Detect head(s) -> single deployment output")

# -----------------------------
# Export via modelopt's blessed path (NOT torch.onnx.export -- see header)
# -----------------------------
print("\n[export] exporting via modelopt get_onnx_bytes_and_metadata() ...")
dummy = torch.zeros(BATCH, 3, IMG_SIZE, IMG_SIZE, device=dev)
_t0 = perf_counter()
payload, _meta = get_onnx_bytes_and_metadata(model, dummy, onnx_opset=OPSET)
export_seconds = perf_counter() - _t0

ob = OnnxBytes.from_bytes(payload) if isinstance(payload, (bytes, bytearray)) else payload
tmp_dir = OUT_ONNX.parent / "_qat_onnx_tmp"
ob.write_to_disk(str(tmp_dir))
import glob  # noqa: E402
produced = sorted(glob.glob(str(tmp_dir / "**" / "*.onnx"), recursive=True))
if not produced:
    sys.exit("[export] no .onnx produced by modelopt export")
src = Path(produced[0])
OUT_ONNX.parent.mkdir(parents=True, exist_ok=True)
src.replace(OUT_ONNX)
print(f"[export] saved: {OUT_ONNX}  ({export_seconds:.1f} s)")

# -----------------------------
# Verify Q/DQ nodes and the single deployment output survived
# -----------------------------
import onnx  # noqa: E402
m = onnx.load(str(OUT_ONNX))
op_counts = {}
for node in m.graph.node:
    op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
n_q = op_counts.get("QuantizeLinear", 0)
n_dq = op_counts.get("DequantizeLinear", 0)


def shape_of(v):
    return [d.dim_value if d.dim_value else d.dim_param
            for d in v.type.tensor_type.shape.dim]


print(f"\n[export] ONNX: {len(m.graph.node)} nodes, "
      f"QuantizeLinear={n_q}, DequantizeLinear={n_dq}")
print(f"[export]   inputs:  {[(i.name, shape_of(i)) for i in m.graph.input]}")
print(f"[export]   outputs: {[(o.name, shape_of(o)) for o in m.graph.output]}")

prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "qat_onnx_export",
    "modelopt": modelopt.__version__,
    "torch": torch.__version__,
    "base_model": str(BASE_MODEL),
    "qat_state": str(QAT_STATE),
    "qat_state_sha256": sha256(QAT_STATE),
    "out_onnx": str(OUT_ONNX),
    "out_onnx_sha256": sha256(OUT_ONNX),
    "imgsz": IMG_SIZE, "batch": BATCH, "opset": OPSET, "device": DEVICE,
    "quantizers_restored": n_quant,
    "quantizers_with_scales": with_amax,
    "onnx_nodes_total": len(m.graph.node),
    "quantize_linear_nodes": n_q,
    "dequantize_linear_nodes": n_dq,
    "num_outputs": len(m.graph.output),
    "export_seconds": export_seconds,
}
prov_path = OUT_ONNX.with_suffix(".onnx.provenance.json")
prov_path.write_text(json.dumps(prov, indent=2))
print(f"[export] provenance: {prov_path}")

if n_q == 0 or n_dq == 0:
    sys.exit("[export] FAIL: no Q/DQ nodes -- TensorRT would build FP32, not INT8.")
if len(m.graph.output) != 1:
    print(f"[export] WARNING: expected 1 output, got {len(m.graph.output)} "
          "-- head export mode may not have collapsed the aux heads.")
print(f"[export] DONE -- {n_q} Q / {n_dq} DQ nodes, "
      f"{len(m.graph.output)} output(s). Ready for TRT INT8 build.")
