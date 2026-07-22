from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import hashlib
import json
import os
import platform
import sys

import numpy as np
import cv2
import torch
from ultralytics import YOLO


# -----------------------------
# Settings
# -----------------------------
# The FP32 baseline checkpoint -- the exact .pt used for the latency/accuracy
# baseline. We do not modify it; export writes a sibling .onnx.
MODEL_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/best.pt"
)

# imgsz is taken from the baseline, NOT assumed: scripts/benchmark_latency.py
# uses IMG_SIZE (default 640) and the baseline runs used 640.
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))

# Export shape. dynamic=False + batch=1 bakes a fixed [1, 3, IMG_SIZE, IMG_SIZE]
# input into the ONNX graph -- there is no batch dimension to sweep. This
# matches the batch=1 latency baseline; a batch sweep is explicitly out of scope.
BATCH = 1
DYNAMIC = False

# FP32 only. No half precision at export -- precision changes happen later, in
# TensorRT, deliberately and one at a time.
HALF = False

# onnxslim graph simplification. We record below whether it actually ran
# (it only runs if the onnxslim package is importable).
SIMPLIFY = True

# opset 17: supported by TensorRT 8.5+ (including the tensorrt:26.05 devcontainer,
# TRT 10.x) and onnxruntime >= 1.14. Pinned explicitly for reproducibility and
# recorded in the provenance file below.
OPSET = int(os.environ.get("OPSET", 17))

# Parity frames: the SAME fixed subset the baseline used, taken in the SAME
# sorted order (deterministic), first N frames.
IMG_DIR = Path(
    "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/images/val"
)
N_PARITY = int(os.environ.get("N_PARITY", 16))

# Detection-level comparison threshold. conf 0.25 matches the baseline. YOLO26 is
# end-to-end: both the .pt forward and the .onnx already emit final, NMS-applied
# detections [1, 300, 6] (padded with zero-conf rows), so there is NO NMS step in
# this script -- we filter each model's own final detections at the same conf and
# match boxes. Any detection difference is therefore export drift, not postprocess.
CONF = float(os.environ.get("CONF", 0.25))

# Parity tolerances (the gate). For an end-to-end model the meaningful numeric
# fidelity is over MATCHED detection pairs, not the full padded [1,300,6] tensor
# (whose element-wise diff is dominated by padding/row-alignment, not drift):
#   BOX_IOU_MIN  -- floor for a matched box pair (identical input => ~1.0)
#   COORD_ATOL   -- max abs corner-coordinate diff, in pixels, over matched pairs
#   CONF_ATOL    -- max abs confidence diff over matched pairs
BOX_IOU_MIN = float(os.environ.get("BOX_IOU_MIN", 0.98))
COORD_ATOL = float(os.environ.get("COORD_ATOL", 1.0))
CONF_ATOL = float(os.environ.get("CONF_ATOL", 1e-3))

# Parity runs on CPU: deterministic and independent of GPU contention. The
# export itself is a CPU trace. GPU precision work belongs to the TensorRT step.
DEVICE = "cpu"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pkg_version(name):
    try:
        return __import__(name).__version__
    except Exception:
        return None


def letterbox(img, new_shape):
    """
    Resize + pad to a square new_shape, Ultralytics-style: preserve aspect,
    pad with 114, centered, scale-up allowed. Returns HWC BGR uint8. Both models
    are fed the identical result, so parity isolates export fidelity, not
    preprocessing.
    """
    h0, w0 = img.shape[:2]
    r = min(new_shape / h0, new_shape / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    top, left = (new_shape - nh) // 2, (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(img_bgr, imgsz):
    """BGR HWC uint8 -> NCHW float32 RGB, /255, batch=1. Fed identically to both."""
    lb = letterbox(img_bgr, imgsz)
    im = lb[:, :, ::-1].transpose(2, 0, 1)                 # BGR->RGB, HWC->CHW
    im = np.ascontiguousarray(im, dtype=np.float32) / 255.0
    return im[None]                                        # [1, 3, imgsz, imgsz]


def pt_raw_forward(net, im_np):
    """Raw detection tensor from the PyTorch model on a preprocessed input."""
    im = torch.from_numpy(im_np)
    with torch.inference_mode():
        out = net(im)
    out = out[0] if isinstance(out, (list, tuple)) else out
    return out.float().cpu().numpy()                       # [1, 4+nc, N]


def box_iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = ua + ub - inter
    return inter / u if u > 0 else 0.0


def extract_dets(raw, conf):
    """
    Final detections from an end-to-end YOLO26 output tensor [1, 300, 6] =
    x1,y1,x2,y2,conf,cls, padded with zero-conf rows. NMS is already applied
    inside the model, so we just drop rows below the conf threshold. Returns
    ndarray [n, 6] = x1,y1,x2,y2,conf,cls, the layout compare_detections expects.
    """
    d = raw[0]                                # [300, 6]
    keep = d[:, 4] >= conf
    return d[keep]


def compare_detections(pt_dets, onnx_dets, iou_min):
    """
    End-to-end parity on final detections (greedy box matching). Requires: same
    count, same class per matched pair, matched-box IoU >= iou_min. Returns
    (ok, msg, metrics) where metrics carries the numeric fidelity over matched
    pairs -- min_matched_iou, max_coord_diff (px), max_conf_diff -- which is the
    meaningful signal for an NMS-free model, not the full padded-tensor diff.
    """
    if len(pt_dets) != len(onnx_dets):
        return False, f"box count pt={len(pt_dets)} onnx={len(onnx_dets)}", {}
    if len(pt_dets) == 0:
        return True, "ok (no dets)", {"min_matched_iou": None,
                                       "max_coord_diff": 0.0, "max_conf_diff": 0.0}
    used = set()
    ious, coord_diffs, conf_diffs = [], [], []
    for pd in pt_dets:
        best, best_j = -1.0, None
        for j, od in enumerate(onnx_dets):
            if j in used:
                continue
            iou = box_iou_xyxy(pd[:4], od[:4])
            if iou > best:
                best, best_j = iou, j
        if best_j is None:
            return False, "no onnx box to match", {}
        od = onnx_dets[best_j]
        if int(pd[5]) != int(od[5]):
            return False, f"class flip pt={int(pd[5])} onnx={int(od[5])}", {}
        if best < iou_min:
            return False, f"matched IoU {best:.4f} < {iou_min}", {}
        used.add(best_j)
        ious.append(best)
        coord_diffs.append(float(np.abs(pd[:4].astype(np.float64) - od[:4].astype(np.float64)).max()))
        conf_diffs.append(abs(float(pd[4]) - float(od[4])))
    return True, "ok", {
        "min_matched_iou": float(min(ious)),
        "max_coord_diff": float(max(coord_diffs)),
        "max_conf_diff": float(max(conf_diffs)),
    }


# -----------------------------
# Export
# -----------------------------
print("============================================================")
print("FP32 YOLO26n -> ONNX export with PyTorch<->ONNX parity gate")
print("============================================================")
print("Model:", MODEL_PATH)
print("imgsz:", IMG_SIZE, " batch:", BATCH, " dynamic:", DYNAMIC,
      " half:", HALF, " opset:", OPSET, " simplify:", SIMPLIFY)

if not MODEL_PATH.exists():
    sys.exit(f"Model not found: {MODEL_PATH}")

onnxslim_version = pkg_version("onnxslim")
if SIMPLIFY and onnxslim_version is None:
    print("NOTE: simplify=True but onnxslim is not installed; Ultralytics will "
          "skip simplification. Install onnxslim to enable it.")

print("\nExporting...")
model = YOLO(str(MODEL_PATH))
# One-time offline export cost, recorded for the report -- not a per-inference
# number. Times only the export call, not the parity/accuracy checks that follow.
_export_t0 = perf_counter()
onnx_out = model.export(
    format="onnx",
    imgsz=IMG_SIZE,
    dynamic=DYNAMIC,
    batch=BATCH,
    half=HALF,
    simplify=SIMPLIFY,
    opset=OPSET,
    device=DEVICE,
)
export_seconds = perf_counter() - _export_t0
onnx_path = Path(onnx_out)
print("Exported:", onnx_path, f"({export_seconds:.1f} s)")

# Read back what the exporter actually baked in (do not trust our request alone).
import onnx  # noqa: E402  -- imported here so the export error path is clearer
onnx_model = onnx.load(str(onnx_path))
opset_in_model = max(op.version for op in onnx_model.opset_import if op.domain in ("", "ai.onnx"))
graph_in = onnx_model.graph.input[0]
onnx_input_shape = [
    (d.dim_value if (d.HasField("dim_value")) else d.dim_param or "?")
    for d in graph_in.type.tensor_type.shape.dim
]
simplify_ran = SIMPLIFY and onnxslim_version is not None
print(f"opset in model: {opset_in_model}   input shape: {onnx_input_shape}   "
      f"simplify ran: {simplify_ran}")


# -----------------------------
# Parity check  (THE POINT OF THE SCRIPT)
# -----------------------------
import onnxruntime as ort  # noqa: E402

# Fresh PyTorch model for the reference forward -- export can fuse in place, so
# reload from disk to compare against the un-exported .pt.
pt_net = YOLO(str(MODEL_PATH)).model.float().eval()

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name

frames = sorted(IMG_DIR.glob("*.jpg"))[:N_PARITY]
if not frames:
    sys.exit(f"No parity frames found in {IMG_DIR}")

print(f"\nParity on {len(frames)} frames (CPU), conf={CONF}, box_iou_min={BOX_IOU_MIN}, "
      f"coord_atol={COORD_ATOL}px, conf_atol={CONF_ATOL}")
print("-" * 78)
print(f"{'frame':<34}{'coord_d':>10}{'conf_d':>10}{'min_iou':>9}{'pt_box':>7}{'onnx_box':>9}{'':>4}")
print("-" * 78)

max_coord_all, max_conf_all = 0.0, 0.0
per_frame = []
all_pass = True

for fp in frames:
    img = cv2.imread(str(fp))
    if img is None:
        sys.exit(f"Could not read frame: {fp}")
    im = preprocess(img, IMG_SIZE)

    pt_raw = pt_raw_forward(pt_net, im)
    onnx_raw = sess.run(None, {inp_name: im})[0]

    if pt_raw.shape != onnx_raw.shape:
        sys.exit(f"Shape mismatch on {fp.name}: pt {pt_raw.shape} vs onnx {onnx_raw.shape}")

    pt_dets = extract_dets(pt_raw, CONF)
    onnx_dets = extract_dets(onnx_raw, CONF)
    det_ok, det_msg, m = compare_detections(pt_dets, onnx_dets, BOX_IOU_MIN)

    coord_d = m.get("max_coord_diff", float("inf"))
    conf_d = m.get("max_conf_diff", float("inf"))
    min_iou = m.get("min_matched_iou", None)
    num_ok = det_ok and coord_d <= COORD_ATOL and conf_d <= CONF_ATOL
    frame_pass = det_ok and num_ok
    all_pass = all_pass and frame_pass
    if det_ok:
        max_coord_all = max(max_coord_all, coord_d)
        max_conf_all = max(max_conf_all, conf_d)

    per_frame.append({
        "frame": fp.name,
        "max_coord_diff": coord_d if det_ok else None,
        "max_conf_diff": conf_d if det_ok else None,
        "min_matched_iou": min_iou,
        "pt_boxes": int(len(pt_dets)),
        "onnx_boxes": int(len(onnx_dets)),
        "det_ok": bool(det_ok),
        "num_ok": bool(num_ok),
        "det_note": det_msg,
        "pass": bool(frame_pass),
    })
    iou_str = f"{min_iou:.4f}" if min_iou is not None else "  -   "
    fail_reason = det_msg if not det_ok else ("coord/conf>atol" if not num_ok else "")
    print(f"{fp.name:<34}{coord_d:>10.2e}{conf_d:>10.2e}{iou_str:>9}{len(pt_dets):>7}{len(onnx_dets):>9}"
          f"{('PASS' if frame_pass else 'FAIL'):>4}"
          + ("" if frame_pass else f"   <- {fail_reason}"))

print("-" * 78)
result = "PASS" if all_pass else "FAIL"
print(f"matched-pair max_coord_diff={max_coord_all:.3e}px  max_conf_diff={max_conf_all:.3e}  "
      f"(coord_atol={COORD_ATOL}, conf_atol={CONF_ATOL})   =>   {result}")


# -----------------------------
# Provenance  (next to the .onnx; reproducible from this record)
# -----------------------------
prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "export_seconds": round(export_seconds, 2),   # one-time offline export cost
    "host": platform.node(),
    "parity_device": DEVICE,
    "result": result,
    "versions": {
        "python": platform.python_version(),
        "ultralytics": pkg_version("ultralytics"),
        "torch": pkg_version("torch"),
        "onnx": pkg_version("onnx"),
        "onnxruntime": pkg_version("onnxruntime"),
        "onnxslim": onnxslim_version,
        "numpy": pkg_version("numpy"),
        "opencv": cv2.__version__,
    },
    "export": {
        "opset_requested": OPSET,
        "opset_in_model": int(opset_in_model),
        "imgsz": IMG_SIZE,
        "batch": BATCH,
        "dynamic": DYNAMIC,
        "half": HALF,
        "simplify_requested": SIMPLIFY,
        "simplify_ran": simplify_ran,
        "onnx_input_shape": onnx_input_shape,
    },
    "source_pt": {
        "path": str(MODEL_PATH),
        "sha256": sha256(MODEL_PATH),
        "bytes": MODEL_PATH.stat().st_size,
    },
    "output_onnx": {
        "path": str(onnx_path),
        "sha256": sha256(onnx_path),
        "bytes": onnx_path.stat().st_size,
    },
    "parity": {
        "n_frames": len(frames),
        "frames": [f.name for f in frames],
        "conf": CONF,
        "box_iou_min": BOX_IOU_MIN,
        "coord_atol": COORD_ATOL,
        "conf_atol": CONF_ATOL,
        "max_coord_diff": max_coord_all,
        "max_conf_diff": max_conf_all,
        "per_frame": per_frame,
    },
}

prov_path = onnx_path.with_name(onnx_path.name + ".provenance.json")
prov_path.write_text(json.dumps(prov, indent=2))
print("Provenance:", prov_path)

if not all_pass:
    sys.exit("PARITY FAILED - the ONNX export does NOT match the PyTorch model. "
             "Do not use this ONNX for TensorRT.")
print("\nPARITY PASSED - ONNX matches PyTorch within tolerance. Safe for the "
      "TensorRT step (next, separately).")
