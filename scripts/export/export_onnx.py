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


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing the source model,
# input format, ONNX export settings or parity dataset.
# ==========================================================

# Source PyTorch checkpoint.
# Replace when exporting another trained model.
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
        "models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.pt",
    )
)

# Model input size.
# Must match the resolution used during training/evaluation.
IMG_SIZE = int(
    os.environ.get("IMG_SIZE", 640)
)

# ONNX input shape.
# Change if deploying with another batch size or dynamic inputs.
BATCH = int(
    os.environ.get("BATCH", 1)
)
DYNAMIC = (
    os.environ.get("DYNAMIC", "0") == "1"
)

# Baseline export remains FP32.
# Change only if a different export precision is intentionally required.
HALF = (
    os.environ.get("HALF", "0") == "1"
)

# ONNX graph simplification.
SIMPLIFY = (
    os.environ.get("SIMPLIFY", "1") == "1"
)

# ONNX opset.
# Verify compatibility with the target TensorRT version when changing.
OPSET = int(
    os.environ.get("OPSET", 17)
)

# Validation images used for PyTorch <-> ONNX parity.
# Replace when changing dataset.
IMG_DIR = Path(
    os.environ.get(
        "IMG_DIR",
        "/home/zcemml1/medtronic_qat_data/"
        "demo_val100_random_yolo/images/val",
    )
)

N_PARITY = int(
    os.environ.get("N_PARITY", 16)
)

# Detection threshold used only for parity comparison.
# This is not the mAP evaluation threshold.
CONF = float(
    os.environ.get("CONF", 0.25)
)

# Parity tolerances.
# Revisit these if the model output representation changes.
BOX_IOU_MIN = float(
    os.environ.get("BOX_IOU_MIN", 0.98)
)
COORD_ATOL = float(
    os.environ.get("COORD_ATOL", 1.0)
)
CONF_ATOL = float(
    os.environ.get("CONF_ATOL", 1e-3)
)

# CPU is used for the reference export/parity check.
DEVICE = os.environ.get(
    "DEVICE",
    "cpu"
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


def pkg_version(name):
    try:
        return __import__(name).__version__
    except Exception:
        return None


def letterbox(img, new_shape):

    # Model-specific preprocessing.
    # Modify if the replacement model does not use
    # Ultralytics-style letterbox resizing.
    h0, w0 = img.shape[:2]

    r = min(
        new_shape / h0,
        new_shape / w0,
    )

    nw = int(round(w0 * r))
    nh = int(round(h0 * r))

    resized = cv2.resize(
        img,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (new_shape, new_shape, 3),
        114,
        dtype=np.uint8,
    )

    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2

    canvas[
        top:top + nh,
        left:left + nw
    ] = resized

    return canvas


def preprocess(img_bgr, imgsz):

    # Modify if the new model uses another channel order,
    # normalization rule or tensor layout.
    lb = letterbox(
        img_bgr,
        imgsz,
    )

    im = lb[
        :, :, ::-1
    ].transpose(
        2, 0, 1
    )

    im = (
        np.ascontiguousarray(
            im,
            dtype=np.float32,
        )
        / 255.0
    )

    return im[None]


def pt_raw_forward(net, im_np):
    im = torch.from_numpy(
        im_np
    )

    with torch.inference_mode():
        out = net(im)

    out = (
        out[0]
        if isinstance(
            out,
            (list, tuple),
        )
        else out
    )

    return (
        out.float()
        .cpu()
        .numpy()
    )


def box_iou_xyxy(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(
        0.0,
        ix2 - ix1,
    )
    ih = max(
        0.0,
        iy2 - iy1,
    )

    inter = iw * ih

    ua = (
        max(0.0, a[2] - a[0])
        * max(0.0, a[3] - a[1])
    )

    ub = (
        max(0.0, b[2] - b[0])
        * max(0.0, b[3] - b[1])
    )

    union = (
        ua + ub - inter
    )

    return (
        inter / union
        if union > 0
        else 0.0
    )


def extract_dets(raw, conf):

    # YOLO26-specific deployment output:
    # [1, 300, 6] = x1,y1,x2,y2,conf,class.
    # Replace this function if another architecture emits
    # raw logits, separate boxes/scores, or requires external NMS.
    detections = raw[0]

    keep = (
        detections[:, 4]
        >= conf
    )

    return detections[keep]


def compare_detections(
    pt_dets,
    onnx_dets,
    iou_min,
):

    # Model-output parity logic.
    # Replace if the new architecture has a different
    # prediction format or matching strategy.

    if len(pt_dets) != len(onnx_dets):
        return (
            False,
            f"box count pt={len(pt_dets)} "
            f"onnx={len(onnx_dets)}",
            {},
        )

    if len(pt_dets) == 0:
        return (
            True,
            "ok (no dets)",
            {
                "min_matched_iou": None,
                "max_coord_diff": 0.0,
                "max_conf_diff": 0.0,
            },
        )

    used = set()

    ious = []
    coord_diffs = []
    conf_diffs = []

    for pd in pt_dets:

        best = -1.0
        best_j = None

        for j, od in enumerate(
            onnx_dets
        ):

            if j in used:
                continue

            iou = box_iou_xyxy(
                pd[:4],
                od[:4],
            )

            if iou > best:
                best = iou
                best_j = j

        if best_j is None:
            return (
                False,
                "no ONNX box to match",
                {},
            )

        od = onnx_dets[
            best_j
        ]

        if int(pd[5]) != int(od[5]):
            return (
                False,
                "class mismatch",
                {},
            )

        if best < iou_min:
            return (
                False,
                f"matched IoU {best:.4f} "
                f"< {iou_min}",
                {},
            )

        used.add(best_j)

        ious.append(best)

        coord_diffs.append(
            float(
                np.abs(
                    pd[:4].astype(np.float64)
                    - od[:4].astype(np.float64)
                ).max()
            )
        )

        conf_diffs.append(
            abs(
                float(pd[4])
                - float(od[4])
            )
        )

    return (
        True,
        "ok",
        {
            "min_matched_iou":
                float(min(ious)),

            "max_coord_diff":
                float(max(coord_diffs)),

            "max_conf_diff":
                float(max(conf_diffs)),
        },
    )


print("=" * 80)
print("PyTorch -> ONNX export")
print("=" * 80)

print("Model:", MODEL_PATH)
print("Image size:", IMG_SIZE)
print("Batch:", BATCH)
print("Dynamic:", DYNAMIC)
print("Half:", HALF)
print("Opset:", OPSET)
print("Simplify:", SIMPLIFY)


if not MODEL_PATH.exists():
    sys.exit(
        f"Model not found: {MODEL_PATH}"
    )


onnxslim_version = pkg_version(
    "onnxslim"
)


print("\nExporting...")

model = YOLO(
    str(MODEL_PATH)
)

export_t0 = perf_counter()

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

export_seconds = (
    perf_counter()
    - export_t0
)

onnx_path = Path(
    onnx_out
)

print(
    "Exported:",
    onnx_path,
    f"({export_seconds:.1f} s)",
)


import onnx

onnx_model = onnx.load(
    str(onnx_path)
)

opset_in_model = max(
    op.version
    for op in onnx_model.opset_import
    if op.domain in (
        "",
        "ai.onnx",
    )
)

graph_in = (
    onnx_model.graph.input[0]
)

onnx_input_shape = [
    (
        d.dim_value
        if d.HasField("dim_value")
        else d.dim_param or "?"
    )
    for d in (
        graph_in.type.tensor_type
        .shape.dim
    )
]

simplify_ran = (
    SIMPLIFY
    and onnxslim_version
    is not None
)


# ==========================================================
# PARITY VALIDATION
# ==========================================================

import onnxruntime as ort


pt_net = (
    YOLO(
        str(MODEL_PATH)
    )
    .model
    .float()
    .eval()
)


sess = ort.InferenceSession(
    str(onnx_path),
    providers=[
        "CPUExecutionProvider"
    ],
)

inp_name = (
    sess.get_inputs()[0].name
)


frames = sorted(
    IMG_DIR.glob("*.jpg")
)[:N_PARITY]


if not frames:
    sys.exit(
        f"No parity frames found in "
        f"{IMG_DIR}"
    )


max_coord_all = 0.0
max_conf_all = 0.0
per_frame = []
all_pass = True


for fp in frames:

    img = cv2.imread(
        str(fp)
    )

    if img is None:
        sys.exit(
            f"Could not read frame: {fp}"
        )

    im = preprocess(
        img,
        IMG_SIZE,
    )

    pt_raw = pt_raw_forward(
        pt_net,
        im,
    )

    onnx_raw = sess.run(
        None,
        {
            inp_name: im
        },
    )[0]


    # If changing architecture, verify that the PyTorch and
    # ONNX output contracts remain directly comparable.
    if pt_raw.shape != onnx_raw.shape:
        sys.exit(
            f"Shape mismatch: "
            f"PT={pt_raw.shape}, "
            f"ONNX={onnx_raw.shape}"
        )


    pt_dets = extract_dets(
        pt_raw,
        CONF,
    )

    onnx_dets = extract_dets(
        onnx_raw,
        CONF,
    )


    (
        det_ok,
        det_msg,
        metrics,
    ) = compare_detections(
        pt_dets,
        onnx_dets,
        BOX_IOU_MIN,
    )


    coord_d = metrics.get(
        "max_coord_diff",
        float("inf"),
    )

    conf_d = metrics.get(
        "max_conf_diff",
        float("inf"),
    )

    min_iou = metrics.get(
        "min_matched_iou",
        None,
    )


    numerical_ok = (
        det_ok
        and coord_d <= COORD_ATOL
        and conf_d <= CONF_ATOL
    )

    frame_pass = (
        det_ok
        and numerical_ok
    )

    all_pass = (
        all_pass
        and frame_pass
    )


    if det_ok:
        max_coord_all = max(
            max_coord_all,
            coord_d,
        )

        max_conf_all = max(
            max_conf_all,
            conf_d,
        )


    per_frame.append(
        {
            "frame":
                fp.name,

            "max_coord_diff":
                coord_d
                if det_ok
                else None,

            "max_conf_diff":
                conf_d
                if det_ok
                else None,

            "min_matched_iou":
                min_iou,

            "pt_boxes":
                int(
                    len(pt_dets)
                ),

            "onnx_boxes":
                int(
                    len(onnx_dets)
                ),

            "pass":
                bool(
                    frame_pass
                ),
        }
    )


result = (
    "PASS"
    if all_pass
    else "FAIL"
)


print(
    f"\nParity result: {result}"
)

print(
    f"Max coordinate difference: "
    f"{max_coord_all:.3e} px"
)

print(
    f"Max confidence difference: "
    f"{max_conf_all:.3e}"
)


# ==========================================================
# PROVENANCE
# ==========================================================

prov = {
    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "export_seconds":
        round(
            export_seconds,
            2,
        ),

    "host":
        platform.node(),

    "parity_device":
        DEVICE,

    "result":
        result,

    "versions": {
        "python":
            platform.python_version(),

        "ultralytics":
            pkg_version(
                "ultralytics"
            ),

        "torch":
            pkg_version(
                "torch"
            ),

        "onnx":
            pkg_version(
                "onnx"
            ),

        "onnxruntime":
            pkg_version(
                "onnxruntime"
            ),

        "onnxslim":
            onnxslim_version,

        "numpy":
            pkg_version(
                "numpy"
            ),

        "opencv":
            cv2.__version__,
    },

    "export": {
        "opset_requested":
            OPSET,

        "opset_in_model":
            int(
                opset_in_model
            ),

        "imgsz":
            IMG_SIZE,

        "batch":
            BATCH,

        "dynamic":
            DYNAMIC,

        "half":
            HALF,

        "simplify_requested":
            SIMPLIFY,

        "simplify_ran":
            simplify_ran,

        "onnx_input_shape":
            onnx_input_shape,
    },

    "source_pt": {
        "path":
            str(
                MODEL_PATH
            ),

        "sha256":
            sha256(
                MODEL_PATH
            ),

        "bytes":
            MODEL_PATH.stat().st_size,
    },

    "output_onnx": {
        "path":
            str(
                onnx_path
            ),

        "sha256":
            sha256(
                onnx_path
            ),

        "bytes":
            onnx_path.stat().st_size,
    },

    "parity": {
        "n_frames":
            len(frames),

        "conf":
            CONF,

        "box_iou_min":
            BOX_IOU_MIN,

        "coord_atol":
            COORD_ATOL,

        "conf_atol":
            CONF_ATOL,

        "max_coord_diff":
            max_coord_all,

        "max_conf_diff":
            max_conf_all,

        "per_frame":
            per_frame,
    },
}


prov_path = (
    onnx_path.with_name(
        onnx_path.name
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
    "Provenance:",
    prov_path,
)


if not all_pass:
    sys.exit(
        "PARITY FAILED - "
        "do not use this ONNX "
        "for TensorRT."
    )


print(
    "\nPARITY PASSED - "
    "ONNX is ready for TensorRT."
)
