import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cv2


# ==========================================================
# MODIFY FOR ANOTHER MODEL / ENGINE
#
# ENGINE_PATH = TensorRT engine being validated
# ONNX_PATH   = reference ONNX used to build / compare it
#
# For a new model, these two files must correspond to the
# same trained network.
# ==========================================================

ENGINE_PATH = Path(os.environ.get(
    "ENGINE_PATH",
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "results/models/yolo26n_sanoscience_full_left/1_fp32/best_fp32.engine",
))

ONNX_PATH = Path(os.environ.get(
    "ONNX_PATH",
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "results/models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.onnx",
))


# ==========================================================
# MODIFY FOR ANOTHER DATASET
#
# Only images are required for parity.
# Labels are not used because this is NOT an accuracy test.
# ==========================================================

IMG_DIR = Path(os.environ.get(
    "IMG_DIR",
    "/home/zcemml1/medtronic_qat_data/"
    "demo_val100_random_yolo/images/val",
))


# ==========================================================
# MODIFY FOR INPUT / EVALUATION SETTINGS
# ==========================================================

N_PARITY = int(
    os.environ.get("N_PARITY", 100)
)

IMG_SIZE = int(
    os.environ.get("IMG_SIZE", 640)
)

CONF = float(
    os.environ.get("CONF", 0.25)
)


# ==========================================================
# MODIFY TOLERANCES IF REQUIRED BY A DIFFERENT MODEL /
# PRECISION
#
# These thresholds define when engine <-> ONNX numeric drift
# is still considered acceptable.
# ==========================================================

BOX_IOU_MIN = float(
    os.environ.get("BOX_IOU_MIN", 0.98)
)

COORD_ATOL = float(
    os.environ.get("COORD_ATOL", 1.0)
)

CONF_ATOL = float(
    os.environ.get("CONF_ATOL", 5e-3)
)


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = os.environ.get("DEVICE")

if DEVICE is None:
    sys.exit(
        "DEVICE is required "
        "(e.g. DEVICE=2)."
    )

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE


import tensorrt as trt
from cuda.bindings import runtime as cudart
import onnxruntime as ort


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost


def CHK(ret):
    err, rest = (
        (ret[0], ret[1:])
        if isinstance(ret, (tuple, list))
        else (ret, ())
    )

    if int(err) != 0:
        raise RuntimeError(
            f"CUDA error code {int(err)}"
        )

    if not rest:
        return None

    return rest[0] if len(rest) == 1 else rest


class TRTEngine:
    def __init__(self, path, logger):

        runtime = trt.Runtime(logger)

        self.engine = runtime.deserialize_cuda_engine(
            Path(path).read_bytes()
        )

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {path}"
            )

        self.context = (
            self.engine.create_execution_context()
        )

        self.inp_name = None
        self.out_name = None

        for i in range(
            self.engine.num_io_tensors
        ):

            n = self.engine.get_tensor_name(i)

            if (
                self.engine.get_tensor_mode(n)
                == trt.TensorIOMode.INPUT
            ):

                self.inp_name = n
                self.inp_shape = tuple(
                    self.engine.get_tensor_shape(n)
                )

            else:

                self.out_name = n
                self.out_shape = tuple(
                    self.engine.get_tensor_shape(n)
                )

        self.out_host = np.empty(
            self.out_shape,
            dtype=np.float32,
        )

        self.d_in = CHK(
            cudart.cudaMalloc(
                int(np.prod(self.inp_shape)) * 4
            )
        )

        self.d_out = CHK(
            cudart.cudaMalloc(
                int(np.prod(self.out_shape)) * 4
            )
        )

        self.stream = CHK(
            cudart.cudaStreamCreate()
        )

        self.context.set_tensor_address(
            self.inp_name,
            int(self.d_in),
        )

        self.context.set_tensor_address(
            self.out_name,
            int(self.d_out),
        )

    def infer(self, im):

        im = np.ascontiguousarray(
            im,
            dtype=np.float32,
        )

        CHK(
            cudart.cudaMemcpyAsync(
                self.d_in,
                im.ctypes.data,
                im.nbytes,
                H2D,
                self.stream,
            )
        )

        self.context.execute_async_v3(
            int(self.stream)
        )

        CHK(
            cudart.cudaMemcpyAsync(
                self.out_host.ctypes.data,
                self.d_out,
                self.out_host.nbytes,
                D2H,
                self.stream,
            )
        )

        CHK(
            cudart.cudaStreamSynchronize(
                self.stream
            )
        )

        return self.out_host.copy()


# ==========================================================
# MODIFY IF PREPROCESSING CHANGES
#
# This must match:
#   - ONNX export
#   - TensorRT inference
#   - training/evaluation preprocessing
# ==========================================================

def letterbox(img, new_shape):

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
        left:left + nw,
    ] = resized

    return canvas


def preprocess(img_bgr, imgsz):

    lb = letterbox(
        img_bgr,
        imgsz,
    )

    im = (
        lb[:, :, ::-1]
        .transpose(2, 0, 1)
    )

    im = np.ascontiguousarray(
        im,
        dtype=np.float32,
    ) / 255.0

    return im[None]


def box_iou_xyxy(a, b):

    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])

    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    ua = (
        max(0.0, a[2] - a[0])
        * max(0.0, a[3] - a[1])
    )

    ub = (
        max(0.0, b[2] - b[0])
        * max(0.0, b[3] - b[1])
    )

    union = ua + ub - inter

    return (
        inter / union
        if union > 0
        else 0.0
    )


# ==========================================================
# MODIFY IF MODEL OUTPUT FORMAT CHANGES
#
# Current model output:
#
#   [1, 300, 6]
#
# Each row:
#   x1, y1, x2, y2, confidence, class
#
# NMS is already included in YOLO26's deployment output.
#
# For another architecture, replace this function with the
# appropriate decoding / NMS logic.
# ==========================================================

def extract_dets(raw, conf):

    d = raw[0]

    return d[
        d[:, 4] >= conf
    ]


def compare_detections(
    a_dets,
    b_dets,
    iou_min,
):

    if len(a_dets) != len(b_dets):

        return (
            False,
            f"box count a={len(a_dets)} "
            f"b={len(b_dets)}",
            {},
        )

    if len(a_dets) == 0:

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

    for pd in a_dets:

        best = -1.0
        best_j = None

        for j, od in enumerate(
            b_dets
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
                "no box to match",
                {},
            )

        od = b_dets[best_j]

        if int(pd[5]) != int(od[5]):

            return (
                False,
                f"class flip "
                f"a={int(pd[5])} "
                f"b={int(od[5])}",
                {},
            )

        if best < iou_min:

            return (
                False,
                f"matched IoU "
                f"{best:.4f} < {iou_min}",
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

            "pairs": [
                {
                    "iou": float(i),
                    "coord_diff": float(c),
                    "conf_diff": float(f),
                }
                for i, c, f
                in zip(
                    ious,
                    coord_diffs,
                    conf_diffs,
                )
            ],
        },
    )


print("=" * 60)
print("TensorRT engine <-> ONNX parity")
print("=" * 60)

print("Engine:", ENGINE_PATH)
print("ONNX:", ONNX_PATH)
print("Device:", DEVICE)
print("Frames:", N_PARITY)
print("Confidence:", CONF)


if not ENGINE_PATH.exists():

    sys.exit(
        f"Engine not found: "
        f"{ENGINE_PATH}"
    )


if not ONNX_PATH.exists():

    sys.exit(
        f"ONNX not found: "
        f"{ONNX_PATH}"
    )


logger = trt.Logger(
    trt.Logger.WARNING
)

eng = TRTEngine(
    ENGINE_PATH,
    logger,
)

sess = ort.InferenceSession(
    str(ONNX_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

onnx_in = (
    sess.get_inputs()[0].name
)


frames = sorted(
    IMG_DIR.glob("*.jpg")
)[:N_PARITY]

if not frames:

    sys.exit(
        f"No frames in {IMG_DIR}"
    )


print("-" * 74)

print(
    f"{'frame':<34}"
    f"{'coord_d':>10}"
    f"{'conf_d':>10}"
    f"{'eng':>5}"
    f"{'onnx':>6}"
    f"{'':>4}"
)

print("-" * 74)


max_coord_all = 0.0
max_conf_all = 0.0

per_frame = []
all_pairs = []

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

    eng_raw = eng.infer(im)

    onnx_raw = sess.run(
        None,
        {onnx_in: im},
    )[0]


    eng_dets = extract_dets(
        eng_raw,
        CONF,
    )

    onnx_dets = extract_dets(
        onnx_raw,
        CONF,
    )


    ok, msg, m = (
        compare_detections(
            eng_dets,
            onnx_dets,
            BOX_IOU_MIN,
        )
    )


    coord_d = m.get(
        "max_coord_diff",
        float("inf"),
    )

    conf_d = m.get(
        "max_conf_diff",
        float("inf"),
    )


    num_ok = (
        ok
        and coord_d <= COORD_ATOL
        and conf_d <= CONF_ATOL
    )

    frame_pass = (
        ok and num_ok
    )

    all_pass = (
        all_pass
        and frame_pass
    )


    if ok:

        max_coord_all = max(
            max_coord_all,
            coord_d,
        )

        max_conf_all = max(
            max_conf_all,
            conf_d,
        )

        for pair in m.get(
            "pairs",
            [],
        ):

            all_pairs.append(
                dict(
                    pair,
                    frame=fp.name,
                )
            )


    per_frame.append({

        "frame":
            fp.name,

        "eng_boxes":
            int(len(eng_dets)),

        "onnx_boxes":
            int(len(onnx_dets)),

        "max_coord_diff":
            coord_d if ok else None,

        "max_conf_diff":
            conf_d if ok else None,

        "min_matched_iou":
            m.get(
                "min_matched_iou"
            ),

        "det_ok":
            bool(ok),

        "det_note":
            msg,

        "pass":
            bool(frame_pass),
    })


    if not frame_pass:

        reason = (
            msg
            if not ok
            else "coord/conf>atol"
        )

        print(
            f"{fp.name:<34}"
            f"{coord_d:>10.2e}"
            f"{conf_d:>10.2e}"
            f"{len(eng_dets):>5}"
            f"{len(onnx_dets):>6}"
            f"  FAIL <- {reason}"
        )


n_pass = sum(
    1
    for r in per_frame
    if r["pass"]
)


print("-" * 74)


result = (
    "PASS"
    if all_pass
    else "FAIL"
)


print(
    f"{n_pass}/{len(frames)} frames pass   "
    f"max_coord_diff="
    f"{max_coord_all:.3e}px   "
    f"max_conf_diff="
    f"{max_conf_all:.3e}   "
    f"=> {result}"
)


def pctile(vals, q):

    if not vals:
        return float("nan")

    s = sorted(vals)

    k = (
        (len(s) - 1)
        * q
    )

    lo = int(k)

    hi = min(
        int(k) + 1,
        len(s) - 1,
    )

    return (
        s[lo]
        + (s[hi] - s[lo])
        * (k - lo)
    )


def dist_row(
    label,
    vals,
    atol,
):

    over = sum(
        1
        for v in vals
        if v > atol
    )

    return (
        f"{label:<18}"
        f"{pctile(vals, .5):>11.3e}"
        f"{pctile(vals, .95):>11.3e}"
        f"{pctile(vals, .99):>11.3e}"
        f"{max(vals):>11.3e}"
        f"{over:>7d}"
        f" ({100 * over / len(vals):.1f}%)"
    )


coord_vals = [
    p["coord_diff"]
    for p in all_pairs
]

conf_vals = [
    p["conf_diff"]
    for p in all_pairs
]

iou_vals = [
    p["iou"]
    for p in all_pairs
]


pd_summary = {}


if all_pairs:

    n_over_coord = sum(
        1
        for v in coord_vals
        if v > COORD_ATOL
    )

    n_over_conf = sum(
        1
        for v in conf_vals
        if v > CONF_ATOL
    )

    n_over_any = sum(
        1
        for p in all_pairs
        if (
            p["coord_diff"]
            > COORD_ATOL
            or
            p["conf_diff"]
            > CONF_ATOL
        )
    )


    print()

    print(
        f"Per-detection deltas over "
        f"{len(all_pairs)} matched pairs"
    )

    print("-" * 74)

    print(
        f"{'':<18}"
        f"{'median':>11}"
        f"{'p95':>11}"
        f"{'p99':>11}"
        f"{'max':>11}"
        f"{'over atol':>17}"
    )

    print(
        dist_row(
            "coord_diff (px)",
            coord_vals,
            COORD_ATOL,
        )
    )

    print(
        dist_row(
            "conf_diff",
            conf_vals,
            CONF_ATOL,
        )
    )

    print(
        f"{'matched IoU':<18}"
        f"{pctile(iou_vals, .5):>11.5f}"
        f"{'min':>11} "
        f"{min(iou_vals):.5f}"
    )

    print("-" * 74)


    pd_summary = {

        "n_pairs":
            len(all_pairs),

        "coord_diff": {

            "median":
                pctile(
                    coord_vals,
                    .5,
                ),

            "p95":
                pctile(
                    coord_vals,
                    .95,
                ),

            "p99":
                pctile(
                    coord_vals,
                    .99,
                ),

            "max":
                max(coord_vals),

            "n_over_atol":
                n_over_coord,
        },

        "conf_diff": {

            "median":
                pctile(
                    conf_vals,
                    .5,
                ),

            "p95":
                pctile(
                    conf_vals,
                    .95,
                ),

            "p99":
                pctile(
                    conf_vals,
                    .99,
                ),

            "max":
                max(conf_vals),

            "n_over_atol":
                n_over_conf,
        },

        "matched_iou": {

            "median":
                pctile(
                    iou_vals,
                    .5,
                ),

            "min":
                min(iou_vals),
        },

        "n_within_tolerance":
            len(all_pairs)
            - n_over_any,
    }


prov = {

    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "engine":
        str(ENGINE_PATH),

    "onnx":
        str(ONNX_PATH),

    "device":
        DEVICE,

    "engine_sha256":
        sha256(ENGINE_PATH),

    "onnx_sha256":
        sha256(ONNX_PATH),

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

    "n_pass":
        n_pass,

    "result":
        result,

    "per_detection":
        pd_summary,

    "per_detection_pairs":
        all_pairs,

    "per_frame":
        per_frame,
}


out = ENGINE_PATH.with_name(
    ENGINE_PATH.name
    + ".parity.json"
)

out.write_text(
    json.dumps(
        prov,
        indent=2,
    )
)

print(
    "Parity record:",
    out,
)


if not all_pass:

    sys.exit(
        "PARITY FAILED -- engine detections "
        "diverge from the ONNX reference."
    )


print(
    f"\nPARITY PASSED -- "
    f"{n_pass}/{len(frames)} frames."
)

print(
    "This validates engine faithfulness, "
    "not model accuracy."
)
