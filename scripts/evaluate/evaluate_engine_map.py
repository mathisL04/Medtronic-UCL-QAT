import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cv2


# ==========================================================
# MODIFY FOR ANOTHER MODEL / EXPERIMENT
# ==========================================================

MODE = os.environ.get("MODE", "engine").lower()

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
# Each root must contain:
#   images/val/
#   labels/val/
#
# Labels are assumed to use YOLO format:
# class cx cy width height
# ==========================================================

EVAL_SET = os.environ.get("EVAL_SET", "val100").lower()

_EVAL_SETS = {
    "val100": Path(
        "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo"
    ),
    "full": Path(
        "/home/zcemml1/medtronic_qat_data/datasets/"
        "sanoscience_yolo_full_nonexpert_stereo"
    ),
}

if EVAL_SET not in _EVAL_SETS:
    sys.exit(
        f"EVAL_SET must be one of {sorted(_EVAL_SETS)} "
        f"(got {EVAL_SET!r})."
    )

_root = _EVAL_SETS[EVAL_SET]

IMG_DIR = _root / "images" / "val"
LABEL_DIR = _root / "labels" / "val"


# ==========================================================
# MODIFY IF MODEL INPUT / OUTPUT CONTRACT CHANGES
# ==========================================================

IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))

CONF = float(os.environ.get("CONF", 0.001))

MAX_DETS = int(os.environ.get("MAX_DETS", 300))


# ==========================================================
# MODIFY FOR ANOTHER DATASET / NUMBER OF CLASSES
# ==========================================================

CLASS_NAMES = {
    0: "surgical_tool",
}


# ==========================================================
# DEVICE
# ==========================================================

DEVICE = os.environ.get("DEVICE")

if MODE == "engine":
    if DEVICE is None:
        sys.exit(
            "DEVICE is required for MODE=engine "
            "(e.g. DEVICE=0)."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

elif MODE != "onnx":
    sys.exit(
        f"MODE must be 'engine' or 'onnx' "
        f"(got {MODE!r})."
    )


from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


# ==========================================================
# MODIFY IF PREPROCESSING CHANGES
#
# Must exactly match preprocessing used for:
# training / calibration / inference.
# ==========================================================

def letterbox_params(h0, w0, s):
    r = min(s / h0, s / w0)

    nw = int(round(w0 * r))
    nh = int(round(h0 * r))

    left = (s - nw) // 2
    top = (s - nh) // 2

    return r, nw, nh, left, top


def preprocess(img_bgr, s):
    h0, w0 = img_bgr.shape[:2]

    r, nw, nh, left, top = letterbox_params(
        h0, w0, s
    )

    resized = cv2.resize(
        img_bgr,
        (nw, nh),
        interpolation=cv2.INTER_LINEAR,
    )

    canvas = np.full(
        (s, s, 3),
        114,
        dtype=np.uint8,
    )

    canvas[
        top:top + nh,
        left:left + nw
    ] = resized

    im = (
        canvas[:, :, ::-1]
        .transpose(2, 0, 1)
    )

    return (
        np.ascontiguousarray(
            im,
            dtype=np.float32
        )[None]
        / 255.0
    )


def unletterbox(boxes, h0, w0, s):
    r, _, _, left, top = letterbox_params(
        h0,
        w0,
        s,
    )

    out = boxes.astype(
        np.float64
    ).copy()

    out[:, [0, 2]] = (
        out[:, [0, 2]]
        - left
    ) / r

    out[:, [1, 3]] = (
        out[:, [1, 3]]
        - top
    ) / r

    out[:, [0, 2]] = (
        out[:, [0, 2]]
        .clip(0, w0)
    )

    out[:, [1, 3]] = (
        out[:, [1, 3]]
        .clip(0, h0)
    )

    return out


class EnginePredictor:

    def __init__(self, path):

        import tensorrt as trt
        from cuda.bindings import runtime as cudart

        self.trt = trt
        self.cudart = cudart

        self.H2D = (
            cudart.cudaMemcpyKind
            .cudaMemcpyHostToDevice
        )

        self.D2H = (
            cudart.cudaMemcpyKind
            .cudaMemcpyDeviceToHost
        )

        runtime = trt.Runtime(
            trt.Logger(
                trt.Logger.WARNING
            )
        )

        self.engine = (
            runtime.deserialize_cuda_engine(
                Path(path).read_bytes()
            )
        )

        if self.engine is None:
            raise RuntimeError(
                f"Failed to deserialize engine: {path}"
            )

        self.context = (
            self.engine.create_execution_context()
        )

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

        self.d_in = self._chk(
            cudart.cudaMalloc(
                int(np.prod(self.inp_shape))
                * 4
            )
        )

        self.d_out = self._chk(
            cudart.cudaMalloc(
                int(np.prod(self.out_shape))
                * 4
            )
        )

        self.stream = self._chk(
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

    @staticmethod
    def _chk(ret):

        err, rest = (
            (ret[0], ret[1:])
            if isinstance(
                ret,
                (tuple, list),
            )
            else (ret, ())
        )

        if int(err) != 0:
            raise RuntimeError(
                f"CUDA error code {int(err)}"
            )

        return (
            rest[0]
            if len(rest) == 1
            else (rest or None)
        )

    def __call__(self, im):

        im = np.ascontiguousarray(
            im,
            dtype=np.float32,
        )

        self._chk(
            self.cudart.cudaMemcpyAsync(
                self.d_in,
                im.ctypes.data,
                im.nbytes,
                self.H2D,
                self.stream,
            )
        )

        self.context.execute_async_v3(
            int(self.stream)
        )

        self._chk(
            self.cudart.cudaMemcpyAsync(
                self.out_host.ctypes.data,
                self.d_out,
                self.out_host.nbytes,
                self.D2H,
                self.stream,
            )
        )

        self._chk(
            self.cudart.cudaStreamSynchronize(
                self.stream
            )
        )

        return self.out_host.copy()


class OnnxPredictor:

    def __init__(self, path):

        import onnxruntime as ort

        self.sess = (
            ort.InferenceSession(
                str(path),
                providers=[
                    "CPUExecutionProvider"
                ],
            )
        )

        self.name = (
            self.sess.get_inputs()[0].name
        )

    def __call__(self, im):

        return self.sess.run(
            None,
            {self.name: im},
        )[0]


# ==========================================================
# MODIFY IF YOUR DATASET LABEL FORMAT CHANGES
#
# Current implementation expects YOLO detection labels.
# ==========================================================

def build_coco_gt(frames):

    images = []
    annotations = []
    ann_id = 1

    for img_id, fp in enumerate(
        frames,
        start=1,
    ):

        img = cv2.imread(str(fp))

        if img is None:
            sys.exit(
                f"Could not read image: {fp}"
            )

        h0, w0 = img.shape[:2]

        images.append({
            "id": img_id,
            "file_name": fp.name,
            "width": w0,
            "height": h0,
        })

        label_fp = (
            LABEL_DIR
            / (fp.stem + ".txt")
        )

        if not label_fp.exists():
            continue

        for line in (
            label_fp
            .read_text()
            .strip()
            .splitlines()
        ):

            if not line.strip():
                continue

            cls, cx, cy, bw, bh = (
                float(v)
                for v in line.split()
            )

            w = bw * w0
            h = bh * h0

            x = cx * w0 - w / 2
            y = cy * h0 - h / 2

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [
                    x,
                    y,
                    w,
                    h,
                ],
                "area": w * h,
                "iscrowd": 0,
            })

            ann_id += 1

    categories = [
        {
            "id": k + 1,
            "name": v,
        }
        for k, v
        in sorted(
            CLASS_NAMES.items()
        )
    ]

    return {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


target = (
    ENGINE_PATH
    if MODE == "engine"
    else ONNX_PATH
)

print("=" * 60)
print(
    f"mAP evaluation "
    f"(MODE={MODE}, EVAL_SET={EVAL_SET})"
)
print("=" * 60)

print("Target:", target)
print("Images:", IMG_DIR)
print("Labels:", LABEL_DIR)
print(
    f"conf={CONF} "
    f"imgsz={IMG_SIZE} "
    f"max_dets={MAX_DETS}"
)

if MODE == "engine":
    print("DEVICE:", DEVICE)

if not target.exists():
    sys.exit(
        f"Not found: {target}"
    )

frames = sorted(
    IMG_DIR.glob("*.jpg")
)

if not frames:
    sys.exit(
        f"No images in {IMG_DIR}"
    )

gt = build_coco_gt(frames)

print(
    f"\nGround truth: "
    f"{len(gt['images'])} images, "
    f"{len(gt['annotations'])} boxes"
)

predict = (
    EnginePredictor(target)
    if MODE == "engine"
    else OnnxPredictor(target)
)

detections = []

for img_id, fp in enumerate(
    frames,
    start=1,
):

    img = cv2.imread(str(fp))

    h0, w0 = img.shape[:2]

    raw = predict(
        preprocess(
            img,
            IMG_SIZE,
        )
    )[0]


    # ======================================================
    # MODIFY IF MODEL OUTPUT FORMAT CHANGES
    #
    # Current YOLO26n deployment contract:
    #
    # [1, 300, 6]
    #
    # columns:
    # x1, y1, x2, y2, confidence, class
    #
    # NMS is already inside the model.
    # ======================================================

    keep = raw[
        raw[:, 4] >= CONF
    ]

    if len(keep) == 0:
        continue

    boxes = unletterbox(
        keep[:, :4],
        h0,
        w0,
        IMG_SIZE,
    )

    for (
        x1,
        y1,
        x2,
        y2,
    ), score, cls in zip(
        boxes,
        keep[:, 4],
        keep[:, 5],
    ):

        detections.append({
            "image_id": img_id,
            "category_id": int(cls) + 1,
            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ],
            "score": float(score),
        })


print(
    f"Detections: "
    f"{len(detections)} "
    f"above conf {CONF}"
)

if not detections:
    sys.exit(
        "No detections above threshold."
    )


per_image = {}

for d in detections:
    image_id = d["image_id"]

    per_image[image_id] = (
        per_image.get(
            image_id,
            0,
        )
        + 1
    )

busiest = max(
    per_image.values()
)

print(
    f"Max dets on any image: "
    f"{busiest} "
    f"(cap {MAX_DETS})"
)


coco_gt = COCO()
coco_gt.dataset = gt
coco_gt.createIndex()

coco_dt = coco_gt.loadRes(
    detections
)

ev = COCOeval(
    coco_gt,
    coco_dt,
    iouType="bbox",
)

ev.params.maxDets = [
    1,
    10,
    MAX_DETS,
]

ev.evaluate()
ev.accumulate()
ev.summarize()

precision = (
    ev.eval["precision"]
)


def _mean_precision(iou_slice):

    vals = iou_slice[
        iou_slice > -1
    ]

    return (
        float(vals.mean())
        if vals.size
        else float("nan")
    )


map50_95 = _mean_precision(
    precision[
        :,
        :,
        :,
        0,
        -1,
    ]
)

map50 = _mean_precision(
    precision[
        0,
        :,
        :,
        0,
        -1,
    ]
)

print(
    "\n--------------------------------"
)

print(
    f"{'mAP50':<12}"
    f"{map50:>10.4f}"
)

print(
    f"{'mAP50-95':<12}"
    f"{map50_95:>10.4f}"
)

print(
    "--------------------------------"
)


prov = {
    "timestamp_utc":
        datetime.now(
            timezone.utc
        ).isoformat(),

    "mode":
        MODE,

    "target":
        str(target),

    "target_sha256":
        sha256(target),

    "device":
        DEVICE
        if MODE == "engine"
        else "cpu",

    "eval_set":
        EVAL_SET,

    "img_dir":
        str(IMG_DIR),

    "n_images":
        len(gt["images"]),

    "n_gt_boxes":
        len(gt["annotations"]),

    "n_detections":
        len(detections),

    "conf":
        CONF,

    "img_size":
        IMG_SIZE,

    "max_dets":
        MAX_DETS,

    "metric":
        "pycocotools COCOeval bbox",

    "map50":
        map50,

    "map50_95":
        map50_95,

    "coco_stats":
        [
            float(s)
            for s in ev.stats
        ],
}


suffix = (
    ".map.json"
    if EVAL_SET == "val100"
    else f".map_{EVAL_SET}.json"
)

out = target.with_name(
    target.name + suffix
)

out.write_text(
    json.dumps(
        prov,
        indent=2,
    )
)

print(
    "mAP record:",
    out,
)
