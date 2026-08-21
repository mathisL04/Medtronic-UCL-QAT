import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone


# ==========================================================
# MODIFY FOR ANOTHER MODEL
# ==========================================================

REPO = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT"
)

MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    str(
        REPO
        / "results/models/yolo26n_sanoscience_full_left/"
          "baseline/best.pt"
    )
))


# ==========================================================
# PRECISION TO TEST
#
# fp32 -> reference through exactly the same evaluation path
# fp16 -> full model.half() + FP16 input
# ==========================================================

PRECISION = os.environ.get(
    "PRECISION",
    "fp16"
).lower()

if PRECISION not in ("fp16", "fp32"):
    sys.exit(
        f"PRECISION must be fp16 or fp32 "
        f"(got {PRECISION!r})."
    )


# ==========================================================
# MODIFY FOR ANOTHER DATASET
#
# Each dataset root is assumed to contain:
#   images/val/
#   labels/val/
#
# Labels are currently assumed to be YOLO-format:
#   class cx cy width height
# ==========================================================

EVAL_SET = os.environ.get(
    "EVAL_SET",
    "val100"
).lower()

_EVAL_SETS = {
    "val100": Path(
        "/home/zcemml1/medtronic_qat_data/"
        "demo_val100_random_yolo"
    ),

    "full": Path(
        "/home/zcemml1/medtronic_qat_data/"
        "datasets/"
        "sanoscience_yolo_full_nonexpert_stereo"
    ),
}

if EVAL_SET not in _EVAL_SETS:
    sys.exit(
        f"EVAL_SET must be one of "
        f"{sorted(_EVAL_SETS)} "
        f"(got {EVAL_SET!r})."
    )

_root = _EVAL_SETS[EVAL_SET]

IMG_DIR = _root / "images" / "val"
LABEL_DIR = _root / "labels" / "val"


# ==========================================================
# MODIFY IF MODEL INPUT / OUTPUT CONTRACT CHANGES
# ==========================================================

IMG_SIZE = int(
    os.environ.get("IMG_SIZE", 640)
)

CONF = float(
    os.environ.get("CONF", 0.001)
)

MAX_DETS = int(
    os.environ.get("MAX_DETS", 300)
)

N_PROBE = int(
    os.environ.get("N_PROBE", 4)
)


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

if DEVICE is None:
    sys.exit(
        "DEVICE is required "
        "(e.g. DEVICE=0)."
    )

os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE


import numpy as np
import cv2
import torch

from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
# MODIFY IF PREPROCESSING CHANGES
#
# This must remain identical to training / export /
# TensorRT evaluation preprocessing.
# ==========================================================

def letterbox_params(h0, w0, s):

    r = min(
        s / h0,
        s / w0,
    )

    nw = int(round(w0 * r))
    nh = int(round(h0 * r))

    return (
        r,
        nw,
        nh,
        (s - nw) // 2,
        (s - nh) // 2,
    )


def preprocess(img_bgr, s):

    h0, w0 = img_bgr.shape[:2]

    r, nw, nh, left, top = (
        letterbox_params(
            h0,
            w0,
            s,
        )
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
            dtype=np.float32,
        )[None]
        / 255.0
    )


def unletterbox(
    boxes,
    h0,
    w0,
    s,
):

    r, _, _, left, top = (
        letterbox_params(
            h0,
            w0,
            s,
        )
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


# ==========================================================
# MODIFY IF DATASET LABEL FORMAT CHANGES
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


print("=" * 60)
print(
    f"PyTorch {PRECISION.upper()} mAP "
    "(native, no TensorRT)"
)
print("=" * 60)

print("model:", MODEL_PATH)
print("device:", DEVICE)
print("torch:", torch.__version__)
print(
    f"eval_set={EVAL_SET} "
    f"conf={CONF} "
    f"imgsz={IMG_SIZE} "
    f"max_dets={MAX_DETS}"
)

if not MODEL_PATH.exists():
    sys.exit(
        f"Model not found: {MODEL_PATH}"
    )


# ==========================================================
# MODIFY IF USING A NON-ULTRALYTICS MODEL
#
# Current implementation expects YOLO(...) and a Detect head.
# ==========================================================

y = YOLO(str(MODEL_PATH))

model = (
    y.model
    .cuda()
    .eval()
)

for p in model.parameters():
    p.requires_grad = False


# ==========================================================
# YOLO26-SPECIFIC
#
# Forces deployment-style single output:
#   [1, 300, 6]
#
# Remove / replace this block if the new model has a
# different head or output mechanism.
# ==========================================================

n_heads = 0

for m in model.modules():

    if isinstance(m, Detect):

        m.export = True
        m.format = "onnx"
        m.dynamic = False

        n_heads += 1

print(
    f"[head] export mode on {n_heads} Detect head(s)"
)


def tensor_census(named):

    total_n = 0
    half_n = 0
    other = {}

    for name, t in named:

        if t is None:
            continue

        n = t.numel()
        total_n += n

        if t.dtype == torch.float16:

            half_n += n

        else:

            other.setdefault(
                str(t.dtype),
                0,
            )

            other[str(t.dtype)] += n

    return (
        total_n,
        half_n,
        other,
    )


if PRECISION == "fp16":

    model.half()

    print(
        "[convert] model.half() "
        "applied to entire module"
    )


p_total, p_half, p_other = (
    tensor_census(
        model.named_parameters()
    )
)

b_total, b_half, b_other = (
    tensor_census(
        model.named_buffers()
    )
)


print(
    f"[census] parameters: "
    f"{p_total:,} "
    f"fp16={p_half:,} "
    f"other={p_other or '{}'}"
)

print(
    f"[census] buffers: "
    f"{b_total:,} "
    f"fp16={b_half:,} "
    f"other={b_other or '{}'}"
)


module_out_dtype = {}
detect_out_dtype = {
    "val": None
}


def first_tensor(x):

    if torch.is_tensor(x):
        return x

    if isinstance(
        x,
        (list, tuple),
    ):

        for e in x:

            t = first_tensor(e)

            if t is not None:
                return t

    return None


def make_hook(
    name,
    typ,
    is_detect,
):

    def hook(
        mod,
        inp,
        out,
    ):

        t = first_tensor(out)

        if t is not None:

            module_out_dtype[name] = (
                typ,
                str(t.dtype),
            )

            if is_detect:
                detect_out_dtype["val"] = (
                    str(t.dtype)
                )

    return hook


handles = []

for name, mod in model.named_modules():

    is_leaf = (
        len(list(mod.children()))
        == 0
    )

    is_detect = isinstance(
        mod,
        Detect,
    )

    if is_leaf or is_detect:

        handles.append(
            mod.register_forward_hook(
                make_hook(
                    name,
                    type(mod).__name__,
                    is_detect,
                )
            )
        )


frames = sorted(
    IMG_DIR.glob("*.jpg")
)

if not frames:

    sys.exit(
        f"No images in {IMG_DIR}"
    )


want_dtype = (
    torch.float16
    if PRECISION == "fp16"
    else torch.float32
)


def infer(im_np):

    x = (
        torch.from_numpy(im_np)
        .to(
            "cuda",
            dtype=want_dtype,
        )
    )

    with torch.inference_mode():
        out = model(x)

    pred = (
        out[0]
        if isinstance(
            out,
            (list, tuple),
        )
        else out
    )

    return pred


print(
    f"\n[probe] "
    f"{N_PROBE} frame(s), "
    f"precision={PRECISION}"
)

probe_ok = True
probe_err = None

out_shape = None
out_dtype = None

try:

    for fp in frames[:N_PROBE]:

        img = cv2.imread(str(fp))

        pred = infer(
            preprocess(
                img,
                IMG_SIZE,
            )
        )

        out_shape = tuple(
            pred.shape
        )

        out_dtype = str(
            pred.dtype
        )

except Exception as e:

    probe_ok = False

    probe_err = (
        f"{type(e).__name__}: {e}"
    )


if probe_ok:

    print(
        f"[probe] CLEAN "
        f"shape={out_shape} "
        f"dtype={out_dtype}"
    )

else:

    print(
        f"[probe] THREW: "
        f"{probe_err}"
    )


n_mod = len(
    module_out_dtype
)

fp16_mods = [
    n
    for n, (_, d)
    in module_out_dtype.items()
    if d == "torch.float16"
]

fp32_mods = [
    (n, t)
    for n, (t, d)
    in module_out_dtype.items()
    if d == "torch.float32"
]

other_mods = [
    (n, t, d)
    for n, (t, d)
    in module_out_dtype.items()
    if d not in (
        "torch.float16",
        "torch.float32",
    )
]


print(
    f"[coverage] modules emitting tensors: "
    f"{n_mod}"
)


if PRECISION == "fp16":

    print(
        f"[coverage] fp16 outputs: "
        f"{len(fp16_mods)}"
    )

    print(
        f"[coverage] fp32 islands: "
        f"{len(fp32_mods)}"
    )

    for n, t in fp32_mods:

        print(
            f"  - {n} ({t})"
        )

    if other_mods:

        print(
            f"[coverage] other dtype: "
            f"{len(other_mods)}"
        )

        for n, t, d in other_mods:

            print(
                f"  - {n} ({t}) {d}"
            )

    print(
        "[coverage] Detect output dtype:",
        detect_out_dtype["val"],
    )


for h in handles:
    h.remove()


if not probe_ok:

    sys.exit(
        "Forward failed during dtype probe."
    )


gt = build_coco_gt(
    frames
)

print(
    f"\n[map] ground truth: "
    f"{len(gt['images'])} images, "
    f"{len(gt['annotations'])} boxes"
)


detections = []

for img_id, fp in enumerate(
    frames,
    start=1,
):

    img = cv2.imread(
        str(fp)
    )

    h0, w0 = img.shape[:2]

    pred = infer(
        preprocess(
            img,
            IMG_SIZE,
        )
    )


    # ======================================================
    # MODIFY IF MODEL OUTPUT FORMAT CHANGES
    #
    # Current YOLO26 deployment output:
    #
    # [1, 300, 6]
    #
    # x1, y1, x2, y2, confidence, class
    #
    # NMS is already included.
    # ======================================================

    raw = (
        pred
        .float()
        .cpu()
        .numpy()[0]
    )

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
            "category_id":
                int(cls) + 1,

            "bbox": [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ],

            "score":
                float(score),
        })


print(
    f"[map] detections above "
    f"conf {CONF}: "
    f"{len(detections)}"
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
    f"[map] max dets/image: "
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

    vals = (
        iou_slice[
            iou_slice > -1
        ]
    )

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
    f"{'precision':<12}"
    f"{PRECISION:>10}"
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

    "stage":
        "pytorch_fp16_accuracy",

    "precision":
        PRECISION,

    "model":
        str(MODEL_PATH),

    "model_sha256":
        sha256(MODEL_PATH),

    "torch":
        torch.__version__,

    "device":
        DEVICE,

    "eval_set":
        EVAL_SET,

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

    "params_total":
        p_total,

    "params_fp16":
        p_half,

    "params_other":
        p_other,

    "buffers_total":
        b_total,

    "buffers_fp16":
        b_half,

    "buffers_other":
        b_other,

    "probe_output_shape":
        list(out_shape)
        if out_shape
        else None,

    "probe_output_dtype":
        out_dtype,

    "modules_emitting_tensor":
        n_mod,

    "modules_fp16_output":
        len(fp16_mods),

    "modules_fp32_output":
        [
            n
            for n, _
            in fp32_mods
        ],

    "detect_head_output_dtype":
        detect_out_dtype["val"],

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
    f".pytorch_"
    f"{PRECISION}_"
    f"{EVAL_SET}.json"
)

out = MODEL_PATH.with_name(
    MODEL_PATH.name
    + suffix
)

out.write_text(
    json.dumps(
        prov,
        indent=2,
    )
)

print(
    "record:",
    out,
)
