import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import cv2


# -----------------------------
# Settings
# -----------------------------
# MODE selects what is evaluated. The SAME metric code runs in both cases, which
# is the point of this script: the FP32 reference in docs/02 (mAP50 0.9394) was
# produced by Ultralytics' mAP implementation, and Ultralytics is not installed
# in the TensorRT venv. Computing an engine's mAP with pycocotools and comparing
# it to an Ultralytics number would fold a metric-implementation change into what
# is supposed to be a precision delta. So we re-measure the ONNX here too, and
# every precision is compared against THAT pycocotools reference.
#   MODE=onnx    -> best.onnx via onnxruntime (CPU, as in docs/02)
#   MODE=engine  -> a TensorRT .engine via ENGINE_PATH
MODE = os.environ.get("MODE", "engine").lower()

ENGINE_PATH = Path(os.environ.get(
    "ENGINE_PATH",
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/best_fp32.engine",
))
ONNX_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/best.onnx"
)

# Which evaluation set to score against. A NAMED selector, not a free path:
# dataset locations stay hardcoded per repo convention, so this cannot be
# pointed at arbitrary data, but the two legitimate sets can be chosen.
#
#   val100 -- 100 images / 237 boxes. Fast, and what V2/V3/V4 were first
#             measured on. Too small to resolve deltas below ~0.01 mAP.
#   full   -- 6,449 images / 14,517 boxes. The real number. Required whenever a
#             delta is small enough that val100's noise floor could swallow it,
#             which is the case for INT8 (docs/05 OPEN 1).
EVAL_SET = os.environ.get("EVAL_SET", "val100").lower()

_EVAL_SETS = {
    "val100": Path("/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo"),
    "full": Path("/home/zcemml1/medtronic_qat_data/datasets/"
                 "sanoscience_yolo_full_nonexpert_stereo"),
}
if EVAL_SET not in _EVAL_SETS:
    sys.exit(f"EVAL_SET must be one of {sorted(_EVAL_SETS)} (got {EVAL_SET!r}).")

_root = _EVAL_SETS[EVAL_SET]
IMG_DIR = _root / "images" / "val"
LABEL_DIR = _root / "labels" / "val"

IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))

# conf 0.001 is the mAP-measurement threshold (docs/02), NOT a deployment
# threshold. It keeps the low-confidence tail so the full precision-recall curve
# is built; conf 0.25 truncates that curve and gives a different, non-comparable
# number. Do not "clean up" this default.
CONF = float(os.environ.get("CONF", 0.001))

# The model emits a fixed [1, 300, 6] with NMS already applied, so 300 is the
# model's own detection cap. COCOeval defaults to maxDets 100, which would
# discard the tail we deliberately kept -- match the model instead.
MAX_DETS = int(os.environ.get("MAX_DETS", 300))

CLASS_NAMES = {0: "surgical_tool"}

# DEVICE required for engine mode (same discipline as the other TRT scripts).
# ONNX mode runs on CPU, matching how the docs/02 reference was produced, so it
# does not need a GPU at all.
DEVICE = os.environ.get("DEVICE")
if MODE == "engine":
    if DEVICE is None:
        sys.exit("DEVICE is required for MODE=engine (e.g. DEVICE=0). No GPU specified, no run.")
    os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE
elif MODE != "onnx":
    sys.exit(f"MODE must be 'engine' or 'onnx' (got {MODE!r}).")

from pycocotools.coco import COCO          # noqa: E402
from pycocotools.cocoeval import COCOeval  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Preprocess  (identical letterbox to validate_engine_parity.py / export_onnx.py)
# -----------------------------
def letterbox_params(h0, w0, s):
    """Scale + padding that maps an (h0, w0) image into an s x s canvas."""
    r = min(s / h0, s / w0)
    nw, nh = int(round(w0 * r)), int(round(h0 * r))
    return r, nw, nh, (s - nw) // 2, (s - nh) // 2


def preprocess(img_bgr, s):
    h0, w0 = img_bgr.shape[:2]
    r, nw, nh, left, top = letterbox_params(h0, w0, s)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((s, s, 3), 114, dtype=np.uint8)
    canvas[top:top + nh, left:left + nw] = resized
    im = canvas[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(im, dtype=np.float32)[None] / 255.0


def unletterbox(boxes, h0, w0, s):
    """Map xyxy boxes from letterboxed s x s space back to original pixels.

    Ground truth is in original image coordinates, so predictions must be mapped
    back before they can be scored against it -- otherwise every box is offset
    and mAP collapses for a reason that has nothing to do with the model.
    """
    r, _, _, left, top = letterbox_params(h0, w0, s)
    out = boxes.astype(np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / r
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0)
    return out


# -----------------------------
# Predictors
# -----------------------------
class EnginePredictor:
    """Minimal TensorRT runner -- same wrapper as validate_engine_parity.py."""

    def __init__(self, path):
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
        self.trt, self.cudart = trt, cudart
        self.H2D = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        self.D2H = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {path}")
        self.context = self.engine.create_execution_context()

        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
                self.inp_name, self.inp_shape = n, tuple(self.engine.get_tensor_shape(n))
            else:
                self.out_name, self.out_shape = n, tuple(self.engine.get_tensor_shape(n))

        self.out_host = np.empty(self.out_shape, dtype=np.float32)
        self.d_in = self._chk(cudart.cudaMalloc(int(np.prod(self.inp_shape)) * 4))
        self.d_out = self._chk(cudart.cudaMalloc(int(np.prod(self.out_shape)) * 4))
        self.stream = self._chk(cudart.cudaStreamCreate())
        self.context.set_tensor_address(self.inp_name, int(self.d_in))
        self.context.set_tensor_address(self.out_name, int(self.d_out))

    @staticmethod
    def _chk(ret):
        err, rest = (ret[0], ret[1:]) if isinstance(ret, (tuple, list)) else (ret, ())
        if int(err) != 0:
            raise RuntimeError(f"CUDA error code {int(err)}")
        return rest[0] if len(rest) == 1 else (rest or None)

    def __call__(self, im):
        im = np.ascontiguousarray(im, dtype=np.float32)
        self._chk(self.cudart.cudaMemcpyAsync(self.d_in, im.ctypes.data, im.nbytes,
                                              self.H2D, self.stream))
        self.context.execute_async_v3(int(self.stream))
        self._chk(self.cudart.cudaMemcpyAsync(self.out_host.ctypes.data, self.d_out,
                                              self.out_host.nbytes, self.D2H, self.stream))
        self._chk(self.cudart.cudaStreamSynchronize(self.stream))
        return self.out_host.copy()


class OnnxPredictor:
    def __init__(self, path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name

    def __call__(self, im):
        return self.sess.run(None, {self.name: im})[0]


# -----------------------------
# Ground truth: YOLO labels -> COCO
# -----------------------------
def build_coco_gt(frames):
    images, annotations, ann_id = [], [], 1
    for img_id, fp in enumerate(frames, start=1):
        img = cv2.imread(str(fp))
        if img is None:
            sys.exit(f"Could not read image: {fp}")
        h0, w0 = img.shape[:2]
        images.append({"id": img_id, "file_name": fp.name, "width": w0, "height": h0})

        label_fp = LABEL_DIR / (fp.stem + ".txt")
        if not label_fp.exists():
            continue
        for line in label_fp.read_text().strip().splitlines():
            if not line.strip():
                continue
            cls, cx, cy, bw, bh = (float(v) for v in line.split())
            # YOLO stores normalised centre/size; COCO wants absolute x,y,w,h.
            w, h = bw * w0, bh * h0
            x, y = cx * w0 - w / 2, cy * h0 - h / 2
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(cls) + 1,      # COCO category ids are 1-based
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
    categories = [{"id": k + 1, "name": v} for k, v in sorted(CLASS_NAMES.items())]
    return {"images": images, "annotations": annotations, "categories": categories}


# -----------------------------
# Run
# -----------------------------
target = ENGINE_PATH if MODE == "engine" else ONNX_PATH

print("============================================================")
print(f"mAP on val100  (MODE={MODE})")
print("============================================================")
print("Target: ", target)
print("Images: ", IMG_DIR)
print("Labels: ", LABEL_DIR)
print(f"conf:    {CONF}   imgsz: {IMG_SIZE}   max_dets: {MAX_DETS}")
if MODE == "engine":
    print("DEVICE: ", DEVICE)

if not target.exists():
    sys.exit(f"Not found: {target}")

frames = sorted(IMG_DIR.glob("*.jpg"))
if not frames:
    sys.exit(f"No images in {IMG_DIR}")

gt = build_coco_gt(frames)
print(f"\nGround truth: {len(gt['images'])} images, {len(gt['annotations'])} boxes")

predict = EnginePredictor(target) if MODE == "engine" else OnnxPredictor(target)

detections = []
for img_id, fp in enumerate(frames, start=1):
    img = cv2.imread(str(fp))
    h0, w0 = img.shape[:2]
    raw = predict(preprocess(img, IMG_SIZE))[0]      # [300, 6] = x1,y1,x2,y2,conf,cls

    keep = raw[raw[:, 4] >= CONF]
    if len(keep) == 0:
        continue
    boxes = unletterbox(keep[:, :4], h0, w0, IMG_SIZE)
    for (x1, y1, x2, y2), score, cls in zip(boxes, keep[:, 4], keep[:, 5]):
        detections.append({
            "image_id": img_id,
            "category_id": int(cls) + 1,
            "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
            "score": float(score),
        })

print(f"Detections:   {len(detections)} above conf {CONF}")
if not detections:
    sys.exit("No detections above threshold -- nothing to score.")

# Whether MAX_DETS actually bites changes the metric. Report it rather than
# leaving it implicit: if the busiest image is well under the cap, the cap is
# not influencing mAP and the number is comparable across any MAX_DETS setting.
per_image = {}
for d in detections:
    per_image[d["image_id"]] = per_image.get(d["image_id"], 0) + 1
busiest = max(per_image.values())
print(f"Max dets on any image: {busiest}  (cap {MAX_DETS}"
      f"{' -- CAP IS BINDING, mAP depends on MAX_DETS' if busiest >= MAX_DETS else ', not binding'})")

# COCOeval writes to stdout; that output is the authoritative table, we just
# pull the two headline numbers out of stats afterwards.
coco_gt = COCO()
coco_gt.dataset = gt
coco_gt.createIndex()
coco_dt = coco_gt.loadRes(detections)

ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
ev.params.maxDets = [1, 10, MAX_DETS]
print()
ev.evaluate()
ev.accumulate()
ev.summarize()

# Read the headline numbers out of the precision array rather than ev.stats.
# COCOeval._summarize hardcodes maxDets=100 for stats[0], so with any other
# MAX_DETS that lookup matches nothing and silently returns -1 instead of the
# mAP. The precision array is indexed [iouThr, recall, class, areaRng, maxDet];
# areaRng 0 is 'all' and maxDet -1 is our configured cap.
precision = ev.eval["precision"]


def _mean_precision(iou_slice):
    vals = iou_slice[iou_slice > -1]
    return float(vals.mean()) if vals.size else float("nan")


map50_95 = _mean_precision(precision[:, :, :, 0, -1])   # IoU 0.50:0.95
map50 = _mean_precision(precision[0, :, :, 0, -1])      # IoU 0.50 (first thr)

print("\n------------------------------------------------------------")
print(f"{'mAP50':<12}{map50:>10.4f}")
print(f"{'mAP50-95':<12}{map50_95:>10.4f}")
print("------------------------------------------------------------")

# -----------------------------
# Provenance  (sha-linked, same discipline as the parity/latency records)
# -----------------------------
prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "mode": MODE,
    "target": str(target),
    "target_sha256": sha256(target),
    "device": DEVICE if MODE == "engine" else "cpu",
    "eval_set": EVAL_SET,
    "img_dir": str(IMG_DIR),
    "n_images": len(gt["images"]),
    "n_gt_boxes": len(gt["annotations"]),
    "n_detections": len(detections),
    "conf": CONF,
    "img_size": IMG_SIZE,
    "max_dets": MAX_DETS,
    "metric": "pycocotools COCOeval bbox",
    "map50": map50,
    "map50_95": map50_95,
    "coco_stats": [float(s) for s in ev.stats],
}
# Record name carries the eval set. Without this a full-val run would silently
# overwrite the val100 result under the same filename -- the same defect class as
# the benchmark writing every precision to a hardcoded "fp32" name. val100 stays
# ".map.json" so existing records and their references remain valid.
suffix = ".map.json" if EVAL_SET == "val100" else f".map_{EVAL_SET}.json"
out = target.with_name(target.name + suffix)
out.write_text(json.dumps(prov, indent=2))
print("mAP record:", out)
