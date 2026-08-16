import os
import sys
import json
import hashlib
from pathlib import Path
from datetime import datetime, timezone


# -----------------------------
# Settings
# -----------------------------
# Native-PyTorch FP16 accuracy check for the YOLO26n baseline. This is NOT a
# TensorRT run: it loads best.pt, converts the WHOLE model to FP16 (model.half())
# and feeds .half() input, then scores mAP with the SAME pycocotools protocol as
# evaluate_engine_map.py so the number is directly comparable to the engine
# figures. The model file is opened read-only; all dtype surgery happens in
# memory on the loaded module.
#
# Two things are measured in one pass:
#   1. COVERAGE -- how much of the model actually runs in FP16. A parameter/buffer
#      dtype census plus a per-module output-dtype hook census locates any op that
#      stayed (or silently upcast to) FP32 -- an "FP32 island".
#   2. ACCURACY -- full-val mAP under pycocotools, to compare FP16 vs FP32.
#
# PRECISION=fp32 runs the identical harness without half(), which is the honest
# reference: comparing FP16 against THIS (same code path) isolates the dtype
# effect, rather than folding in PyTorch-vs-TensorRT differences by comparing to
# the engine's 0.7747.
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
MODEL_PATH = Path(os.environ.get(
    "MODEL_PATH",
    str(REPO / "models/yolo26n_sanoscience_full_left/baseline/best.pt")))

PRECISION = os.environ.get("PRECISION", "fp16").lower()
if PRECISION not in ("fp16", "fp32"):
    sys.exit(f"PRECISION must be fp16 or fp32 (got {PRECISION!r}).")

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
CONF = float(os.environ.get("CONF", 0.001))    # mAP threshold, keep the tail
MAX_DETS = int(os.environ.get("MAX_DETS", 300))  # model's own [1,300,6] cap
N_PROBE = int(os.environ.get("N_PROBE", 4))    # frames for the coverage probe
CLASS_NAMES = {0: "surgical_tool"}

DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=0). No GPU specified, no run.")
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE

import numpy as np                             # noqa: E402
import cv2                                     # noqa: E402
import torch                                   # noqa: E402
from ultralytics import YOLO                   # noqa: E402
from ultralytics.nn.modules.head import Detect  # noqa: E402
from pycocotools.coco import COCO              # noqa: E402
from pycocotools.cocoeval import COCOeval      # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Preprocess / postprocess  (identical to evaluate_engine_map.py)
# -----------------------------
def letterbox_params(h0, w0, s):
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
    r, _, _, left, top = letterbox_params(h0, w0, s)
    out = boxes.astype(np.float64).copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - left) / r
    out[:, [1, 3]] = (out[:, [1, 3]] - top) / r
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0)
    return out


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
            w, h = bw * w0, bh * h0
            x, y = cx * w0 - w / 2, cy * h0 - h / 2
            annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": int(cls) + 1,
                "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
            })
            ann_id += 1
    categories = [{"id": k + 1, "name": v} for k, v in sorted(CLASS_NAMES.items())]
    return {"images": images, "annotations": annotations, "categories": categories}


# -----------------------------
# Load model, put Detect head into export mode (single [1,300,6] output)
# -----------------------------
print("============================================================")
print(f"PyTorch {PRECISION.upper()} mAP  (native, no TensorRT)")
print("============================================================")
print("model:  ", MODEL_PATH)
print("device: ", DEVICE, " torch:", torch.__version__)
print(f"eval_set: {EVAL_SET}   conf: {CONF}   imgsz: {IMG_SIZE}   max_dets: {MAX_DETS}")

if not MODEL_PATH.exists():
    sys.exit(f"Model not found: {MODEL_PATH}")

y = YOLO(str(MODEL_PATH))
model = y.model.cuda().eval()
for p in model.parameters():
    p.requires_grad = False
n_heads = 0
for m in model.modules():
    if isinstance(m, Detect):
        m.export = True
        m.format = "onnx"
        m.dynamic = False
        n_heads += 1
print(f"[head] export mode on {n_heads} Detect head(s) -> single deployment output")


# -----------------------------
# Parameter / buffer dtype census (before and after half)
# -----------------------------
def tensor_census(named):
    total_n, half_n, other = 0, 0, {}
    for name, t in named:
        if t is None:
            continue
        n = t.numel()
        total_n += n
        if t.dtype == torch.float16:
            half_n += n
        else:
            other.setdefault(str(t.dtype), 0)
            other[str(t.dtype)] += n
    return total_n, half_n, other


# -----------------------------
# Convert to FP16 (the whole model) if requested
# -----------------------------
if PRECISION == "fp16":
    model.half()
    print("[convert] model.half() applied to the entire module")

p_total, p_half, p_other = tensor_census(model.named_parameters())
b_total, b_half, b_other = tensor_census(model.named_buffers())
print(f"[census] parameters: {p_total:,}  fp16={p_half:,}  other={p_other or '{}'}")
print(f"[census] buffers:    {b_total:,}  fp16={b_half:,}  other={b_other or '{}'}")


# -----------------------------
# Per-module output-dtype hooks -- locate any FP32 island in the forward graph
# -----------------------------
module_out_dtype = {}          # name -> (type, dtype-str)  (leaf modules only)
detect_out_dtype = {"val": None}


def first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for e in x:
            t = first_tensor(e)
            if t is not None:
                return t
    return None


def make_hook(name, typ, is_detect):
    def hook(mod, inp, out):
        t = first_tensor(out)
        if t is not None:
            module_out_dtype[name] = (typ, str(t.dtype))
            if is_detect:
                detect_out_dtype["val"] = str(t.dtype)
    return hook


handles = []
for name, mod in model.named_modules():
    is_leaf = len(list(mod.children())) == 0
    is_detect = isinstance(mod, Detect)
    if is_leaf or is_detect:
        handles.append(mod.register_forward_hook(make_hook(name, type(mod).__name__, is_detect)))


# -----------------------------
# Coverage probe: run a few frames, report clean/throws + output dtype
# -----------------------------
frames = sorted(IMG_DIR.glob("*.jpg"))
if not frames:
    sys.exit(f"No images in {IMG_DIR}")
dev = "cuda"
want_dtype = torch.float16 if PRECISION == "fp16" else torch.float32


def infer(im_np):
    x = torch.from_numpy(im_np).to(dev, dtype=want_dtype)
    with torch.inference_mode():
        out = model(x)
    pred = out[0] if isinstance(out, (list, tuple)) else out
    return pred


print(f"\n[probe] forward on {N_PROBE} val frame(s) in {PRECISION} ...")
probe_ok, probe_err = True, None
out_shape, out_dtype = None, None
try:
    for fp in frames[:N_PROBE]:
        img = cv2.imread(str(fp))
        pred = infer(preprocess(img, IMG_SIZE))
        out_shape = tuple(pred.shape)
        out_dtype = str(pred.dtype)
except Exception as e:
    probe_ok, probe_err = False, f"{type(e).__name__}: {e}"

if probe_ok:
    print(f"[probe] CLEAN. model output shape={out_shape} dtype={out_dtype}")
else:
    print(f"[probe] THREW: {probe_err}")

# Classify module output dtypes gathered during the probe.
n_mod = len(module_out_dtype)
fp16_mods = [n for n, (_, d) in module_out_dtype.items() if d == "torch.float16"]
fp32_mods = [(n, t) for n, (t, d) in module_out_dtype.items() if d == "torch.float32"]
other_mods = [(n, t, d) for n, (t, d) in module_out_dtype.items()
              if d not in ("torch.float16", "torch.float32")]

print(f"\n[coverage] modules that emitted a tensor during probe: {n_mod}")
if PRECISION == "fp16":
    print(f"[coverage]   fp16 output: {len(fp16_mods)}")
    print(f"[coverage]   fp32 output (FP32 islands): {len(fp32_mods)}")
    for n, t in fp32_mods:
        print(f"[coverage]       - {n}  ({t})")
    if other_mods:
        print(f"[coverage]   other dtype: {len(other_mods)}")
        for n, t, d in other_mods:
            print(f"[coverage]       - {n}  ({t}) {d}")
    print(f"[coverage]   Detect head final output dtype: {detect_out_dtype['val']}")

for h in handles:
    h.remove()

if not probe_ok:
    sys.exit("[probe] forward failed -- fix the dtype/conversion before scoring mAP.")

# -----------------------------
# Full mAP
# -----------------------------
gt = build_coco_gt(frames)
print(f"\n[map] ground truth: {len(gt['images'])} images, {len(gt['annotations'])} boxes")

detections = []
for img_id, fp in enumerate(frames, start=1):
    img = cv2.imread(str(fp))
    h0, w0 = img.shape[:2]
    pred = infer(preprocess(img, IMG_SIZE))
    raw = pred.float().cpu().numpy()[0]              # [300, 6] = x1,y1,x2,y2,conf,cls
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

print(f"[map] detections above conf {CONF}: {len(detections)}")
if not detections:
    sys.exit("No detections above threshold -- nothing to score.")

per_image = {}
for d in detections:
    per_image[d["image_id"]] = per_image.get(d["image_id"], 0) + 1
busiest = max(per_image.values())
print(f"[map] max dets on any image: {busiest}  (cap {MAX_DETS}"
      f"{' -- CAP BINDING' if busiest >= MAX_DETS else ', not binding'})")

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

precision = ev.eval["precision"]


def _mean_precision(iou_slice):
    vals = iou_slice[iou_slice > -1]
    return float(vals.mean()) if vals.size else float("nan")


map50_95 = _mean_precision(precision[:, :, :, 0, -1])
map50 = _mean_precision(precision[0, :, :, 0, -1])

print("\n------------------------------------------------------------")
print(f"{'precision':<12}{PRECISION:>10}")
print(f"{'mAP50':<12}{map50:>10.4f}")
print(f"{'mAP50-95':<12}{map50_95:>10.4f}")
print("------------------------------------------------------------")

prov = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "stage": "pytorch_fp16_accuracy",
    "precision": PRECISION,
    "model": str(MODEL_PATH),
    "model_sha256": sha256(MODEL_PATH),
    "torch": torch.__version__,
    "device": DEVICE,
    "eval_set": EVAL_SET,
    "n_images": len(gt["images"]),
    "n_gt_boxes": len(gt["annotations"]),
    "n_detections": len(detections),
    "conf": CONF, "img_size": IMG_SIZE, "max_dets": MAX_DETS,
    "params_total": p_total, "params_fp16": p_half, "params_other": p_other,
    "buffers_total": b_total, "buffers_fp16": b_half, "buffers_other": b_other,
    "probe_output_shape": list(out_shape) if out_shape else None,
    "probe_output_dtype": out_dtype,
    "modules_emitting_tensor": n_mod,
    "modules_fp16_output": len(fp16_mods),
    "modules_fp32_output": [n for n, _ in fp32_mods],
    "detect_head_output_dtype": detect_out_dtype["val"],
    "metric": "pycocotools COCOeval bbox",
    "map50": map50, "map50_95": map50_95,
    "coco_stats": [float(s) for s in ev.stats],
}
suffix = f".pytorch_{PRECISION}_{EVAL_SET}.json"
out = MODEL_PATH.with_name(MODEL_PATH.name + suffix)
out.write_text(json.dumps(prov, indent=2))
print("record:", out)
