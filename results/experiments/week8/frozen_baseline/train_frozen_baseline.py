"""Week 8 — New QAT test framework 2: FROZEN-baseline training.

Step 1 of the plan: take COCO-pretrained yolo26n.pt, FREEZE backbone+neck
(model.0-22), fine-tune ONLY the detection head (model.23) on the surgical
dataset (1 class: surgical_tool). This is the detection analog of "freeze all
weights, train only the last layer". Then (elsewhere) export -> TensorRT ->
measure, and apply QAT on top for comparison.

Same data/split as usual: 20,756 train / 6,449 val, split by episode.
Runs on a Geneva A100 (DEVICE required — no default, same discipline as benchmarks).

Config is by env knobs (repo convention); structural paths hardcoded.
  DEVICE (required), EPOCHS(100 ceiling), PATIENCE(20), BATCH(16), IMG_SIZE(640),
  WORKERS(0 -> fork/overcommit guard), LR0(0.01), RUN_NAME(week8_frozen_head)
"""
import os, sys
from pathlib import Path

# -----------------------------
# Settings
# -----------------------------
REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
DATA_YAML = "/home/zcemml1/medtronic_qat_data/datasets/sanoscience_yolo_full_nonexpert_stereo/sanoscience_yolo.yaml"
MODEL_NAME = str(REPO / "yolo26n.pt")          # COCO-pretrained init (NOT our best.pt)
FREEZE = 23                                     # freeze model.0-22 (backbone+neck), train only head (23)

PROJECT_DIR = str(REPO / "runs_week8")
RUN_NAME = os.environ.get("RUN_NAME", "week8_frozen_head")
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
EPOCHS   = int(os.environ.get("EPOCHS", 100))   # ceiling; early-stopping decides
PATIENCE = int(os.environ.get("PATIENCE", 20))
BATCH    = int(os.environ.get("BATCH", 16))
WORKERS  = int(os.environ.get("WORKERS", 0))    # 0 -> immune to fork-OOM / shm exhaustion on this host
CACHE    = os.environ.get("CACHE", "ram")       # 'ram'/'disk'/'' — decode once, not every epoch.
                                                # On Geneva WORKERS>0 fork-OOMs (vm.overcommit_memory=2),
                                                # so caching (thread-pool preload) is how we avoid GPU
                                                # data-starvation instead of dataloader workers.
LR0      = float(os.environ.get("LR0", 0.01))

DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    sys.exit("DEVICE is required (e.g. DEVICE=0). No GPU specified, no run.")

import torch
from ultralytics import YOLO

print("=" * 70)
print("Week 8 — FROZEN-baseline training (backbone+neck frozen, head trains)")
print("=" * 70)
print(f"init:   {MODEL_NAME}  (COCO-pretrained)")
print(f"freeze: {FREEZE}  (model.0-{FREEZE-1} frozen; model.{FREEZE} Detect head trains)")
print(f"data:   {DATA_YAML}  (1 class: surgical_tool)")
print(f"device: {DEVICE}  epochs<={EPOCHS} patience={PATIENCE} batch={BATCH} workers={WORKERS} lr0={LR0}")
if torch.cuda.is_available():
    print(f"GPU:    {torch.cuda.get_device_name(0)}")

model = YOLO(MODEL_NAME)
results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    patience=PATIENCE,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=int(DEVICE),
    project=PROJECT_DIR,
    name=RUN_NAME,
    exist_ok=True,
    pretrained=True,
    freeze=FREEZE,            # <-- the frozen-baseline rule
    lr0=LR0,
    workers=WORKERS,
    cache=(CACHE if CACHE else False),   # RAM/disk cache -> feed the GPU without fork-based workers
    plots=True,
)
print("\nDONE. best.pt at:", Path(PROJECT_DIR)/RUN_NAME/"weights"/"best.pt")
