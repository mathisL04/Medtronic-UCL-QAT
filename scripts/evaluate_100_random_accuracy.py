from pathlib import Path
from ultralytics import YOLO
import os

# -----------------------------
# Settings
# -----------------------------
DEVICE = int(os.environ.get("DEVICE", 3))

MODEL_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/baseline/best.pt"
)

DATA_YAML = Path(
    "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/"
    "sanoscience_yolo_val100_random.yaml"
)

IMG_SIZE = 640

# Two thresholds, because the metrics want different ones and mixing them up
# silently changes the answer:
#
#   MAP_CONF  = 0.001  the Ultralytics/COCO mAP protocol. Keeps the
#                      low-confidence tail so the full precision-recall curve is
#                      built. This is the ONLY threshold at which mAP is
#                      comparable to docs/02 and to the engine numbers from
#                      evaluate_engine_map.py.
#   OPER_CONF = 0.25   the deployment threshold. Precision/Recall are
#                      threshold-dependent single points on that curve, so they
#                      are only meaningful at the value you would actually ship.
#
# This script previously ran model.val() at 0.25 and reported the result as
# mAP -- which truncates the curve and understates it by ~0.05 mAP50
# (0.8915 vs 0.9408 on this subset). Do not "simplify" these back into one knob.
MAP_CONF = float(os.environ.get("MAP_CONF", 0.001))
OPER_CONF = float(os.environ.get("OPER_CONF", 0.25))

# -----------------------------
# Run validation
# -----------------------------
print("Accuracy evaluation on 100 random validation images")
print("Model:", MODEL_PATH)
print("Data:", DATA_YAML)
print("GPU device:", DEVICE)
print("Image size:", IMG_SIZE)
print(f"mAP conf: {MAP_CONF}   operating conf: {OPER_CONF}")

model = YOLO(str(MODEL_PATH))


def run(conf):
    return model.val(
        data=str(DATA_YAML),
        imgsz=IMG_SIZE,
        conf=conf,
        device=DEVICE,
        verbose=False,
        plots=False,
        save=False,
    )


map_metrics = run(MAP_CONF)
oper_metrics = run(OPER_CONF)

# -----------------------------
# Print clean table
# -----------------------------
print("\nAccuracy summary")
print(f"{'Metric':<15} {'Value':>10}   {'measured at conf':>16}")
print("-" * 48)
print(f"{'mAP50':<15} {map_metrics.box.map50:>10.4f}   {MAP_CONF:>16}")
print(f"{'mAP50-95':<15} {map_metrics.box.map:>10.4f}   {MAP_CONF:>16}")
print(f"{'Precision':<15} {oper_metrics.box.mp:>10.4f}   {OPER_CONF:>16}")
print(f"{'Recall':<15} {oper_metrics.box.mr:>10.4f}   {OPER_CONF:>16}")
print("-" * 48)
print(f"For reference, mAP truncated at the operating threshold "
      f"(NOT the mAP protocol):")
print(f"{'mAP50':<15} {oper_metrics.box.map50:>10.4f}   {OPER_CONF:>16}")
print(f"{'mAP50-95':<15} {oper_metrics.box.map:>10.4f}   {OPER_CONF:>16}")

print("\nDone.")
