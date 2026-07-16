from pathlib import Path
from ultralytics import YOLO
import os

# -----------------------------
# Settings
# -----------------------------
DEVICE = int(os.environ.get("DEVICE", 3))

MODEL_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/best.pt"
)

DATA_YAML = Path(
    "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/"
    "sanoscience_yolo_val100_random.yaml"
)

IMG_SIZE = 640
CONF = 0.25

# -----------------------------
# Run validation
# -----------------------------
print("Accuracy evaluation on 100 random validation images")
print("Model:", MODEL_PATH)
print("Data:", DATA_YAML)
print("GPU device:", DEVICE)
print("Image size:", IMG_SIZE)
print("Confidence:", CONF)

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),
    imgsz=IMG_SIZE,
    conf=CONF,
    device=DEVICE,
    verbose=False,
    plots=False,
    save=False,
)

# -----------------------------
# Extract metrics
# -----------------------------
precision = metrics.box.mp
recall = metrics.box.mr
map50 = metrics.box.map50
map50_95 = metrics.box.map

# -----------------------------
# Print clean table
# -----------------------------
print("\nAccuracy summary")
print(f"{'Metric':<15} {'Value':>10}")
print("-" * 28)
print(f"{'Precision':<15} {precision:>10.4f}")
print(f"{'Recall':<15} {recall:>10.4f}")
print(f"{'mAP50':<15} {map50:>10.4f}")
print(f"{'mAP50-95':<15} {map50_95:>10.4f}")

print("\nDone.")
