from pathlib import Path
import torch
from ultralytics import YOLO

# --------------------------------------------------
# Cork / UCL configuration
# --------------------------------------------------
DATA_YAML = "/home/zcemml1/medtronic_qat_data/datasets/sanoscience_yolo_full_nonexpert_stereo/sanoscience_yolo.yaml"

# Clean pretrained YOLO26n weights
MODEL_NAME = "yolo26n.pt"

PROJECT_DIR = "/home/zcemml1/medtronic_qat_data/runs_sanoscience"
RUN_NAME = "yolo26n_sanoscience_full_left"

IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
WORKERS = 8

DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("=" * 80)
print("YOLO26n Sanoscience full training on Cork")
print("=" * 80)
print(f"Dataset YAML: {DATA_YAML}")
print(f"Model: {MODEL_NAME}")
print(f"Project dir: {PROJECT_DIR}")
print(f"Run name: {RUN_NAME}")
print(f"Image size: {IMG_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Workers: {WORKERS}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

assert Path(DATA_YAML).exists(), f"Dataset YAML not found: {DATA_YAML}"

model = YOLO(MODEL_NAME)

results = model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    workers=WORKERS,
    device=DEVICE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    pretrained=True,
    plots=True,
    save=True,
    amp=True,
)

print("=" * 80)
print("Training complete")
print("=" * 80)
print("Best weights:")
print(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
print("Last weights:")
print(f"{PROJECT_DIR}/{RUN_NAME}/weights/last.pt")
