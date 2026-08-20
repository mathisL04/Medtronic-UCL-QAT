from pathlib import Path
import torch
from ultralytics import YOLO


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing the dataset, model,
# compute environment, or training hyperparameters.
# ==========================================================

# Dataset configuration.
# Replace with the YAML generated for the new dataset.
# This path is specific to the Cork/UCL environment.
DATA_YAML = (
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo/"
    "sanoscience_yolo.yaml"
)

# Initial model/checkpoint.
# Replace with another Ultralytics model or compatible .pt checkpoint.
MODEL_NAME = "yolo26n.pt"

# Training output location and experiment name.
# PROJECT_DIR is environment-specific.
PROJECT_DIR = (
    "/home/zcemml1/medtronic_qat_data/"
    "runs_sanoscience"
)
RUN_NAME = "yolo26n_sanoscience_full_left"

# Main training hyperparameters.
# Adapt batch size and workers to the available compute resources.
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 16
WORKERS = 8

# GPU selection.
# Replace 0 with another CUDA device index if required.
DEVICE = 0 if torch.cuda.is_available() else "cpu"


print("=" * 80)
print("YOLO training on Cork")
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
    print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

assert Path(DATA_YAML).exists(), (
    f"Dataset YAML not found: {DATA_YAML}"
)

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

    # Training precision.
    # Disable if AMP is unsupported or a full-FP32 training run is required.
    amp=True,
)

print("=" * 80)
print("Training complete")
print("=" * 80)

print("Best weights:")
print(
    f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt"
)

print("Last weights:")
print(
    f"{PROJECT_DIR}/{RUN_NAME}/weights/last.pt"
)
