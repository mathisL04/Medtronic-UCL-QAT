from ultralytics import YOLO
from pathlib import Path
import torch

# --------------------------------------------------
# Configuration
# --------------------------------------------------
DATA_YAML = "/workspace/datasets/sanoscience_yolo_full_nonexpert_stereo/sanoscience_yolo.yaml"

# Important: start from clean pretrained YOLO26n, not UTenn best.pt
MODEL_NAME = "yolo26n.pt"

PROJECT_DIR = "/workspace/runs_sanoscience"
RUN_NAME = "yolo26n_sanoscience_full_left"

IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 4
WORKERS = 2

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 80)
    print("Train YOLO26n on Sanoscience tool sample")
    print("=" * 80)
    print(f"Data yaml: {DATA_YAML}")
    print(f"Model: {MODEL_NAME}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = YOLO(MODEL_NAME)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        plots=True,
        workers=WORKERS,
        patience=10,
        cache=False,
    )

    print("\nTraining complete.")
    print(f"Run folder: {Path(PROJECT_DIR) / RUN_NAME}")
    print(f"Best weights: {Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()