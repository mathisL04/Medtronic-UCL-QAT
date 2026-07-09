from ultralytics import YOLO
from pathlib import Path
import torch

DATA_YAML = "/workspace/datasets/utenn_yolo/utenn_yolo.yaml"
MODEL_NAME = "yolo26n.pt"

PROJECT_DIR = "/workspace/runs_utenn"
RUN_NAME = "yolo26n_utenn_grasper_test"

IMG_SIZE = 416
EPOCHS = 5
BATCH_SIZE = 2

DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 80)
    print("Training YOLO26n on UTenn grasper dataset")
    print("=" * 80)

    print(f"Dataset: {DATA_YAML}")
    print(f"Base model: {MODEL_NAME}")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not Path(DATA_YAML).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

    model = YOLO(MODEL_NAME)

    model.train(
        data=DATA_YAML,
        imgsz=IMG_SIZE,
        epochs=EPOCHS,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_DIR,
        name=RUN_NAME,
        workers=2,
        pretrained=True,
        plots=True,
        save=True,
        exist_ok=True,
    )

    print("\nTraining complete.")
    print("Best weights:")
    print(f"{PROJECT_DIR}/{RUN_NAME}/weights/best.pt")
    print("Last weights:")
    print(f"{PROJECT_DIR}/{RUN_NAME}/weights/last.pt")


if __name__ == "__main__":
    main()