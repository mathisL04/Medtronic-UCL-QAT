from ultralytics import YOLO
from pathlib import Path
import torch


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing the dataset, model,
# training hardware, or training hyperparameters.
# ==========================================================

# Dataset configuration.
# Replace with the YAML generated for the new dataset.
DATA_YAML = (
    "/workspace/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo/"
    "sanoscience_yolo.yaml"
)

# Initial model/checkpoint.
# Replace with another Ultralytics model or compatible .pt checkpoint.
# For a clean experiment, start from pretrained weights rather than a
# previously fine-tuned model.
MODEL_NAME = "yolo26n.pt"

# Training output location and experiment name.
PROJECT_DIR = "/workspace/runs_sanoscience"
RUN_NAME = "yolo26n_sanoscience_full_left"

# Main training hyperparameters.
# Adapt these to the model, dataset and available GPU memory.
IMG_SIZE = 640
EPOCHS = 50
BATCH_SIZE = 4
WORKERS = 2

# GPU selection.
# Replace 0 with another CUDA device index if required.
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def main():
    print("=" * 80)
    print("Train YOLO model")
    print("=" * 80)
    print(f"Data yaml: {DATA_YAML}")
    print(f"Model: {MODEL_NAME}")
    print(f"Image size: {IMG_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")

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

        # Additional training parameters.
        # Modify these if changing the training/regularisation strategy.
        patience=10,
        cache=False,
    )

    print("\nTraining complete.")
    print(
        f"Run folder: "
        f"{Path(PROJECT_DIR) / RUN_NAME}"
    )
    print(
        f"Best weights: "
        f"{Path(PROJECT_DIR) / RUN_NAME / 'weights' / 'best.pt'}"
    )


if __name__ == "__main__":
    main()
