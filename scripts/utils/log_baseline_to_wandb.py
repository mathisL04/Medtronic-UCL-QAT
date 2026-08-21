from pathlib import Path
import pandas as pd
import wandb

REPO = Path("/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT")
RUNS = Path("/home/zcemml1/medtronic_qat_data/runs_sanoscience")

MODEL = REPO / "results/models/yolo26n_sanoscience_full_left/0_baseline_pytorch/best.pt"
MODEL_CARD = REPO / "results/models/yolo26n_sanoscience_full_left/model_card.md"

REPORTS = REPO / "results/reports/1_training"
RESULTS_CSV = REPORTS / "results.csv"
RESULTS_PNG = REPORTS / "results.png"
REPORT_CONFUSION = REPORTS / "confusion_matrix.png"

VAL_RESULTS = RUNS / "val_yolo26n_sanoscience_full_left"
PRED_IMAGES = RUNS / "Validation_test_on_trained_model"

run = wandb.init(
    entity="mathislaurent04-ucl",
    project="medtronic-yolo26n-qat",
    name="fp32_pytorch_baseline_yolo26n_sanoscience",
    job_type="baseline-visualisation",
    config={
        "model": "yolo26n_sanoscience_full_left",
        "base_model": "yolo26n.pt",
        "precision": "FP32 PyTorch",
        "task": "surgical_tool_detection",
        "class": "surgical_tool",
        "train_images": 20756,
        "val_images": 6449,
        "val_instances": 14517,
        "imgsz": 640,
        "device": "NVIDIA A100-SXM4-80GB",
    },
)

wandb.log({
    "metrics/precision": 0.938,
    "metrics/recall": 0.864,
    "metrics/mAP50": 0.934,
    "metrics/mAP50_95": 0.782,
    "speed/preprocess_ms": 0.5,
    "speed/inference_ms": 0.7,
    "speed/postprocess_ms": 0.1,
    "speed/end_to_end_ms": 1.3,
})

if RESULTS_CSV.exists():
    df = pd.read_csv(RESULTS_CSV)
    wandb.log({"training/results_csv": wandb.Table(dataframe=df)})

images = {}

for key, path in {
    "training/results_plot": RESULTS_PNG,
    "training/confusion_matrix": REPORT_CONFUSION,
    "validation/confusion_matrix": VAL_RESULTS / "confusion_matrix.png",
    "validation/confusion_matrix_normalized": VAL_RESULTS / "confusion_matrix_normalized.png",
    "validation/PR_curve": VAL_RESULTS / "BoxPR_curve.png",
    "validation/F1_curve": VAL_RESULTS / "BoxF1_curve.png",
    "validation/P_curve": VAL_RESULTS / "BoxP_curve.png",
    "validation/R_curve": VAL_RESULTS / "BoxR_curve.png",
    "validation/batch0_predictions": VAL_RESULTS / "val_batch0_pred.jpg",
    "validation/batch0_labels": VAL_RESULTS / "val_batch0_labels.jpg",
    "validation/batch1_predictions": VAL_RESULTS / "val_batch1_pred.jpg",
    "validation/batch1_labels": VAL_RESULTS / "val_batch1_labels.jpg",
}.items():
    if path.exists():
        images[key] = wandb.Image(str(path))

if images:
    wandb.log(images)

if PRED_IMAGES.exists():
    sample_images = sorted(PRED_IMAGES.glob("*.jpg"))[:100]
    table = wandb.Table(columns=["image_name", "prediction_image"])

    for img in sample_images:
        table.add_data(img.name, wandb.Image(str(img)))

    wandb.log({"validation/prediction_samples": table})

artifact = wandb.Artifact(
    name="yolo26n_sanoscience_full_left_fp32",
    type="model",
    description="FP32 PyTorch YOLO26n trained on Sanoscience surgical-tool detection.",
)

artifact.add_file(str(MODEL))

if MODEL_CARD.exists():
    artifact.add_file(str(MODEL_CARD))

run.log_artifact(artifact)

run.finish()

