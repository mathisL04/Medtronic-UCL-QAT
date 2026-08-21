from pathlib import Path
import csv
import wandb

RESULTS_CSV = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "results/reports/1_training/results.csv"
)

run = wandb.init(
    entity="mathislaurent04-ucl",
    project="medtronic-yolo26n-qat",
    name="fp32_pytorch_baseline_training_history",
    job_type="training-history-visualisation",
    config={
        "model": "yolo26n_sanoscience_full_left",
        "precision": "FP32 PyTorch",
        "source": "Ultralytics results.csv",
        "note": "Historical training curves imported after training.",
    },
)

with open(RESULTS_CSV, "r") as f:
    reader = csv.DictReader(f)

    for i, row in enumerate(reader):
        log_data = {}

        for key, value in row.items():
            clean_key = key.strip()

            try:
                log_data[clean_key] = float(value)
            except ValueError:
                pass

        # Use epoch if present, otherwise use row number
        epoch = int(log_data.get("epoch", i))

        wandb.log(log_data, step=epoch)

run.finish()
