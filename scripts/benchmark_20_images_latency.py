from pathlib import Path
import csv
from statistics import mean
from ultralytics import YOLO

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/best.pt"
)

IMAGE_DIR = Path(
    "/home/zcemml1/medtronic_qat_data/demo_val20_yolo/images/val"
)

OUT_CSV = Path(
    "/home/zcemml1/medtronic_qat_data/runs_sanoscience/"
    "latency_demo_20_images.csv"
)

# -----------------------------
# Settings
# -----------------------------
IMG_SIZE = 640
CONF = 0.25
DEVICE = 3

# -----------------------------
# Load model
# -----------------------------
model = YOLO(str(MODEL_PATH))

images = sorted(IMAGE_DIR.glob("*.jpg"))

print(f"Model: {MODEL_PATH}")
print(f"Image directory: {IMAGE_DIR}")
print(f"Number of images: {len(images)}")
print(f"Device: {DEVICE}")

rows = []

# -----------------------------
# Run prediction image by image
# -----------------------------
for img in images:
    result = model.predict(
        source=str(img),
        imgsz=IMG_SIZE,
        conf=CONF,
        device=DEVICE,
        save=False,
        verbose=False,
    )[0]

    preprocess_ms = result.speed["preprocess"]
    inference_ms = result.speed["inference"]
    postprocess_ms = result.speed["postprocess"]

    total_ms = preprocess_ms + inference_ms + postprocess_ms

    num_boxes = len(result.boxes) if result.boxes is not None else 0

    rows.append({
        "image": img.name,
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "total_composed_ms": total_ms,
        "num_boxes": num_boxes,
    })

# -----------------------------
# Save CSV
# -----------------------------
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "image",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "total_composed_ms",
            "num_boxes",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

# -----------------------------
# Print summary
# -----------------------------
avg_pre = mean(r["preprocess_ms"] for r in rows)
avg_inf = mean(r["inference_ms"] for r in rows)
avg_post = mean(r["postprocess_ms"] for r in rows)
avg_total = mean(r["total_composed_ms"] for r in rows)
fps = 1000.0 / avg_total

print("\nPer-image latency CSV saved to:")
print(OUT_CSV)

print("\nAverage latency over 20 images:")
print(f"Preprocess:   {avg_pre:.4f} ms/image")
print(f"Inference:    {avg_inf:.4f} ms/image")
print(f"Postprocess:  {avg_post:.4f} ms/image")
print(f"Total:        {avg_total:.4f} ms/image")
print(f"Approx FPS:   {fps:.2f}")

print("\nFirst 5 rows:")
for r in rows[:5]:
    print(r)
