from pathlib import Path
from statistics import mean, stdev
import csv
import gc
import os
import socket

import cv2
import torch
from ultralytics import YOLO


# -----------------------------
# Settings
# -----------------------------
DEVICE = int(os.environ.get("DEVICE", 0))
SEED = int(os.environ.get("SEED", 42))
IMG_SIZE = int(os.environ.get("IMG_SIZE", 640))
CONF = float(os.environ.get("CONF", 0.25))
BENCHMARK_REPEATS = int(os.environ.get("BENCHMARK_REPEATS", 5))
WARMUP_IMAGES = int(os.environ.get("WARMUP_IMAGES", 30))

MODEL_PATH = Path(
    "/home/zcemml1/medtronic_qat/Medtronics-UCL-QAT/"
    "models/yolo26n_sanoscience_full_left/best.pt"
)

IMG_DIR = Path(
    "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/images/val"
)

OUT_DIR = Path("/home/zcemml1/medtronic_qat_data/runs_sanoscience")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HOST = socket.gethostname()


def clean_runtime():
    """
    Clean Python and PyTorch/CUDA state for this process.
    This does not clear other users' GPU memory, but it clears our benchmark process.
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()


# -----------------------------
# Runtime setup
# -----------------------------
torch.set_num_threads(1)

if torch.cuda.is_available():
    torch.cuda.set_device(DEVICE)

torch.backends.cudnn.benchmark = True

clean_runtime()


print("============================================================")
print("Stable FP32 Geneva RAM-preloaded latency benchmark")
print("============================================================")
print("Host:", HOST)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("PyTorch visible device:", DEVICE)

if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(DEVICE))

print("Model:", MODEL_PATH)
print("Image directory:", IMG_DIR)
print("Image size:", IMG_SIZE)
print("Confidence:", CONF)
print("Repeats:", BENCHMARK_REPEATS)
print("Warmup images:", WARMUP_IMAGES)


# -----------------------------
# Load image paths
# -----------------------------
image_paths = sorted(IMG_DIR.glob("*.jpg"))

if not image_paths:
    raise ValueError(f"No images found in {IMG_DIR}")

print("Benchmark images:", len(image_paths))


# -----------------------------
# Preload images into RAM
# -----------------------------
print("\nPreloading images into RAM...")

ram_images = []

for path in image_paths:
    img = cv2.imread(str(path))

    if img is None:
        raise ValueError(f"Could not read image: {path}")

    ram_images.append((path.name, img))

print(f"Loaded {len(ram_images)} images into RAM.")
print("Disk reading is now outside the timed benchmark loop.")


# -----------------------------
# Load model once
# -----------------------------
print("\nLoading model once...")

model = YOLO(str(MODEL_PATH))
model.model.eval()

clean_runtime()

print("Model loaded.")


# -----------------------------
# Warmup
# -----------------------------
print("\nWarmup...")

warmup_list = ram_images[: min(WARMUP_IMAGES, len(ram_images))]

with torch.inference_mode():
    for _, img in warmup_list:
        _ = model.predict(
            source=img,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            save=False,
            verbose=False,
        )

if torch.cuda.is_available():
    torch.cuda.synchronize()

print("Warmup complete.")


# -----------------------------
# Benchmark repeats
# -----------------------------
all_run_summaries = []

for run_idx in range(1, BENCHMARK_REPEATS + 1):
    print(f"\nRun {run_idx}/{BENCHMARK_REPEATS}")

    clean_runtime()

    rows = []

    with torch.inference_mode():
        for image_name, img in ram_images:
            result = model.predict(
                source=img,
                imgsz=IMG_SIZE,
                conf=CONF,
                device=DEVICE,
                save=False,
                verbose=False,
            )[0]

            pre = result.speed["preprocess"]
            inf = result.speed["inference"]
            post = result.speed["postprocess"]
            total = pre + inf + post
            boxes = len(result.boxes) if result.boxes is not None else 0

            rows.append({
                "run": run_idx,
                "image": image_name,
                "preprocess_ms": pre,
                "inference_ms": inf,
                "postprocess_ms": post,
                "total_ms": total,
                "num_boxes": boxes,
            })

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def stage_stats(key):
        vals = [r[key] for r in rows]
        return mean(vals), stdev(vals), min(vals), max(vals)

    pre_m, pre_s, pre_min, pre_max = stage_stats("preprocess_ms")
    inf_m, inf_s, inf_min, inf_max = stage_stats("inference_ms")
    post_m, post_s, post_min, post_max = stage_stats("postprocess_ms")
    total_m, total_s, total_min, total_max = stage_stats("total_ms")
    fps = 1000.0 / total_m

    all_run_summaries.append({
        "run": run_idx,
        "preprocess_mean_ms": pre_m,
        "inference_mean_ms": inf_m,
        "postprocess_mean_ms": post_m,
        "total_mean_ms": total_m,
        "fps": fps,
    })

    out_csv = OUT_DIR / f"fp32_geneva_ram_latency_seed{SEED}_run{run_idx}.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved per-image CSV: {out_csv}")
    print()
    print(f"{'Stage':<15} {'Mean ms':>10} {'Std ms':>10} {'Min ms':>10} {'Max ms':>10}")
    print("-" * 60)
    print(f"{'Preprocess':<15} {pre_m:>10.3f} {pre_s:>10.3f} {pre_min:>10.3f} {pre_max:>10.3f}")
    print(f"{'Inference':<15} {inf_m:>10.3f} {inf_s:>10.3f} {inf_min:>10.3f} {inf_max:>10.3f}")
    print(f"{'Postprocess':<15} {post_m:>10.3f} {post_s:>10.3f} {post_min:>10.3f} {post_max:>10.3f}")
    print(f"{'Total':<15} {total_m:>10.3f} {total_s:>10.3f} {total_min:>10.3f} {total_max:>10.3f}")
    print("-" * 60)
    print(f"Approx FPS: {fps:.2f}")


# -----------------------------
# Final summary across repeats
# -----------------------------
summary_csv = OUT_DIR / f"fp32_geneva_ram_latency_seed{SEED}_summary.csv"

with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_run_summaries[0].keys())
    writer.writeheader()
    writer.writerows(all_run_summaries)

def repeat_stats(key):
    vals = [r[key] for r in all_run_summaries]
    return mean(vals), stdev(vals), min(vals), max(vals)

print("\n============================================================")
print("Final repeated-run summary")
print("============================================================")
print("CSV summary:", summary_csv)
print()
print(f"{'Metric':<25} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
print("-" * 70)

for key, label in [
    ("preprocess_mean_ms", "Preprocess mean ms"),
    ("inference_mean_ms", "Inference mean ms"),
    ("postprocess_mean_ms", "Postprocess mean ms"),
    ("total_mean_ms", "Total mean ms"),
    ("fps", "FPS"),
]:
    m, s, mn, mx = repeat_stats(key)
    print(f"{label:<25} {m:>10.3f} {s:>10.3f} {mn:>10.3f} {mx:>10.3f}")

print("-" * 70)
print("Done.")
