from huggingface_hub import HfApi, hf_hub_download
from ultralytics import YOLO
from pathlib import Path
import torch
import time

# -----------------------------
# Configuration
# -----------------------------
REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"

RGB_DIR = (
    "Surgical/utenn/benchtop_datasets_round2_with_part_seg/"
    "videos/chunk-000/observation.images.rgb"
)

LOCAL_CACHE = "/workspace/openh_api_cache"
OUTPUT_DIR = "/workspace/yolo_openh_smoke"

MODEL_NAME = "yolo26n.pt"
IMG_SIZE = 640

# Use GPU if PyTorch can see CUDA, otherwise use CPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"

print("======================================")
print("Open-H YOLO26n smoke test")
print("======================================")
print(f"Model: {MODEL_NAME}")
print(f"Device selected: {DEVICE}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")

# -----------------------------
# 1. Connect to Hugging Face
# -----------------------------
api = HfApi()

# -----------------------------
# 2. List RGB videos in the target folder
# -----------------------------
items = api.list_repo_tree(
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=RGB_DIR,
    recursive=False,
)

video_files = sorted([
    item.path for item in items
    if item.path.endswith(".mp4")
])

print("\nFound videos:")
for video in video_files:
    print(" -", video)

if not video_files:
    raise RuntimeError("No MP4 videos found. Check the Hugging Face path.")

# -----------------------------
# 3. Load YOLO26n once
# -----------------------------
print("\nLoading YOLO model...")
model = YOLO(MODEL_NAME)

# -----------------------------
# 4. Download and run YOLO on each video
# -----------------------------
summary = []

for video_file in video_files:
    print("\n======================================")
    print("Fetching from Hugging Face:")
    print(video_file)

    local_video_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=video_file,
        local_dir=LOCAL_CACHE,
    )

    print("Local cached path:")
    print(local_video_path)

    video_stem = Path(local_video_path).stem
    run_name = f"utenn_rgb_{video_stem}_{'gpu' if DEVICE != 'cpu' else 'cpu'}"

    print(f"\nRunning YOLO26n on {video_stem}...")
    print(f"Output folder name: {run_name}")

    start_time = time.time()
    frame_count = 0

    # stream=True prevents accumulating all results in RAM
    results_stream = model.predict(
        source=local_video_path,
        imgsz=IMG_SIZE,
        device=DEVICE,
        save=True,
        project=OUTPUT_DIR,
        name=run_name,
        exist_ok=True,
        stream=True,
    )

    for _ in results_stream:
        frame_count += 1

    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    print(f"\nFinished {video_stem}")
    print(f"Frames processed: {frame_count}")
    print(f"Elapsed time: {elapsed:.2f} seconds")
    print(f"Approx throughput: {fps:.2f} FPS")

    summary.append({
        "video": video_stem,
        "frames": frame_count,
        "elapsed": elapsed,
        "fps": fps,
        "output": f"{OUTPUT_DIR}/{run_name}",
    })

# -----------------------------
# 5. Final summary
# -----------------------------
print("\n======================================")
print("Done. Summary:")
print("======================================")

for item in summary:
    print(
        f"{item['video']}: "
        f"{item['frames']} frames, "
        f"{item['elapsed']:.2f}s, "
        f"{item['fps']:.2f} FPS"
    )
    print(f"Saved to: {item['output']}")

print(f"\nAll results saved under: {OUTPUT_DIR}")