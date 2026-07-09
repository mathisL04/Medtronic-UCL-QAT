from huggingface_hub import HfApi, hf_hub_download
from pathlib import Path
import json
import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"

ROOT = "Surgical/utenn/benchtop_datasets_round2_with_part_seg"

CACHE = "/workspace/openh_api_cache"
OUT = Path("/workspace/utenn_explore")
OUT.mkdir(parents=True, exist_ok=True)

EPISODES = ["episode_000000", "episode_000001"]

# Frames to inspect visually
SAMPLE_INDICES = [0, 10, 30, 60, 90, 120]

api = HfApi()


# --------------------------------------------------
# Helper functions
# --------------------------------------------------
def download_file(path_in_repo: str) -> str:
    """Download one file from Hugging Face into local cache."""
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=path_in_repo,
        local_dir=CACHE,
    )


def print_json_file(path_in_repo: str, episode: str) -> None:
    """Download, print, and save a JSON file locally."""
    local_path = download_file(path_in_repo)

    with open(local_path, "r") as f:
        data = json.load(f)

    print("\n" + "-" * 90)
    print(path_in_repo)
    print("-" * 90)
    print(json.dumps(data, indent=2)[:12000])

    out_json = OUT / f"{episode}_{Path(path_in_repo).name}"
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved readable copy to: {out_json}")


def inspect_mask(mask: np.ndarray) -> None:
    """Print unique IDs or colors in a segmentation mask."""
    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)

    if mask.ndim == 2:
        ids, counts = np.unique(mask, return_counts=True)
        print("Unique label IDs and pixel counts:")
        for label_id, count in zip(ids, counts):
            print(f"  ID {int(label_id)}: {int(count)} pixels")

    else:
        flat = mask.reshape(-1, mask.shape[-1])
        colors, counts = np.unique(flat, axis=0, return_counts=True)

        print("Unique colors and pixel counts:")
        for color, count in zip(colors[:30], counts[:30]):
            print(f"  Color {color.tolist()}: {int(count)} pixels")

        print("Total unique colors:", len(colors))


def make_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Create a simple RGB + segmentation overlay for visual checking."""
    if mask.ndim == 2:
        mask_norm = mask.astype(np.float32)

        if mask_norm.max() > 0:
            mask_norm = mask_norm / mask_norm.max()

        heat = cv2.applyColorMap(
            (mask_norm * 255).astype(np.uint8),
            cv2.COLORMAP_JET,
        )

        overlay = cv2.addWeighted(frame, 0.65, heat, 0.35, 0)
        return overlay

    # If mask is already colored, blend directly
    mask_resized = mask
    if mask_resized.shape[:2] != frame.shape[:2]:
        mask_resized = cv2.resize(
            mask_resized,
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    overlay = cv2.addWeighted(frame, 0.65, mask_resized, 0.35, 0)
    return overlay


# --------------------------------------------------
# 1. Inspect segmentation schemas
# --------------------------------------------------
for episode in EPISODES:
    print("\n" + "=" * 90)
    print(f"SEGMENTATION SCHEMA: {episode}")
    print("=" * 90)

    schema_root = f"{ROOT}/raw/seg_schema/{episode}"

    schema_items = list(
        api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo=schema_root,
            recursive=True,
        )
    )

    print("\nSchema files found:")
    for item in schema_items:
        print(" -", item.path)

    for item in schema_items:
        if item.path.endswith(".json"):
            print_json_file(item.path, episode)


# --------------------------------------------------
# 2. Inspect masks and matching RGB frames
# --------------------------------------------------
for episode in EPISODES:
    print("\n" + "=" * 90)
    print(f"MASK / RGB EXPLORATION: {episode}")
    print("=" * 90)

    # Download parquet
    parquet_file = f"{ROOT}/data/chunk-000/{episode}.parquet"
    parquet_path = download_file(parquet_file)
    df = pd.read_parquet(parquet_path)

    print("\nParquet shape:", df.shape)
    print("Columns:")
    for col in df.columns:
        print(" -", col)

    # Download RGB video
    rgb_video_file = f"{ROOT}/videos/chunk-000/observation.images.rgb/{episode}.mp4"
    rgb_video_path = download_file(rgb_video_file)

    print("\nRGB video:", rgb_video_path)

    cap = cv2.VideoCapture(rgb_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RGB video: {rgb_video_path}")

    for idx in SAMPLE_INDICES:
        if idx >= len(df):
            continue

        row = df.iloc[idx]

        seg_relpath = row["observation.meta.seg_png_relpath"]
        seg_file = f"{ROOT}/{seg_relpath}"
        seg_path = download_file(seg_file)

        mask = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            print(f"Could not read mask: {seg_path}")
            continue

        print("\n" + "-" * 90)
        print(f"{episode} frame {idx}")
        print("Segmentation path:", seg_relpath)
        inspect_mask(mask)

        # Read matching RGB frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            print(f"Could not read RGB frame {idx}")
            continue

        # Save outputs
        rgb_out = OUT / f"{episode}_frame_{idx:05d}_rgb.jpg"
        mask_out = OUT / f"{episode}_frame_{idx:05d}_mask.png"
        overlay_out = OUT / f"{episode}_frame_{idx:05d}_overlay.jpg"

        cv2.imwrite(str(rgb_out), frame)
        cv2.imwrite(str(mask_out), mask)

        overlay = make_overlay(frame, mask)
        cv2.imwrite(str(overlay_out), overlay)

        print("Saved RGB:", rgb_out)
        print("Saved mask:", mask_out)
        print("Saved overlay:", overlay_out)

    cap.release()


print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print("Explore outputs saved in:")
print(OUT)