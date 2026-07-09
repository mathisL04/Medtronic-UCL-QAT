from huggingface_hub import hf_hub_download
from pathlib import Path
from collections import Counter
import cv2
import json
import pandas as pd
import numpy as np
import subprocess

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"

ROOT = (
    "Surgical/sanoscience/sanoscience_v1_2_merged/"
    "nonexpert_stereo"
)

EPISODE = "episode_000000"

LOCAL_CACHE = "/workspace/sanoscience_cache"
OUTPUT_DIR = Path("/workspace/sanoscience_explore/nonexpert_stereo") / EPISODE
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_VIDEO = f"{ROOT}/videos/chunk-000/observation.images.color/{EPISODE}.mp4"
SEG_VIDEO = f"{ROOT}/videos/chunk-000/observation.images.segmentation/{EPISODE}.mp4"
PARQUET_FILE = f"{ROOT}/data/chunk-000/{EPISODE}.parquet"

INFO_JSON = f"{ROOT}/meta/info.json"
MODALITY_JSON = f"{ROOT}/meta/modality.json"
EPISODES_JSONL = f"{ROOT}/meta/episodes.jsonl"


def download(path: str) -> str:
    print(f"Downloading/loading: {path}")
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=path,
        cache_dir=LOCAL_CACHE,
    )


def video_info(path: str, name: str):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    print(f"\n{name} video info")
    print(f"  frames: {frame_count}")
    print(f"  fps: {fps}")
    print(f"  width: {width}")
    print(f"  height: {height}")

    if width > height * 1.5:
        print("  stereo layout: likely side-by-side ✅")
    else:
        print("  stereo layout: likely mono")

    return frame_count, fps, width, height


def read_frame(path: str, frame_idx: int):
    """
    Extract one frame using ffmpeg instead of OpenCV.

    Sanoscience videos are AV1 encoded. OpenCV can read metadata
    but may fail to decode actual frames. ffmpeg handles AV1 correctly.
    """
    tmp_dir = OUTPUT_DIR / "_tmp_frames"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(path).stem
    out_path = tmp_dir / f"{safe_name}_frame_{frame_idx:05d}.png"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        path,
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-frames:v",
        "1",
        str(out_path),
    ]

    subprocess.run(
        cmd,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame = cv2.imread(str(out_path))

    if frame is None:
        raise RuntimeError(f"Could not read extracted frame: {out_path}")

    return frame


def save_stereo_splits(frame, out_prefix: Path):
    h, w = frame.shape[:2]

    if w > h * 1.5:
        mid = w // 2
        left = frame[:, :mid]
        right = frame[:, mid:]

        cv2.imwrite(str(out_prefix) + "_left.png", left)
        cv2.imwrite(str(out_prefix) + "_right.png", right)


def print_top_segmentation_colours(seg_frame, frame_idx: int, top_k: int = 20):
    """
    Segmentation is stored as video, so compression may create many near-identical colours.
    We quantize colours to reduce compression noise.
    """
    rgb = cv2.cvtColor(seg_frame, cv2.COLOR_BGR2RGB)

    sampled = rgb[::5, ::5].reshape(-1, 3)
    quantized = (sampled // 16) * 16

    counts = Counter(map(tuple, quantized.tolist()))
    top = counts.most_common(top_k)

    print(f"\nTop approximate segmentation colours for frame {frame_idx}")
    print("  RGB colour       sampled pixel count")
    for colour, count in top:
        print(f"  {colour}   {count}")


def main():
    print("=" * 80)
    print("Sanoscience segmentation exploration")
    print("=" * 80)

    # Metadata
    info_path = download(INFO_JSON)
    modality_path = download(MODALITY_JSON)
    episodes_path = download(EPISODES_JSONL)

    with open(info_path, "r") as f:
        info = json.load(f)

    with open(modality_path, "r") as f:
        modality = json.load(f)

    print("\nMetadata loaded")
    print(f"  info.json keys: {list(info.keys())}")
    print(f"  modality.json keys: {list(modality.keys())}")

    # One episode
    color_path = download(COLOR_VIDEO)
    seg_path = download(SEG_VIDEO)
    parquet_path = download(PARQUET_FILE)

    # Parquet info
    df = pd.read_parquet(parquet_path)

    print("\nParquet info")
    print(f"  rows / frames: {len(df)}")
    print(f"  columns: {list(df.columns)}")
    print("\nFirst rows:")
    print(df.head())

    # Video info
    color_frames, color_fps, color_w, color_h = video_info(color_path, "Color")
    seg_frames, seg_fps, seg_w, seg_h = video_info(seg_path, "Segmentation")

    if color_frames == seg_frames:
        print("\nFrame count alignment: OK ✅")
    else:
        print("\nWARNING: color and segmentation frame counts differ")

    if (color_w, color_h) == (seg_w, seg_h):
        print("Resolution alignment: OK ✅")
    else:
        print("WARNING: color and segmentation resolutions differ")

    # Save sample frames
    sample_indices = sorted(set([
        0,
        min(10, color_frames - 1),
        min(30, color_frames - 1),
        color_frames // 2,
        color_frames - 1,
    ]))

    print(f"\nSaving sample frames to: {OUTPUT_DIR}")
    print(f"Sample frame indices: {sample_indices}")

    for idx in sample_indices:
        color_frame = read_frame(color_path, idx)
        seg_frame = read_frame(seg_path, idx)

        color_out = OUTPUT_DIR / f"{EPISODE}_frame_{idx:05d}_color.jpg"
        seg_out = OUTPUT_DIR / f"{EPISODE}_frame_{idx:05d}_segmentation.png"
        overlay_out = OUTPUT_DIR / f"{EPISODE}_frame_{idx:05d}_overlay.jpg"

        cv2.imwrite(str(color_out), color_frame)
        cv2.imwrite(str(seg_out), seg_frame)

        overlay = cv2.addWeighted(color_frame, 0.65, seg_frame, 0.35, 0)
        cv2.imwrite(str(overlay_out), overlay)

        save_stereo_splits(color_frame, OUTPUT_DIR / f"{EPISODE}_frame_{idx:05d}_color")
        save_stereo_splits(seg_frame, OUTPUT_DIR / f"{EPISODE}_frame_{idx:05d}_segmentation")
        save_stereo_splits(overlay, OUTPUT_DIR / f"{EPISODE}_frame_{idx:05d}_overlay")

        print_top_segmentation_colours(seg_frame, idx)

    print("\nDone.")
    print("Open debug images in VS Code:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()