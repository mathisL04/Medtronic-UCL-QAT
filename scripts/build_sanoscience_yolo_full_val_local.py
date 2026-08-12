from huggingface_hub import hf_hub_download
from pathlib import Path
import os
import subprocess
import cv2
import numpy as np
import yaml
import imageio_ffmpeg

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"

ROOT = (
    "Surgical/sanoscience/sanoscience_v1_2_merged/"
    "nonexpert_stereo"
)

LOCAL_CACHE = "/workspace/sanoscience_cache"
DATASET_DIR = Path("/workspace/datasets/sanoscience_yolo_full_nonexpert_stereo")

MAX_EPISODES = 1602

# Optional sharding for parallel preprocessing
START_EPISODE_INDEX = int(os.environ.get("START_EPISODE_INDEX", "0"))
END_EPISODE_INDEX = int(os.environ.get("END_EPISODE_INDEX", str(MAX_EPISODES)))
SHARD_NAME = os.environ.get("SHARD_NAME", f"{START_EPISODE_INDEX}_{END_EPISODE_INDEX}")
TARGET_FPS = 5

USE_VIEW = "left"  # first experiment: left camera only

FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()

CLASS_NAMES = {
    0: "surgical_tool",
}

# Quantized RGB colours identified from the segmentation masks.
# These correspond to the green surgical tools in your screenshots.
TOOL_COLOURS_RGB = [
    (96, 240, 0),
    (96, 240, 112),
]

MIN_AREA = 150
MIN_BOX_SIZE = 8
SAVE_EMPTY_IMAGES = False
MAX_DEBUG_IMAGES = 300


def download(path: str) -> str:
    return hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=path,
        cache_dir=LOCAL_CACHE,
    )


def video_info(path: str):
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return frame_count, fps, width, height


def extract_frame_ffmpeg(video_path: str, frame_idx: int, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        FFMPEG_BINARY,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        video_path,
        "-vf",
        f"select=eq(n\\,{frame_idx})",
        "-frames:v",
        "1",
        str(out_path),
    ]

    subprocess.run(cmd, check=True)

    frame = cv2.imread(str(out_path))

    if frame is None:
        raise RuntimeError(f"Could not read extracted frame: {out_path}")

    return frame


def split_stereo(frame, view: str):
    h, w = frame.shape[:2]

    if w <= h * 1.5:
        return frame

    mid = w // 2

    if view == "left":
        return frame[:, :mid]

    if view == "right":
        return frame[:, mid:]

    raise ValueError(f"Unknown view: {view}")


def segmentation_to_boxes(seg_bgr):
    """
    Convert segmentation image into YOLO boxes.

    We quantize colours to reduce MP4 compression noise.
    Then we select the green surgical tool classes.
    """
    seg_rgb = cv2.cvtColor(seg_bgr, cv2.COLOR_BGR2RGB)

    # Quantize to match the colours printed by exploration script
    seg_q = (seg_rgb // 16) * 16

    mask = np.zeros(seg_q.shape[:2], dtype=bool)

    for colour in TOOL_COLOURS_RGB:
        colour_arr = np.array(colour, dtype=np.uint8)
        mask |= np.all(seg_q == colour_arr, axis=-1)

    mask_u8 = mask.astype(np.uint8) * 255

    # Clean tiny compression fragments
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    boxes = []

    img_h, img_w = mask_u8.shape[:2]

    for label_id in range(1, num_labels):
        x, y, w, h, area = stats[label_id]

        if area < MIN_AREA:
            continue

        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue

        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        box_w = w / img_w
        box_h = h / img_h

        boxes.append((0, x_center, y_center, box_w, box_h, x, y, w, h, area))

    return boxes, mask_u8


def save_yolo_label(label_path: Path, boxes):
    label_path.parent.mkdir(parents=True, exist_ok=True)

    with open(label_path, "w") as f:
        for box in boxes:
            class_id, x_center, y_center, box_w, box_h, *_ = box
            f.write(
                f"{class_id} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{box_w:.6f} {box_h:.6f}\n"
            )


def save_debug_image(debug_path: Path, image, boxes):
    debug = image.copy()

    for box in boxes:
        _, _, _, _, _, x, y, w, h, area = box
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(
            debug,
            f"tool {area}",
            (x, max(20, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
        )

    debug_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_path), debug)


def prepare_dirs():
    for split in ["train", "val"]:
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    (DATASET_DIR / "debug").mkdir(parents=True, exist_ok=True)


def write_yaml():
    yaml_path = DATASET_DIR / "sanoscience_yolo.yaml"

    data = {
        "path": str(DATASET_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"\nYOLO yaml written to: {yaml_path}")


def main():
    print("=" * 80)
    print("Build Sanoscience YOLO tool dataset")
    print("=" * 80)

    prepare_dirs()

    episodes = [f"episode_{i:06d}" for i in range(MAX_EPISODES)]

    train_cutoff = int(0.8 * len(episodes))
    train_episodes = set(episodes[:train_cutoff])

    total_images = 0
    total_boxes = 0
    debug_count = 0

    tmp_dir = DATASET_DIR / f"_tmp_frames_{SHARD_NAME}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, episode in enumerate(episodes):
        if ep_idx < START_EPISODE_INDEX or ep_idx >= END_EPISODE_INDEX:
            continue

        chunk = f"chunk-{ep_idx // 1000:03d}"
        split = "train" if episode in train_episodes else "val"

        print(f"\n[{ep_idx + 1}/{len(episodes)}] Processing {episode} → {split}")

        color_video = f"{ROOT}/videos/{chunk}/observation.images.color/{episode}.mp4"
        seg_video = f"{ROOT}/videos/{chunk}/observation.images.segmentation/{episode}.mp4"

        color_path = download(color_video)
        seg_path = download(seg_video)

        frame_count, fps, width, height = video_info(color_path)
        seg_frame_count, seg_fps, seg_width, seg_height = video_info(seg_path)

        if frame_count != seg_frame_count:
            print("  WARNING: color/segmentation frame count mismatch, skipping")
            continue

        if (width, height) != (seg_width, seg_height):
            print("  WARNING: color/segmentation resolution mismatch, skipping")
            continue

        stride = max(1, round(fps / TARGET_FPS))
        frame_indices = list(range(0, frame_count, stride))

        print(f"  frames: {frame_count}, fps: {fps:.1f}, stride: {stride}, sampled: {len(frame_indices)}")

        for frame_idx in frame_indices:
            color_tmp = tmp_dir / f"{episode}_color_{frame_idx:05d}.png"
            seg_tmp = tmp_dir / f"{episode}_seg_{frame_idx:05d}.png"

            color_frame = extract_frame_ffmpeg(color_path, frame_idx, color_tmp)
            seg_frame = extract_frame_ffmpeg(seg_path, frame_idx, seg_tmp)

            # Delete temporary extracted PNGs immediately after loading them into memory.
            color_tmp.unlink(missing_ok=True)
            seg_tmp.unlink(missing_ok=True)

            color_view = split_stereo(color_frame, USE_VIEW)
            seg_view = split_stereo(seg_frame, USE_VIEW)

            boxes, mask = segmentation_to_boxes(seg_view)

            if len(boxes) == 0 and not SAVE_EMPTY_IMAGES:
                continue

            stem = f"{episode}_{USE_VIEW}_{frame_idx:05d}"

            image_out = DATASET_DIR / "images" / split / f"{stem}.jpg"
            label_out = DATASET_DIR / "labels" / split / f"{stem}.txt"

            cv2.imwrite(str(image_out), color_view)
            save_yolo_label(label_out, boxes)

            total_images += 1
            total_boxes += len(boxes)

            if debug_count < MAX_DEBUG_IMAGES:
                debug_out = DATASET_DIR / "debug" / f"{stem}_debug.jpg"
                save_debug_image(debug_out, color_view, boxes)
                debug_count += 1

    write_yaml()

    print("\n" + "=" * 80)
    print("Dataset build complete")
    print("=" * 80)
    print(f"Dataset dir: {DATASET_DIR}")
    print(f"Images saved: {total_images}")
    print(f"Boxes saved: {total_boxes}")
    print(f"Debug images: {debug_count}")
    print("\nOpen debug images here:")
    print(f"  {DATASET_DIR / 'debug'}")


if __name__ == "__main__":
    main()