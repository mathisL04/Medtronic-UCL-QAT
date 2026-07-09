from huggingface_hub import hf_hub_download
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import yaml
import shutil

# --------------------------------------------------
# Configuration
# --------------------------------------------------
REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
ROOT = "Surgical/utenn/benchtop_datasets_round2_with_part_seg"

CACHE = "/workspace/openh_api_cache"
OUT = Path("/workspace/datasets/utenn_yolo")

# Simple first split:
# episode_000000 = training
# episode_000001 = validation
EPISODE_SPLIT = {
    "episode_000000": "train",
    "episode_000001": "val",
}

# YOLO classes
CLASS_NAMES = {
    0: "grasper",
    1: "grasper2",
}

# Segmentation IDs found in label_schema.json
SEG_IDS_TO_CLASS = {
    0: [1, 2, 3, 4],  # grasper parts
    1: [5, 6, 7, 8],  # grasper2 parts
}

# Ignore tiny masks / noise
MIN_PIXELS = 20


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


def bbox_from_mask(mask: np.ndarray, seg_ids: list[int]):
    """
    Convert segmentation IDs into one YOLO bounding box.

    Input:
        mask: segmentation mask, shape H x W
        seg_ids: list of pixel IDs belonging to one class

    Output:
        x_center, y_center, width, height normalized between 0 and 1
    """
    binary = np.isin(mask, seg_ids)

    pixel_count = int(binary.sum())
    if pixel_count < MIN_PIXELS:
        return None

    ys, xs = np.where(binary)

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    h, w = mask.shape[:2]

    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    box_w = (x_max - x_min + 1) / w
    box_h = (y_max - y_min + 1) / h

    return x_center, y_center, box_w, box_h, pixel_count


def prepare_output_dirs():
    """Create clean YOLO dataset folder."""
    if OUT.exists():
        print(f"Removing existing dataset folder: {OUT}")
        shutil.rmtree(OUT)

    for split in ["train", "val"]:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    (OUT / "debug").mkdir(parents=True, exist_ok=True)


def write_yaml():
    """Write YOLO dataset YAML file."""
    yaml_path = OUT / "utenn_yolo.yaml"

    data = {
        "path": str(OUT),
        "train": "images/train",
        "val": "images/val",
        "names": CLASS_NAMES,
    }

    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    print(f"\nWrote YAML file: {yaml_path}")


def draw_debug_boxes(frame, boxes):
    """Draw boxes on frame for quick visual debugging."""
    debug = frame.copy()
    h, w = debug.shape[:2]

    for class_id, x_center, y_center, box_w, box_h in boxes:
        x1 = int((x_center - box_w / 2) * w)
        y1 = int((y_center - box_h / 2) * h)
        x2 = int((x_center + box_w / 2) * w)
        y2 = int((y_center + box_h / 2) * h)

        label = CLASS_NAMES[class_id]

        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            debug,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    return debug


# --------------------------------------------------
# Main episode conversion
# --------------------------------------------------
def convert_episode(episode: str, split: str):
    print("\n" + "=" * 90)
    print(f"Converting {episode} → {split}")
    print("=" * 90)

    parquet_file = f"{ROOT}/data/chunk-000/{episode}.parquet"
    rgb_video_file = f"{ROOT}/videos/chunk-000/observation.images.rgb/{episode}.mp4"

    parquet_path = download_file(parquet_file)
    rgb_video_path = download_file(rgb_video_file)

    df = pd.read_parquet(parquet_path)

    print("Parquet shape:", df.shape)
    print("RGB video:", rgb_video_path)

    cap = cv2.VideoCapture(rgb_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {rgb_video_path}")

    images_written = 0
    boxes_written = 0
    empty_label_files = 0

    for row_idx, row in df.iterrows():
        frame_idx = int(row["observation.meta.frame_index"])
        step_idx = int(row["step_index"])

        # -----------------------------
        # Load segmentation mask
        # -----------------------------
        seg_relpath = row["observation.meta.seg_png_relpath"]
        seg_file = f"{ROOT}/{seg_relpath}"
        seg_path = download_file(seg_file)

        mask = cv2.imread(seg_path, cv2.IMREAD_UNCHANGED)

        if mask is None:
            print(f"Skipping unreadable mask: {seg_path}")
            continue

        if mask.ndim != 2:
            raise ValueError(f"Expected 2D label mask, got shape {mask.shape}")

        # -----------------------------
        # Load matching RGB frame
        # -----------------------------
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            print(f"Skipping unreadable video frame: {episode} frame {frame_idx}")
            continue

        # Check image and mask alignment
        frame_h, frame_w = frame.shape[:2]
        mask_h, mask_w = mask.shape[:2]

        if (frame_h, frame_w) != (mask_h, mask_w):
            raise ValueError(
                f"Frame/mask size mismatch: frame={frame.shape}, mask={mask.shape}"
            )

        # -----------------------------
        # Convert mask IDs to YOLO boxes
        # -----------------------------
        yolo_lines = []
        debug_boxes = []

        for class_id, seg_ids in SEG_IDS_TO_CLASS.items():
            bbox = bbox_from_mask(mask, seg_ids)

            if bbox is None:
                continue

            x_center, y_center, box_w, box_h, pixel_count = bbox

            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

            debug_boxes.append((class_id, x_center, y_center, box_w, box_h))
            boxes_written += 1

        # -----------------------------
        # Save image and YOLO label
        # -----------------------------
        stem = f"{episode}_{step_idx:05d}"

        image_path = OUT / "images" / split / f"{stem}.jpg"
        label_path = OUT / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(image_path), frame)

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        if len(yolo_lines) == 0:
            empty_label_files += 1

        images_written += 1

        # Save a few debug images with boxes drawn
        if row_idx in [0, 10, 30, 60, 90, 120]:
            debug = draw_debug_boxes(frame, debug_boxes)
            debug_path = OUT / "debug" / f"{stem}_debug_boxes.jpg"
            cv2.imwrite(str(debug_path), debug)

    cap.release()

    print(f"Images written: {images_written}")
    print(f"Boxes written: {boxes_written}")
    print(f"Empty label files: {empty_label_files}")


def main():
    prepare_output_dirs()

    for episode, split in EPISODE_SPLIT.items():
        convert_episode(episode, split)

    write_yaml()

    print("\n" + "=" * 90)
    print("DONE")
    print("=" * 90)
    print("YOLO dataset created at:")
    print(OUT)

    print("\nExpected structure:")
    print(OUT / "images" / "train")
    print(OUT / "labels" / "train")
    print(OUT / "images" / "val")
    print(OUT / "labels" / "val")
    print(OUT / "utenn_yolo.yaml")


if __name__ == "__main__":
    main()