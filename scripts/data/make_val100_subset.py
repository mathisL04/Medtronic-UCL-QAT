from pathlib import Path
import os
import random
import shutil
import sys


# -----------------------------
# Settings
# -----------------------------
# Build a fixed, reproducible N-image random subset of the validation split,
# with matching label files, a manifest, and a YOLO data YAML. The seed makes
# the selection deterministic, so re-running reproduces the same subset. This
# is the scripted form of the ad-hoc subset step used for the latency and
# accuracy benchmarks.
N_IMAGES = int(os.environ.get("N_IMAGES", 100))
SEED = int(os.environ.get("SEED", 42))
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"

FULL = Path(
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo"
)

SUB = Path("/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo")

image_dir = FULL / "images" / "val"
label_dir = FULL / "labels" / "val"

out_image_dir = SUB / "images" / "val"
out_label_dir = SUB / "labels" / "val"

manifest_path = SUB / f"val{N_IMAGES}_seed{SEED}_manifest.txt"
yaml_path = SUB / f"sanoscience_yolo_val{N_IMAGES}_random.yaml"


print("============================================================")
print(f"Build fixed {N_IMAGES}-image random validation subset")
print("============================================================")
print("Source dataset:", FULL)
print("Output subset:", SUB)
print("Seed:", SEED)


# -----------------------------
# Guard against silent clobber
# -----------------------------
# The selection is deterministic for a given seed, so recreating is safe, but
# we refuse to wipe an existing non-empty subset unless OVERWRITE=1 is set.
if SUB.exists() and any(SUB.iterdir()) and not OVERWRITE:
    sys.exit(
        f"Refusing to overwrite existing subset at {SUB}.\n"
        "Set OVERWRITE=1 to rebuild it in place."
    )

if SUB.exists() and OVERWRITE:
    shutil.rmtree(SUB)

out_image_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Sample images
# -----------------------------
images = sorted(image_dir.glob("*.jpg"))

if len(images) < N_IMAGES:
    raise ValueError(
        f"Requested {N_IMAGES} images, but only found "
        f"{len(images)} validation images in {image_dir}."
    )

random.seed(SEED)
sampled_images = random.sample(images, N_IMAGES)


# -----------------------------
# Copy images + matching labels
# -----------------------------
for img in sampled_images:
    label = label_dir / f"{img.stem}.txt"

    if not label.exists():
        raise FileNotFoundError(f"Missing label file for {img.name}: {label}")

    shutil.copy2(img, out_image_dir / img.name)
    shutil.copy2(label, out_label_dir / label.name)


# -----------------------------
# Write manifest
# -----------------------------
with open(manifest_path, "w") as f:
    for img in sampled_images:
        f.write(img.name + "\n")


# -----------------------------
# Write YOLO data YAML
# -----------------------------
# Both train and val point at the same subset split: this dataset is used only
# for evaluation (accuracy and latency), never for training.
yaml_path.write_text(
    f"path: {SUB}\n"
    "train: images/val\n"
    "val: images/val\n"
    "\n"
    "names:\n"
    "  0: surgical_tool\n"
)


# -----------------------------
# Summary
# -----------------------------
n_images = len(list(out_image_dir.glob("*.jpg")))
n_labels = len(list(out_label_dir.glob("*.txt")))

print()
print("Subset created.")
print("Images copied:", n_images)
print("Labels copied:", n_labels)
print("Manifest:", manifest_path)
print("Data YAML:", yaml_path)
print()
print("First 5 sampled images:")
for img in sampled_images[:5]:
    print(" -", img.name)
print("Done.")
