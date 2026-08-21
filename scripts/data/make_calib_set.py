from pathlib import Path
import os
import hashlib
import random
import shutil
import sys
import collections


# ==========================================================
# USER CONFIGURATION
# Modify this section when changing dataset, split strategy,
# calibration size, grouping logic, or evaluation data.
# ==========================================================

# Number of calibration samples/groups.
N_EPISODES = int(
    os.environ.get("N_EPISODES", 500)
)

# Random seed for reproducible sampling.
SEED = int(
    os.environ.get("SEED", 42)
)

# Allow rebuilding an existing calibration set.
OVERWRITE = (
    os.environ.get("OVERWRITE", "0") == "1"
)

# Root dataset directory.
# Change when using another dataset.
FULL = Path(
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo"
)

# Output directory for the calibration set.
OUT = Path(
    "/home/zcemml1/medtronic_qat_data/"
    "calib_int8_train_yolo"
)

# Evaluation datasets that calibration data must not overlap.
# Update these paths when using another validation/evaluation setup.
VAL_IMAGE_DIR = (
    FULL / "images" / "val"
)

VAL100_IMAGE_DIR = Path(
    "/home/zcemml1/medtronic_qat_data/"
    "demo_val100_random_yolo/images/val"
)

# Calibration source split.
# Usually training data or another evaluation-disjoint subset.
train_image_dir = (
    FULL / "images" / "train"
)

out_image_dir = (
    OUT / "images"
)

manifest_path = (
    OUT
    / f"calib{N_EPISODES}_seed{SEED}_manifest.txt"
)


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1 << 20),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def episode_of(name):

    # Dataset-specific grouping rule.
    # Replace when file names do not encode episodes in this format.
    #
    # Current:
    # episode_001283_left_00078.jpg
    # -> episode_001283

    return name.split("_left_")[0]


print("============================================================")
print(
    f"Build INT8 calibration set: "
    f"{N_EPISODES} frames"
)
print("============================================================")
print("Source:", train_image_dir)
print("Output:", OUT)
print("Seed:", SEED)


if (
    OUT.exists()
    and any(OUT.iterdir())
    and not OVERWRITE
):
    sys.exit(
        f"Refusing to overwrite existing calibration set "
        f"at {OUT}.\n"
        "Set OVERWRITE=1 to rebuild it."
    )

if (
    OUT.exists()
    and OVERWRITE
):
    shutil.rmtree(OUT)

out_image_dir.mkdir(
    parents=True,
    exist_ok=True,
)


# Image extension may need modification for another dataset.
train_images = sorted(
    train_image_dir.glob("*.jpg")
)

if not train_images:
    sys.exit(
        f"No training images found in "
        f"{train_image_dir}"
    )


# Grouping strategy is dataset-specific.
# Current dataset groups correlated frames by surgical episode.
by_episode = collections.defaultdict(list)

for img in train_images:
    by_episode[
        episode_of(img.name)
    ].append(img)

episodes = sorted(
    by_episode
)

print(
    f"\nTrain split: "
    f"{len(train_images)} frames across "
    f"{len(episodes)} groups"
)

if len(episodes) < N_EPISODES:
    sys.exit(
        f"Requested {N_EPISODES} groups but only "
        f"{len(episodes)} are available."
    )


# Sampling strategy.
# Current implementation selects one frame from each sampled episode.
# Modify if the new dataset is not temporally/group correlated.
rng = random.Random(SEED)

chosen_episodes = sorted(
    rng.sample(
        episodes,
        N_EPISODES,
    )
)

selected = [
    (
        ep,
        rng.choice(by_episode[ep]),
    )
    for ep in chosen_episodes
]


# Evaluation overlap checks.
# Modify/remove group-level checks if the new dataset has no episode/group concept.
val_names = {
    p.name
    for p in VAL_IMAGE_DIR.glob("*.jpg")
}

val_episodes = {
    episode_of(n)
    for n in val_names
}

val100_names = (
    {
        p.name
        for p in VAL100_IMAGE_DIR.glob("*.jpg")
    }
    if VAL100_IMAGE_DIR.exists()
    else set()
)

val100_episodes = {
    episode_of(n)
    for n in val100_names
}

calib_names = {
    img.name
    for _, img in selected
}

calib_episodes = {
    ep
    for ep, _ in selected
}

overlaps = {
    "frames vs val":
        calib_names & val_names,

    "frames vs val100":
        calib_names & val100_names,

    "groups vs val":
        calib_episodes & val_episodes,

    "groups vs val100":
        calib_episodes & val100_episodes,
}

print("\nDisjointness checks:")

for label, hits in overlaps.items():
    print(
        f"  {label:<22} "
        f"{len(hits)} overlap"
    )

if any(overlaps.values()):
    sys.exit(
        "Calibration data overlaps evaluation data."
    )


for _, img in selected:
    shutil.copy2(
        img,
        out_image_dir / img.name,
    )


rows = []
agg = hashlib.sha256()

for ep, img in selected:

    digest = sha256_file(
        out_image_dir / img.name
    )

    rows.append(
        f"{ep}\t"
        f"{img.name}\t"
        f"{digest}"
    )

    agg.update(
        f"{img.name}:{digest}".encode()
    )

set_sha = agg.hexdigest()


# Manifest format may be adapted if another provenance
# structure is preferred.
manifest_path.write_text(
    f"# INT8 calibration set\n"
    f"# source_split: {train_image_dir}\n"
    f"# n_samples: {N_EPISODES}\n"
    f"# seed: {SEED}\n"
    f"# set_sha256: {set_sha}\n"
    f"# columns: group\tframe\tsha256\n"
    + "\n".join(rows)
    + "\n"
)


n_copied = len(
    list(
        out_image_dir.glob("*.jpg")
    )
)

print()
print("Calibration set created.")
print("Frames copied:", n_copied)
print("Groups:", len(calib_episodes))
print("Set sha256:", set_sha)
print("Manifest:", manifest_path)

print("\nFirst 5 selected:")

for ep, img in selected[:5]:
    print(
        f"  {ep}  {img.name}"
    )

print("Done.")
