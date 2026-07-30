from pathlib import Path
import os
import hashlib
import random
import shutil
import sys
import collections


# -----------------------------
# Settings
# -----------------------------
# Build a fixed, reproducible INT8 calibration set for the PTQ stage (docs/05).
# Same discipline as make_val100_subset.py: deterministic seed, written manifest,
# refuses to clobber silently.
#
# WHY IT COMES FROM THE TRAIN SPLIT.
# Calibration observes activation ranges; it must never touch the data the
# accuracy delta is measured on, or INT8 mAP is biased upward. The dataset is
# split BY EPISODE (train 000000-001280, val 001281-001601) and frames within an
# episode are highly correlated -- so taking "other frames" from a val episode
# would still leak into the val100 and full-6449 evaluations. Only an
# episode-level split is safe, and train already is one.
#
# WHY ONE FRAME PER EPISODE.
# 500 random frames drawn from 20,756 would cluster into far fewer episodes and
# give correlated samples -- effectively a much smaller calibration set than the
# count suggests. One frame per episode maximises activation diversity per frame.
N_EPISODES = int(os.environ.get("N_EPISODES", 500))
SEED = int(os.environ.get("SEED", 42))
OVERWRITE = os.environ.get("OVERWRITE", "0") == "1"

FULL = Path(
    "/home/zcemml1/medtronic_qat_data/datasets/"
    "sanoscience_yolo_full_nonexpert_stereo"
)

OUT = Path("/home/zcemml1/medtronic_qat_data/calib_int8_train_yolo")

# The evaluation sets this calibration data must stay disjoint from.
VAL_IMAGE_DIR = FULL / "images" / "val"
VAL100_IMAGE_DIR = Path(
    "/home/zcemml1/medtronic_qat_data/demo_val100_random_yolo/images/val"
)

train_image_dir = FULL / "images" / "train"
out_image_dir = OUT / "images"

manifest_path = OUT / f"calib{N_EPISODES}_seed{SEED}_manifest.txt"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def episode_of(name):
    """episode_001283_left_00078.jpg -> episode_001283"""
    return name.split("_left_")[0]


print("============================================================")
print(f"Build INT8 calibration set: {N_EPISODES} frames, 1 per episode")
print("============================================================")
print("Source split:", train_image_dir)
print("Output:      ", OUT)
print("Seed:        ", SEED)


# -----------------------------
# Guard against silent clobber
# -----------------------------
if OUT.exists() and any(OUT.iterdir()) and not OVERWRITE:
    sys.exit(
        f"Refusing to overwrite existing calibration set at {OUT}.\n"
        "Set OVERWRITE=1 to rebuild it in place."
    )

if OUT.exists() and OVERWRITE:
    shutil.rmtree(OUT)

out_image_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Group train frames by episode
# -----------------------------
train_images = sorted(train_image_dir.glob("*.jpg"))
if not train_images:
    sys.exit(f"No training images found in {train_image_dir}")

by_episode = collections.defaultdict(list)
for img in train_images:
    by_episode[episode_of(img.name)].append(img)

episodes = sorted(by_episode)
print(f"\nTrain split: {len(train_images)} frames across {len(episodes)} episodes")

if len(episodes) < N_EPISODES:
    sys.exit(
        f"Requested {N_EPISODES} episodes but the train split only has "
        f"{len(episodes)}. Lower N_EPISODES."
    )


# -----------------------------
# Sample: N episodes, one frame from each
# -----------------------------
# A single seeded RNG drives both choices so the whole selection is reproducible
# from (SEED, N_EPISODES) alone. The frame is chosen at random rather than taking
# the first, which would over-sample procedure openings.
rng = random.Random(SEED)
chosen_episodes = sorted(rng.sample(episodes, N_EPISODES))
selected = [(ep, rng.choice(by_episode[ep])) for ep in chosen_episodes]


# -----------------------------
# Disjointness check -- the whole point of this script
# -----------------------------
# Verified at BOTH levels. Frame-level alone is not sufficient: correlated frames
# from a shared episode would still contaminate the accuracy measurement.
val_names = {p.name for p in VAL_IMAGE_DIR.glob("*.jpg")}
val_episodes = {episode_of(n) for n in val_names}
val100_names = {p.name for p in VAL100_IMAGE_DIR.glob("*.jpg")} if VAL100_IMAGE_DIR.exists() else set()
val100_episodes = {episode_of(n) for n in val100_names}

calib_names = {img.name for _, img in selected}
calib_episodes = {ep for ep, _ in selected}

overlaps = {
    "frames vs val": calib_names & val_names,
    "frames vs val100": calib_names & val100_names,
    "episodes vs val": calib_episodes & val_episodes,
    "episodes vs val100": calib_episodes & val100_episodes,
}

print("\nDisjointness checks:")
for label, hits in overlaps.items():
    print(f"  {label:<22} {len(hits)} overlap")

if any(overlaps.values()):
    sys.exit(
        "REFUSING TO WRITE: calibration set overlaps the evaluation data.\n"
        "Calibrating on evaluation frames biases INT8 accuracy upward."
    )


# -----------------------------
# Copy frames
# -----------------------------
# Images only -- PTQ calibration is unlabelled. It runs frames through the
# network to observe activation ranges; no ground truth is involved.
for _, img in selected:
    shutil.copy2(img, out_image_dir / img.name)


# -----------------------------
# Manifest + set hash
# -----------------------------
# The set hash identifies this exact calibration data in the engine provenance,
# so an INT8 engine can always be traced to what it was calibrated on.
rows = []
agg = hashlib.sha256()
for ep, img in selected:
    digest = sha256_file(out_image_dir / img.name)
    rows.append(f"{ep}\t{img.name}\t{digest}")
    agg.update(f"{img.name}:{digest}".encode())
set_sha = agg.hexdigest()

manifest_path.write_text(
    f"# INT8 calibration set for docs/05 (PTQ)\n"
    f"# source_split: {train_image_dir}\n"
    f"# n_episodes: {N_EPISODES}   one frame per episode\n"
    f"# seed: {SEED}\n"
    f"# set_sha256: {set_sha}\n"
    f"# disjoint from val and val100 at BOTH frame and episode level\n"
    f"# columns: episode\tframe\tsha256\n"
    + "\n".join(rows) + "\n"
)


# -----------------------------
# Summary
# -----------------------------
n_copied = len(list(out_image_dir.glob("*.jpg")))
print()
print("Calibration set created.")
print("Frames copied:   ", n_copied)
print("Episodes:        ", len(calib_episodes))
print("Set sha256:      ", set_sha)
print("Manifest:        ", manifest_path)
print()
print("First 5 selected:")
for ep, img in selected[:5]:
    print(f"  {ep}  {img.name}")
print("Done.")
