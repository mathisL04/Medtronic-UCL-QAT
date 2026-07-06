from huggingface_hub import HfApi
from collections import defaultdict

REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"

ROOT = (
    "Surgical/sanoscience/sanoscience_v1_2_merged/"
    "nonexpert_stereo"
)

api = HfApi()

print("=" * 80)
print("Sanoscience structure inspection")
print("=" * 80)
print(f"Repo: {REPO_ID}")
print(f"Root: {ROOT}")

files = []

for item in api.list_repo_tree(
    repo_id=REPO_ID,
    repo_type="dataset",
    path_in_repo=ROOT,
    recursive=True,
):
    if item.__class__.__name__ == "RepoFile":
        files.append(item.path)

print(f"\nTotal files found: {len(files)}")

checks = {
    "README": ["README.md"],
    "Metadata": ["meta/"],
    "Parquet data": [".parquet"],
    "All videos": [".mp4"],
    "Color / RGB videos": ["color", "rgb"],
    "Segmentation videos": ["segmentation", "seg"],
    "Depth videos": ["depth"], t
    "Normals videos": ["normal"],
    "Optical flow videos": ["flow"],
}

for name, patterns in checks.items():
    matched = [
        f for f in files
        if any(p.lower() in f.lower() for p in patterns)
    ]

    print(f"\n{name}: {len(matched)}")
    for f in matched[:15]:
        print("  ", f)

    if len(matched) > 15:
        print(f"  ... {len(matched) - 15} more")

print("\nTop-level folders/files:")
top = defaultdict(int)

for f in files:
    rel = f.replace(ROOT + "/", "")
    top[rel.split("/")[0]] += 1

for key, count in top.items():
    print(f"  {key}: {count} files")

print("\nDone.")