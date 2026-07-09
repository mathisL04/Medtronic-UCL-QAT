from huggingface_hub import HfApi

REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
ROOT = "Surgical/utenn/benchtop_datasets_round2_with_part_seg"

api = HfApi()

folders = [
    ROOT,
    f"{ROOT}/meta",
    f"{ROOT}/data/chunk-000",
    f"{ROOT}/videos/chunk-000",
    f"{ROOT}/videos/chunk-000/observation.images.rgb",
    f"{ROOT}/videos/chunk-000/observation.images.part_id",
    f"{ROOT}/raw",
    f"{ROOT}/raw/seg",
    f"{ROOT}/raw/seg_schema",
    f"{ROOT}/raw/toolposes",
]

for folder in folders:
    print("\n" + "=" * 90)
    print(folder)
    print("=" * 90)

    try:
        items = api.list_repo_tree(
            repo_id=REPO_ID,
            repo_type="dataset",
            path_in_repo=folder,
            recursive=False,
        )

        for item in items:
            print(item.path)

    except Exception as e:
        print("Could not list:", folder)
        print(e)