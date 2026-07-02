from huggingface_hub import hf_hub_download
import pandas as pd
import json

REPO_ID = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
ROOT = "Surgical/utenn/benchtop_datasets_round2_with_part_seg"
CACHE = "/workspace/openh_api_cache"

files = [
    f"{ROOT}/meta/info.json",
    f"{ROOT}/meta/tasks.jsonl",
    f"{ROOT}/meta/episodes.jsonl",
    f"{ROOT}/data/chunk-000/episode_000000.parquet",
    f"{ROOT}/data/chunk-000/episode_000001.parquet",
]

for file in files:
    print("\n" + "=" * 90)
    print("Downloading/reading:", file)
    print("=" * 90)

    local_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=file,
        local_dir=CACHE,
    )

    print("Local path:", local_path)

    if file.endswith(".json"):
        with open(local_path, "r") as f:
            data = json.load(f)
        print(json.dumps(data, indent=2)[:4000])

    elif file.endswith(".jsonl"):
        with open(local_path, "r") as f:
            for i, line in enumerate(f):
                print(line.strip())
                if i >= 10:
                    break

    elif file.endswith(".parquet"):
        df = pd.read_parquet(local_path)
        print("Shape:", df.shape)
        print("\nColumns:")
        for col in df.columns:
            print(" -", col)

        print("\nFirst 5 rows:")
        print(df.head())
