from pathlib import Path
from collections import Counter
import hashlib

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = PROJECT_ROOT / "Train.csv"
TEST_CSV = PROJECT_ROOT / "Test.csv"

TRAIN_DIR = PROJECT_ROOT / "data/train"
TEST_DIR = PROJECT_ROOT / "data/test"

OUTPUT_DIR = PROJECT_ROOT / "outputs/audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_observations(csv_path, image_root, split):
    df = pd.read_csv(csv_path)
    results = []

    for _, obs in df.iterrows():
        folder = str(obs["FolderName"])
        side = str(obs["Side"])
        start = int(obs["Start"])
        end = int(obs["End"])

        files = []

        for slice_id in range(start, end + 1):
            path = image_root / folder / f"{folder}_{side}_{slice_id:03d}.png"

            if path.exists():
                files.append((slice_id, path))

        missing = [
            i
            for i in range(start, end + 1)
            if not (image_root / folder / f"{folder}_{side}_{i:03d}.png").exists()
        ]

        sizes = []
        modes = []
        image_hashes = []

        for slice_id, path in files:
            with Image.open(path) as image:
                sizes.append(image.size)
                modes.append(image.mode)

            image_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())

        sequence_hash = (
            hashlib.sha256("".join(image_hashes).encode()).hexdigest()
            if image_hashes
            else None
        )

        record = {
            "Split": split,
            "ID": obs["ID"],
            "FolderName": folder,
            "Side": side,
            "Start": start,
            "End": end,
            "PlantNumber": obs.get("PlantNumber", np.nan),
            "RootVolume": obs.get("RootVolume", np.nan),
            "ExpectedImages": end - start + 1,
            "FoundImages": len(files),
            "MissingSlices": ",".join(map(str, missing)),
            "SequenceHash": sequence_hash,
            "UniqueSizes": str(sorted(set(sizes))),
            "UniqueModes": str(sorted(set(modes))),
        }

        results.append(record)

    return pd.DataFrame(results)


print("=" * 70)
print("OBSERVATION IMAGE AUDIT")
print("=" * 70)

train = inspect_observations(TRAIN_CSV, TRAIN_DIR, "train")
test = inspect_observations(TEST_CSV, TEST_DIR, "test")

train.to_csv(OUTPUT_DIR / "train_image_audit.csv", index=False)
test.to_csv(OUTPUT_DIR / "test_image_audit.csv", index=False)

# ------------------------------------------------------------
# Image completeness
# ------------------------------------------------------------

for name, df in [("TRAIN", train), ("TEST", test)]:
    print(f"\n{name}")
    print("-" * 50)

    print("Observations:", len(df))
    print("Complete observations:", (df["FoundImages"] == df["ExpectedImages"]).sum())
    print("Incomplete observations:", (df["FoundImages"] != df["ExpectedImages"]).sum())
    print("Total expected images:", df["ExpectedImages"].sum())
    print("Total found images:", df["FoundImages"].sum())

    print("\nImage sizes:")
    print(df["UniqueSizes"].value_counts().head(10))

    print("\nImage modes:")
    print(df["UniqueModes"].value_counts())

# ------------------------------------------------------------
# Identical sequences within training observations
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IDENTICAL TRAINING IMAGE SEQUENCES")
print("=" * 70)

valid = train.dropna(subset=["SequenceHash"])

groups = valid.groupby("SequenceHash").agg(
    observations=("ID", "count"),
    unique_volumes=("RootVolume", "nunique"),
    volume_min=("RootVolume", "min"),
    volume_max=("RootVolume", "max"),
    folders=("FolderName", "nunique"),
)

duplicate_groups = groups[groups["observations"] > 1]

print("Unique image sequences:", len(groups))
print("Repeated image sequences:", len(duplicate_groups))

conflicting = duplicate_groups[duplicate_groups["unique_volumes"] > 1]

print("Repeated sequences with different RootVolume:", len(conflicting))

if len(conflicting):
    print("\nExamples:")
    print(conflicting.sort_values("unique_volumes", ascending=False).head(20))

# ------------------------------------------------------------
# Folder overlap between train and test
# ------------------------------------------------------------

train_folders = set(train["FolderName"])
test_folders = set(test["FolderName"])

print("\n" + "=" * 70)
print("TRAIN / TEST FOLDER OVERLAP")
print("=" * 70)

print("Train folders:", len(train_folders))
print("Test folders:", len(test_folders))
print("Shared folders:", len(train_folders & test_folders))

print("\nAudit files saved to:", OUTPUT_DIR)
print("AUDIT COMPLETE")
