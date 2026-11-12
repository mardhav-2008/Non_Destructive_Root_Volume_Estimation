import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import (
    RootVolumeDataset,
    build_category_maps,
)


TRAIN_CSV = ROOT / "Train.csv"
IMAGE_ROOT = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs" / "splits"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(TRAIN_CSV)

df["strip_count"] = df["End"].astype(int) - df["Start"].astype(int) + 1

required_columns = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
    "RootVolume",
    "Genotype",
    "Stage",
]

if df[required_columns].isnull().any().any():
    raise ValueError("Missing values found in required columns.")

# --------------------------------------------------
# Grouped train/validation split
# --------------------------------------------------

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_indices, val_indices = next(
    splitter.split(
        df,
        groups=df["FolderName"],
    )
)

train_df = df.iloc[train_indices].copy()
val_df = df.iloc[val_indices].copy()

# Confirm no folder leakage.
train_folders = set(train_df["FolderName"])
val_folders = set(val_df["FolderName"])

assert train_folders.isdisjoint(val_folders)

# --------------------------------------------------
# Category maps
# --------------------------------------------------

category_maps = build_category_maps(df)

# --------------------------------------------------
# Save artifacts
# --------------------------------------------------

train_df.to_csv(
    OUTPUT_DIR / "train_split.csv",
    index=False,
)

val_df.to_csv(
    OUTPUT_DIR / "validation_split.csv",
    index=False,
)

with open(OUTPUT_DIR / "category_maps.json", "w") as file:
    json.dump(category_maps, file, indent=4)

# --------------------------------------------------
# Dataset verification
# --------------------------------------------------

train_dataset = RootVolumeDataset(
    dataframe=train_df,
    image_root=IMAGE_ROOT,
    category_maps=category_maps,
    augment=True,
)

val_dataset = RootVolumeDataset(
    dataframe=val_df,
    image_root=IMAGE_ROOT,
    category_maps=category_maps,
    augment=False,
)

sample = train_dataset[0]
validation_sample = val_dataset[0]

print("=" * 65)
print("DATA PIPELINE PREPARATION")
print("=" * 65)

print(f"Total observations: {len(df)}")
print(f"Training observations: {len(train_dataset)}")
print(f"Validation observations: {len(val_dataset)}")

print(f"\nTraining folders: {len(train_folders)}")
print(f"Validation folders: {len(val_folders)}")
print(f"Shared folders: {len(train_folders & val_folders)}")

print("\nCategory vocabulary sizes:")
for column, mapping in category_maps.items():
    print(f"{column}: {len(mapping)}")

print("\nTraining sample:")
print("Image shape:", tuple(sample["image"].shape))
print("Categorical shape:", tuple(sample["categorical"].shape))
print("Numeric features:", sample["numeric"].tolist())
print("Target:", sample["target"].item())

print("\nValidation sample:")
print("Image shape:", tuple(validation_sample["image"].shape))
print("Target:", validation_sample["target"].item())

print("\nTarget distributions:")
print(train_df["RootVolume"].describe())
print()
print(val_df["RootVolume"].describe())

print("\nSaved:")
print(OUTPUT_DIR / "train_split.csv")
print(OUTPUT_DIR / "validation_split.csv")
print(OUTPUT_DIR / "category_maps.json")

print("\nDATA PREPARATION COMPLETE")
