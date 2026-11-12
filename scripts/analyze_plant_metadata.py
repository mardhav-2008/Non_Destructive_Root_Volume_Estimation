from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT / "Train.csv")
audit = pd.read_csv(ROOT / "outputs/audit/train_image_audit.csv")

if len(train) != len(audit):
    raise ValueError("Train and audit row counts do not match.")

df = train.copy()
df["SequenceHash"] = audit["SequenceHash"].values

# --------------------------------------------------
# 1. Variation within identical image sequences
# --------------------------------------------------

groups = df.groupby("SequenceHash").agg(
    observations=("RootVolume", "size"),
    unique_plants=("PlantNumber", "nunique"),
    unique_volumes=("RootVolume", "nunique"),
    volume_mean=("RootVolume", "mean"),
    volume_std=("RootVolume", "std"),
    volume_min=("RootVolume", "min"),
    volume_max=("RootVolume", "max"),
    folder_count=("FolderName", "nunique"),
)

repeated = groups[groups["observations"] > 1]
conflicting = repeated[repeated["unique_volumes"] > 1]

print("\n" + "=" * 65)
print("IDENTICAL IMAGE SEQUENCE ANALYSIS")
print("=" * 65)

print(f"Unique sequences: {len(groups)}")
print(f"Repeated sequences: {len(repeated)}")
print(f"Repeated sequences with differing labels: {len(conflicting)}")
print(f"Mean observations per repeated sequence: {repeated['observations'].mean():.2f}")
print(
    f"Mean RootVolume std within repeated sequences: {conflicting['volume_std'].mean():.4f}"
)
print(
    f"Mean RootVolume range within repeated sequences: "
    f"{(conflicting['volume_max'] - conflicting['volume_min']).mean():.4f}"
)

print("\nLargest within-sequence RootVolume variation:")
print(conflicting.sort_values("volume_std", ascending=False).head(10).to_string())

# --------------------------------------------------
# 2. RootVolume distribution by PlantNumber
# --------------------------------------------------

plant_stats = df.groupby("PlantNumber")["RootVolume"].agg(
    ["count", "mean", "std", "median", "min", "max"]
)

print("\n" + "=" * 65)
print("ROOT VOLUME BY PLANT NUMBER")
print("=" * 65)
print(plant_stats.to_string())

# --------------------------------------------------
# 3. PlantNumber distribution by folder
# --------------------------------------------------

plant_folder = df.groupby(["FolderName", "PlantNumber"]).size().unstack(fill_value=0)

print("\n" + "=" * 65)
print("PLANT NUMBER COVERAGE")
print("=" * 65)
print(f"Folders: {df['FolderName'].nunique()}")
print(f"Plant numbers: {sorted(df['PlantNumber'].unique())}")
print("\nObservations per PlantNumber:")
print(df["PlantNumber"].value_counts().sort_index().to_string())

print("\nNumber of folders containing each PlantNumber:")
print((plant_folder > 0).sum().to_string())

# --------------------------------------------------
# 4. Save detailed results
# --------------------------------------------------

out = ROOT / "outputs" / "audit"
out.mkdir(parents=True, exist_ok=True)

groups.to_csv(out / "sequence_volume_statistics.csv")
plant_stats.to_csv(out / "plant_number_volume_statistics.csv")
plant_folder.to_csv(out / "plant_number_folder_coverage.csv")

print(f"\nResults saved to: {out}")
