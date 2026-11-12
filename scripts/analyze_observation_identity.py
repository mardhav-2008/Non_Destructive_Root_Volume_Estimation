from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT / "Train.csv")
audit = pd.read_csv(ROOT / "outputs/audit/train_image_audit.csv")

df = train.copy()
df["SequenceHash"] = audit["SequenceHash"].values


def report_grouping(name, keys):
    grouped = df.groupby(keys, dropna=False)["RootVolume"].agg(
        observations="size",
        unique_labels="nunique",
        mean_volume="mean",
        std_volume="std",
        min_volume="min",
        max_volume="max",
    )

    repeated = grouped[grouped["observations"] > 1]
    conflicting = repeated[repeated["unique_labels"] > 1]

    print(f"\n{name}")
    print("-" * 65)
    print(f"Unique groups: {len(grouped)}")
    print(f"Repeated groups: {len(repeated)}")
    print(f"Repeated groups with conflicting labels: {len(conflicting)}")

    if len(conflicting):
        print("\nLargest conflicts:")
        print(
            conflicting.sort_values("std_volume", ascending=False).head(10).to_string()
        )

    return grouped, conflicting


print("=" * 70)
print("OBSERVATION IDENTITY ANALYSIS")
print("=" * 70)

# Same pixels, same plant?
image_plant, image_plant_conflicts = report_grouping(
    "IMAGE SEQUENCE + PLANT NUMBER",
    ["SequenceHash", "PlantNumber"],
)

# Same scan, plant, side and interval?
metadata_groups, metadata_conflicts = report_grouping(
    "FOLDER + PLANT + SIDE + INTERVAL",
    ["FolderName", "PlantNumber", "Side", "Start", "End"],
)

# Same image, plant and experimental metadata?
full_groups, full_conflicts = report_grouping(
    "IMAGE + PLANT + GENOTYPE + STAGE",
    ["SequenceHash", "PlantNumber", "Genotype", "Stage"],
)

# How much does PlantNumber vary within folders?
plant_counts = df.groupby("FolderName")["PlantNumber"].nunique()

print("\nPLANT COVERAGE PER FOLDER")
print("-" * 65)
print(plant_counts.describe().to_string())

print("\nFolders with multiple plants:", (plant_counts > 1).sum())

# Save detailed identity tables.
out = ROOT / "outputs" / "audit"
out.mkdir(parents=True, exist_ok=True)

image_plant.to_csv(out / "image_plant_identity.csv")
metadata_groups.to_csv(out / "metadata_identity.csv")
full_groups.to_csv(out / "full_identity.csv")

print(f"\nSaved identity reports to: {out}")
