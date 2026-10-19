from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

path = PROJECT_ROOT / "observation_segmentation_features.csv"

df = pd.read_csv(path)


# --------------------------------------------------
# Only observations with segmentation
# --------------------------------------------------

labelled = df[df["labelled_slices"] > 0].copy()

complete = df[df["label_coverage"] == 1].copy()


print(f"All training observations: {len(df)}")
print(f"With labelled slices:      {len(labelled)}")
print(f"Complete coverage:         {len(complete)}")


# --------------------------------------------------
# Correlations
# --------------------------------------------------

features = [
    "mean_area",
    "max_area",
    "mean_objects",
    "max_objects",
    "mean_bbox_width",
    "max_bbox_width",
    "mean_bbox_height",
    "max_bbox_height",
]


print("\nCorrelation with RootVolume")
print("=" * 40)

for feature in features:
    subset = labelled[[feature, "RootVolume"]].dropna()

    if len(subset) < 3:
        continue

    correlation = subset[feature].corr(subset["RootVolume"])

    print(f"{feature:20s} {correlation: .3f} (n={len(subset)})")


# --------------------------------------------------
# Plot the strongest simple feature:
# mean segmentation area
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(labelled["mean_area"], labelled["RootVolume"])

plt.xlabel("Mean segmentation area")
plt.ylabel("RootVolume")

plt.title("RootVolume vs mean segmentation area")

plt.grid(True)

plt.show()


# --------------------------------------------------
# Same thing, complete coverage only
# --------------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(complete["mean_area"], complete["RootVolume"])

plt.xlabel("Mean segmentation area")
plt.ylabel("RootVolume")

plt.title("Complete coverage only")

plt.grid(True)

plt.show()
