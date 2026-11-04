from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DETECTIONS_PATH = PROJECT_ROOT / "yolo_detections.csv"
TRAIN_PATH = PROJECT_ROOT / "Train.csv"


detections = pd.read_csv(DETECTIONS_PATH)
train = pd.read_csv(TRAIN_PATH)


print("=" * 70)
print("YOLO SPATIAL STRUCTURE ANALYSIS")
print("=" * 70)

print()
print("Detections:", len(detections))
print("Unique scans:", detections["FolderName"].nunique())


# ============================================================
# 1. Global x distribution
# ============================================================

print()
print("=" * 70)
print("1. GLOBAL X POSITION")
print("=" * 70)

x = detections["x_center_norm"]

print(x.describe())


# ============================================================
# 2. Quantiles
# ============================================================

print()
print("X POSITION QUANTILES")

for q in np.linspace(0, 1, 11):
    print(
        f"{q:4.1f}: "
        f"{x.quantile(q):.4f}"
    )


# ============================================================
# 3. KMeans on horizontal positions
# ============================================================

print()
print("=" * 70)
print("2. HORIZONTAL CLUSTERING")
print("=" * 70)

for k in range(2, 11):

    values = detections[
        ["x_center_norm"]
    ].dropna().values

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20,
    )

    model.fit(values)

    centers = np.sort(
        model.cluster_centers_.flatten()
    )

    print(
        f"k={k:2d}: "
        + ", ".join(
            f"{c:.3f}"
            for c in centers
        )
    )


# ============================================================
# 4. Spatial distribution per scan
# ============================================================

print()
print("=" * 70)
print("3. EXAMPLE SCANS")
print("=" * 70)

scan_counts = (
    detections
    .groupby(["FolderName", "Side"])
    .size()
    .sort_values(ascending=False)
)

print()
print("Most detection-rich scan/side combinations:")

print(
    scan_counts.head(20)
)


# ============================================================
# 5. Print x positions for example scans
# ============================================================

print()

examples = (
    scan_counts
    .head(10)
    .index
)

for folder, side in examples:

    subset = detections[
        (detections["FolderName"] == folder)
        & (detections["Side"] == side)
    ]

    print()
    print("-" * 70)
    print(folder, side)

    print(
        "detections:",
        len(subset)
    )

    print(
        "x positions:"
    )

    print(
        np.round(
            subset["x_center_norm"].values,
            3,
        )
    )


# ============================================================
# 6. Compare PlantNumber against spatial positions
# ============================================================

print()
print("=" * 70)
print("4. PLANT NUMBER VS SPATIAL POSITION")
print("=" * 70)

# For this analysis we need the observation's
# FolderName + Side + Start + End.
#
# Every detection whose slice lies inside an observation
# interval is assigned to that observation.

merged = detections.merge(
    train[
        [
            "FolderName",
            "PlantNumber",
            "Side",
            "Start",
            "End",
            "RootVolume",
            "Genotype",
        ]
    ],
    on=["FolderName", "Side"],
    how="inner",
)


merged = merged[
    (merged["Slice"] >= merged["Start"])
    & (merged["Slice"] <= merged["End"])
].copy()


print()
print(
    "Detection-observation matches:",
    len(merged)
)


if len(merged) > 0:

    grouped = (
        merged
        .groupby("PlantNumber")["x_center_norm"]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        )
    )

    print()
    print(grouped)


# ============================================================
# 7. Per-folder plant ordering
# ============================================================

print()
print("=" * 70)
print("5. WITHIN-SCAN PLANT ORDERING")
print("=" * 70)

# For every FolderName + Side + PlantNumber,
# calculate median detection x position.

plant_positions = (
    merged
    .groupby(
        [
            "FolderName",
            "Side",
            "PlantNumber",
        ]
    )
    .agg(
        x_median=(
            "x_center_norm",
            "median",
        ),
        x_mean=(
            "x_center_norm",
            "mean",
        ),
        detections=(
            "x_center_norm",
            "count",
        ),
    )
    .reset_index()
)


# Only retain plant groups with at least
# two detections.

plant_positions = plant_positions[
    plant_positions["detections"] >= 2
].copy()


print()
print(
    "Plant groups with >=2 detections:",
    len(plant_positions)
)


# ============================================================
# 8. Does PlantNumber increase with x?
# ============================================================

print()
print("Per-scan ordering:")

ordering_results = []


for (folder, side), group in plant_positions.groupby(
    ["FolderName", "Side"]
):

    if group["PlantNumber"].nunique() < 2:
        continue

    ordered = group.sort_values(
        "x_median"
    )

    plant_order = ordered[
        "PlantNumber"
    ].tolist()

    x_order = ordered[
        "x_median"
    ].tolist()

    # Spearman-like rank correlation
    if len(plant_order) >= 2:

        rank_corr = pd.Series(
            plant_order
        ).corr(
            pd.Series(x_order),
            method="spearman",
        )

        ordering_results.append(
            {
                "FolderName": folder,
                "Side": side,
                "n_plants": len(group),
                "rank_corr": rank_corr,
                "plant_order": plant_order,
            }
        )


ordering = pd.DataFrame(
    ordering_results
)


print()

if len(ordering) > 0:

    print(
        ordering[
            [
                "FolderName",
                "Side",
                "n_plants",
                "rank_corr",
                "plant_order",
            ]
        ].head(30).to_string(index=False)
    )

    print()
    print(
        "Mean rank correlation:",
        ordering["rank_corr"].mean()
    )

    print(
        "Median rank correlation:",
        ordering["rank_corr"].median()
    )

    print(
        "Fraction positive:",
        (
            ordering["rank_corr"] > 0
        ).mean()
    )

else:

    print(
        "Not enough spatially matched "
        "plant observations."
    )


# ============================================================
# 9. RootVolume vs x-position
# ============================================================

print()
print("=" * 70)
print("6. ROOT VOLUME VS DETECTION POSITION")
print("=" * 70)

if len(merged) > 0:

    obs_position = (
        merged
        .groupby(
            [
                "FolderName",
                "Side",
                "PlantNumber",
                "Start",
                "End",
            ]
        )
        .agg(
            x_mean=(
                "x_center_norm",
                "mean",
            ),
            x_median=(
                "x_center_norm",
                "median",
            ),
            x_std=(
                "x_center_norm",
                "std",
            ),
            detection_count=(
                "x_center_norm",
                "count",
            ),
            RootVolume=(
                "RootVolume",
                "first",
            ),
        )
        .reset_index()
    )

    print()
    print(
        obs_position[
            [
                "x_mean",
                "x_median",
                "x_std",
                "detection_count",
                "RootVolume",
            ]
        ].corr()["RootVolume"]
    )


# ============================================================
# Done
# ============================================================

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
