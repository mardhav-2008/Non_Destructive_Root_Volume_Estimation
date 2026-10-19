from pathlib import Path

import numpy as np  # noqa: F401
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_PATH = PROJECT_ROOT / "Train.csv"
FEATURE_PATH = PROJECT_ROOT / "segmentation_features.csv"

train = pd.read_csv(TRAIN_PATH)
features = pd.read_csv(FEATURE_PATH)


# --------------------------------------------------
# Aggregate segmentation features for each
# train observation
# --------------------------------------------------

rows = []

for _, obs in train.iterrows():
    scan = obs["FolderName"]
    side = obs["Side"]
    start = obs["Start"]
    end = obs["End"]

    # Relevant labelled slices
    subset = features[
        (features["scan"] == scan)
        & (features["side"] == side)
        & (features["slice"] >= start)
        & (features["slice"] <= end)
    ].copy()

    # How many slices were actually labelled?
    expected_slices = end - start + 1
    labelled_slices = len(subset)

    if labelled_slices == 0:
        rows.append(
            {
                "ID": obs["ID"],
                "RootVolume": obs["RootVolume"],
                "scan": scan,
                "side": side,
                "start": start,
                "end": end,
                "expected_slices": expected_slices,
                "labelled_slices": 0,
                "label_coverage": 0.0,
            }
        )

        continue

    # --------------------------------------------------
    # Aggregate features
    # --------------------------------------------------

    areas = subset["total_area"]

    rows.append(
        {
            "ID": obs["ID"],
            "RootVolume": obs["RootVolume"],
            "scan": scan,
            "side": side,
            "start": start,
            "end": end,
            "expected_slices": expected_slices,
            "labelled_slices": labelled_slices,
            # Coverage
            "label_coverage": (labelled_slices / expected_slices),
            # Object counts
            "total_objects": subset["num_objects"].sum(),
            "mean_objects": subset["num_objects"].mean(),
            "max_objects": subset["num_objects"].max(),
            # Area
            "total_area": areas.sum(),
            "mean_area": areas.mean(),
            "max_area": areas.max(),
            "std_area": areas.std(),
            # Spatial position
            "mean_x": subset["mean_x"].mean(),
            "mean_y": subset["mean_y"].mean(),
            # Bounding boxes
            "mean_bbox_width": (subset["max_bbox_width"].mean()),
            "max_bbox_width": (subset["max_bbox_width"].max()),
            "mean_bbox_height": (subset["max_bbox_height"].mean()),
            "max_bbox_height": (subset["max_bbox_height"].max()),
        }
    )


result = pd.DataFrame(rows)


# --------------------------------------------------
# Save
# --------------------------------------------------

output_path = PROJECT_ROOT / "observation_segmentation_features.csv"

result.to_csv(output_path, index=False)


print(f"Saved {len(result)} observations to {output_path}")

print()
print(
    "Observations with at least one labelled slice:",
    (result["labelled_slices"] > 0).sum(),
)

print(
    "Observations with complete label coverage:",
    (result["labelled_slices"] == result["expected_slices"]).sum(),
)

print()
print(result.head(10).to_string(index=False))
