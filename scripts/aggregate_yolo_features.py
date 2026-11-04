import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "Train.csv"
YOLO_CSV = "yolo_slice_features.csv"
OUTPUT_CSV = "observation_yolo_features.csv"


# ============================================================
# Load data
# ============================================================

train = pd.read_csv(TRAIN_CSV)
yolo = pd.read_csv(YOLO_CSV)

print("=" * 70)
print("Loaded data")
print("=" * 70)

print(f"Train observations: {len(train)}")
print(f"YOLO slice rows:    {len(yolo)}")


# ============================================================
# Aggregate one observation
# ============================================================


def aggregate_observation(row):
    folder = row["FolderName"]
    side = row["Side"]
    start = row["Start"]
    end = row["End"]

    # Exact observation interval
    mask = (
        (yolo["FolderName"] == folder)
        & (yolo["Side"] == side)
        & (yolo["Slice"] >= start)
        & (yolo["Slice"] <= end)
    )

    df = yolo.loc[mask].sort_values("Slice").copy()

    # Every image was processed by YOLO, so an empty interval
    # should be extremely unlikely.
    if len(df) == 0:
        return pd.Series(
            {
                "yolo_n_slices": 0,
                "yolo_detected_slices": 0,
                "yolo_detection_coverage": 0.0,
                "yolo_objects_mean": 0.0,
                "yolo_objects_std": 0.0,
                "yolo_objects_min": 0.0,
                "yolo_objects_max": 0.0,
                "yolo_objects_median": 0.0,
                "yolo_area_mean": 0.0,
                "yolo_area_std": 0.0,
                "yolo_area_min": 0.0,
                "yolo_area_max": 0.0,
                "yolo_area_median": 0.0,
                "yolo_max_area_mean": 0.0,
                "yolo_max_area_max": 0.0,
                "yolo_width_mean": 0.0,
                "yolo_width_std": 0.0,
                "yolo_width_max": 0.0,
                "yolo_height_mean": 0.0,
                "yolo_height_std": 0.0,
                "yolo_height_max": 0.0,
                "yolo_conf_mean": 0.0,
                "yolo_conf_max": 0.0,
                "yolo_longest_detection_run": 0,
                "yolo_first_detection": np.nan,
                "yolo_last_detection": np.nan,
                "yolo_area_trend": 0.0,
            }
        )

    # --------------------------------------------------------
    # Basic detection information
    # --------------------------------------------------------

    detected = df["object_count"] > 0

    n_slices = len(df)
    n_detected = int(detected.sum())

    coverage = n_detected / n_slices if n_slices else 0.0

    # --------------------------------------------------------
    # Longest consecutive detection run
    # --------------------------------------------------------

    longest_run = 0
    current_run = 0

    for value in detected:
        if value:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    # --------------------------------------------------------
    # First / last detected slice
    # --------------------------------------------------------

    detected_slices = df.loc[detected, "Slice"]

    if len(detected_slices):
        first_detection = detected_slices.min()
        last_detection = detected_slices.max()
    else:
        first_detection = np.nan
        last_detection = np.nan

    # --------------------------------------------------------
    # Area trend
    #
    # Linear slope of total mask area against slice index.
    # Only use detected slices.
    # --------------------------------------------------------

    trend_df = df.loc[detected, ["Slice", "total_area"]].dropna()

    if len(trend_df) >= 2 and trend_df["Slice"].nunique() >= 2:
        x = trend_df["Slice"].to_numpy()
        y = trend_df["total_area"].to_numpy()

        area_trend = np.polyfit(x, y, 1)[0]
    else:
        area_trend = 0.0

    # --------------------------------------------------------
    # Helper functions
    # --------------------------------------------------------

    def stats(column):
        values = df[column].fillna(0)

        return {
            f"{column}_mean": values.mean(),
            f"{column}_std": values.std(ddof=0),
            f"{column}_min": values.min(),
            f"{column}_max": values.max(),
            f"{column}_median": values.median(),
        }

    # --------------------------------------------------------
    # Construct result
    # --------------------------------------------------------

    result = {
        "yolo_n_slices": n_slices,
        "yolo_detected_slices": n_detected,
        "yolo_detection_coverage": coverage,
        "yolo_objects_mean": df["object_count"].mean(),
        "yolo_objects_std": df["object_count"].std(ddof=0),
        "yolo_objects_min": df["object_count"].min(),
        "yolo_objects_max": df["object_count"].max(),
        "yolo_objects_median": df["object_count"].median(),
        "yolo_area_mean": df["total_area"].mean(),
        "yolo_area_std": df["total_area"].std(ddof=0),
        "yolo_area_min": df["total_area"].min(),
        "yolo_area_max": df["total_area"].max(),
        "yolo_area_median": df["total_area"].median(),
        "yolo_max_area_mean": df["max_area"].mean(),
        "yolo_max_area_max": df["max_area"].max(),
        "yolo_width_mean": df["mean_width"].mean(),
        "yolo_width_std": df["mean_width"].std(ddof=0),
        "yolo_width_max": df["max_width"].max(),
        "yolo_height_mean": df["mean_height"].mean(),
        "yolo_height_std": df["mean_height"].std(ddof=0),
        "yolo_height_max": df["max_height"].max(),
        "yolo_conf_mean": df["mean_confidence"].mean(),
        "yolo_conf_max": df["max_confidence"].max(),
        "yolo_longest_detection_run": longest_run,
        "yolo_first_detection": first_detection,
        "yolo_last_detection": last_detection,
        "yolo_area_trend": area_trend,
    }

    return pd.Series(result)


# ============================================================
# Run aggregation
# ============================================================

print("\nAggregating YOLO features...")

features = train.apply(aggregate_observation, axis=1)

# Keep the original training metadata + target
result = pd.concat([train, features], axis=1)

result.to_csv(OUTPUT_CSV, index=False)


# ============================================================
# Diagnostics
# ============================================================

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)

print(f"Rows:    {len(result)}")
print(f"Columns: {len(result.columns)}")

print("\nDetection coverage:")
print(result["yolo_detection_coverage"].describe())

print(
    "\nObservations with at least one detection:",
    (result["yolo_detected_slices"] > 0).sum(),
)

print("Observations with zero detections:", (result["yolo_detected_slices"] == 0).sum())

print("\nSaved to:")
print(OUTPUT_CSV)
