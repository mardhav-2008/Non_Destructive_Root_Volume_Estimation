import pandas as pd
import numpy as np


YOLO_CSV = "yolo_slice_features.csv"
TRAIN_CSV = "Train.csv"
OUTPUT_CSV = "observation_yolo_profile_features.csv"


yolo = pd.read_csv(YOLO_CSV)
train = pd.read_csv(TRAIN_CSV)


def longest_run(values):
    best = 0
    current = 0

    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return best


def count_peaks(values):
    """
    Count simple local maxima.

    A point is considered a peak when it is greater than or
    equal to both immediate neighbours.
    """
    if len(values) < 3:
        return 0

    count = 0

    for i in range(1, len(values) - 1):
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            if values[i] > 0:
                count += 1

    return count


def extract_profile(row):

    folder = row["FolderName"]
    side = row["Side"]
    start = row["Start"]
    end = row["End"]

    df = (
        yolo[
            (yolo["FolderName"] == folder)
            & (yolo["Side"] == side)
            & (yolo["Slice"] >= start)
            & (yolo["Slice"] <= end)
        ]
        .sort_values("Slice")
        .copy()
    )

    # --------------------------------------------------------
    # Empty case
    # --------------------------------------------------------

    if len(df) == 0:
        return pd.Series(
            {
                "profile_peak_area": 0,
                "profile_peak_slice_norm": 0.5,
                "profile_area_weighted_position": 0.5,
                "profile_area_mean": 0,
                "profile_area_std": 0,
                "profile_area_max": 0,
                "profile_area_min": 0,
                "profile_area_range": 0,
                "profile_area_median": 0,
                "profile_peak_to_mean": 0,
                "profile_peak_to_median": 0,
                "profile_area_first_half": 0,
                "profile_area_second_half": 0,
                "profile_second_first_ratio": 1,
                "profile_peak_count": 0,
                "profile_area_above_mean_fraction": 0,
                "profile_detected_fraction": 0,
            }
        )

    # --------------------------------------------------------
    # Area profile
    # --------------------------------------------------------

    slices = df["Slice"].to_numpy(dtype=float)
    area = df["total_area"].fillna(0).to_numpy(dtype=float)

    n = len(area)

    mean_area = np.mean(area)
    std_area = np.std(area)
    max_area = np.max(area)
    min_area = np.min(area)
    median_area = np.median(area)

    area_range = max_area - min_area

    # --------------------------------------------------------
    # Peak
    # --------------------------------------------------------

    peak_idx = int(np.argmax(area))
    peak_slice = slices[peak_idx]

    if end > start:
        peak_slice_norm = (peak_slice - start) / (end - start)
    else:
        peak_slice_norm = 0.5

    # --------------------------------------------------------
    # Area-weighted centre
    # --------------------------------------------------------

    total_area = np.sum(area)

    if total_area > 0:
        weighted_position = (
            np.sum(((slices - start) / max(end - start, 1)) * area) / total_area
        )
    else:
        weighted_position = 0.5

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    peak_to_mean = max_area / mean_area if mean_area > 0 else 0

    peak_to_median = max_area / median_area if median_area > 0 else 0

    # --------------------------------------------------------
    # First / second half
    # --------------------------------------------------------

    midpoint = (start + end) / 2

    first_half = df[df["Slice"] <= midpoint]["total_area"]
    second_half = df[df["Slice"] > midpoint]["total_area"]

    first_mean = first_half.mean() if len(first_half) else 0

    second_mean = second_half.mean() if len(second_half) else 0

    if first_mean > 0:
        second_first_ratio = second_mean / first_mean
    else:
        second_first_ratio = 1.0

    # --------------------------------------------------------
    # Fraction above mean
    # --------------------------------------------------------

    if n:
        above_mean_fraction = np.mean(area > mean_area)
    else:
        above_mean_fraction = 0

    # --------------------------------------------------------
    # Detected fraction
    # --------------------------------------------------------

    detected_fraction = np.mean(df["object_count"].to_numpy() > 0)

    # --------------------------------------------------------
    # Peaks
    # --------------------------------------------------------

    peak_count = count_peaks(area)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    return pd.Series(
        {
            "profile_peak_area": max_area,
            "profile_peak_slice_norm": peak_slice_norm,
            "profile_area_weighted_position": weighted_position,
            "profile_area_mean": mean_area,
            "profile_area_std": std_area,
            "profile_area_max": max_area,
            "profile_area_min": min_area,
            "profile_area_range": area_range,
            "profile_area_median": median_area,
            "profile_peak_to_mean": peak_to_mean,
            "profile_peak_to_median": peak_to_median,
            "profile_area_first_half": first_mean,
            "profile_area_second_half": second_mean,
            "profile_second_first_ratio": second_first_ratio,
            "profile_peak_count": peak_count,
            "profile_area_above_mean_fraction": above_mean_fraction,
            "profile_detected_fraction": detected_fraction,
        }
    )


# ============================================================
# Extract
# ============================================================

print("=" * 70)
print("Extracting YOLO profile features")
print("=" * 70)

features = train.apply(extract_profile, axis=1)

result = pd.concat([train, features], axis=1)

result.to_csv(OUTPUT_CSV, index=False)

print(f"\nRows:    {len(result)}")
print(f"Columns: {len(result.columns)}")

print(f"\nSaved to: {OUTPUT_CSV}")

print("\nProfile feature summary:")
print(result[[c for c in result.columns if c.startswith("profile_")]].describe().T)
