from pathlib import Path

import numpy as np
import pandas as pd


TRACK_FILE = Path("yolo_tracks.csv")
TRAIN_FILE = Path("Train.csv")


print("=" * 70)
print("YOLO TRACK ANALYSIS")
print("=" * 70)

tracks = pd.read_csv(TRACK_FILE)
train = pd.read_csv(TRAIN_FILE)

print(f"Track detections: {len(tracks):,}")

# ------------------------------------------------------------
# One row per track
# ------------------------------------------------------------

track_summary = (
    tracks.groupby(["FolderName", "Side", "TrackID"])
    .agg(
        track_length=("Slice", "count"),
        x_mean=("x_center_norm", "mean"),
        x_std=("x_center_norm", "std"),
        x_min=("x_center_norm", "min"),
        x_max=("x_center_norm", "max"),
        area_sum=("mask_area", "sum"),
        area_mean=("mask_area", "mean"),
        area_max=("mask_area", "max"),
        confidence_mean=("confidence", "mean"),
        confidence_max=("confidence", "max"),
        first_slice=("Slice", "min"),
        last_slice=("Slice", "max"),
    )
    .reset_index()
)

track_summary["x_range"] = track_summary["x_max"] - track_summary["x_min"]

track_summary["depth_span"] = (
    track_summary["last_slice"] - track_summary["first_slice"] + 1
)

track_summary["continuity"] = track_summary["track_length"] / track_summary[
    "depth_span"
].clip(lower=1)


# ============================================================
# 1. TRACK STATISTICS
# ============================================================

print()
print("=" * 70)
print("1. TRACK STATISTICS")
print("=" * 70)

print(track_summary["track_length"].describe())

print()
print("Continuity:")
print(track_summary["continuity"].describe())


# ============================================================
# 2. TRACKS PER SCAN
# ============================================================

tracks_per_scan = track_summary.groupby(["FolderName", "Side"]).size()

print()
print("=" * 70)
print("2. TRACKS PER SCAN")
print("=" * 70)

print(tracks_per_scan.describe())


# ============================================================
# 3. LONGEST TRACKS
# ============================================================

print()
print("=" * 70)
print("3. LONGEST TRACKS")
print("=" * 70)

print(
    track_summary.sort_values("track_length", ascending=False)
    .head(30)
    .to_string(index=False)
)


# ============================================================
# 4. TRACK-LEVEL AREA RELATIONSHIPS
# ============================================================

print()
print("=" * 70)
print("4. TRACK FEATURE CORRELATIONS")
print("=" * 70)

for col in [
    "track_length",
    "x_mean",
    "x_std",
    "x_range",
    "area_sum",
    "area_mean",
    "area_max",
    "confidence_mean",
    "confidence_max",
    "continuity",
]:
    print(
        f"{col:20s}: "
        f"mean={track_summary[col].mean():.4f}, "
        f"median={track_summary[col].median():.4f}"
    )


# ============================================================
# 5. OBSERVATION-LEVEL FEATURES
# ============================================================

# Attach RootVolume to scan/side.
obs = train[
    [
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
].copy()


# Only tracks that overlap the observation interval
merged = tracks.merge(
    obs[
        [
            "ID",
            "FolderName",
            "Side",
            "Start",
            "End",
            "RootVolume",
        ]
    ],
    on=["FolderName", "Side"],
    how="inner",
)

merged = merged[
    (merged["Slice"] >= merged["Start"]) & (merged["Slice"] <= merged["End"])
].copy()


# Aggregate track information inside each observation
obs_features = (
    merged.groupby("ID")
    .agg(
        track_count=("TrackID", "nunique"),
        total_track_detections=("TrackID", "count"),
        mean_track_length=("track_length", "mean"),
        max_track_length=("track_length", "max"),
        mean_track_area=("mask_area", "mean"),
        total_track_area=("mask_area", "sum"),
        max_track_area=("mask_area", "max"),
        mean_x=("x_center_norm", "mean"),
        x_std=("x_center_norm", "std"),
        mean_confidence=("confidence", "mean"),
    )
    .reset_index()
)

obs_features = obs_features.merge(
    obs[["ID", "RootVolume"]],
    on="ID",
    how="left",
)


# ============================================================
# 6. CORRELATION WITH ROOT VOLUME
# ============================================================

print()
print("=" * 70)
print("6. TRACK FEATURES VS ROOT VOLUME")
print("=" * 70)

feature_cols = [
    "track_count",
    "total_track_detections",
    "mean_track_length",
    "max_track_length",
    "mean_track_area",
    "total_track_area",
    "max_track_area",
    "mean_x",
    "x_std",
    "mean_confidence",
]

for col in feature_cols:
    valid = obs_features[[col, "RootVolume"]].dropna()

    if len(valid) < 3:
        continue

    corr = valid[col].corr(valid["RootVolume"])

    print(f"{col:25s}: {corr:+.4f}")


# ============================================================
# 7. TRACK COUNT / AREA BY ROOT VOLUME
# ============================================================

print()
print("=" * 70)
print("7. EXAMPLE OBSERVATIONS")
print("=" * 70)

print(
    obs_features.sort_values("RootVolume", ascending=False)
    .head(15)
    .to_string(index=False)
)

print()
print("Analysis complete.")
