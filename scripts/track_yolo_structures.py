from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = Path("yolo_detections.csv")
OUTPUT_FILE = Path("yolo_tracks.csv")

# Maximum horizontal movement allowed between adjacent slices.
# Since x is normalized to [0, 1], start conservatively.
MAX_X_DISTANCE = 0.08

# A track can survive a missing detection for this many slices.
MAX_GAP = 1

# Minimum number of detections for a track to be retained.
MIN_TRACK_LENGTH = 3


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("YOLO SPATIAL TRACKING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Detections: {len(df):,}")
print(f"Scans: {df[['FolderName', 'Side']].drop_duplicates().shape[0]:,}")


# ============================================================
# TRACK ONE SCAN / SIDE
# ============================================================


def track_scan(group):
    """
    Track detections through consecutive slices.

    Each detection can belong to at most one track.

    Matching is based primarily on horizontal position.
    """

    group = group.sort_values(["Slice", "x_center_norm"]).copy()

    # Active tracks:
    #
    # track_id -> {
    #     "last_slice": int,
    #     "last_x": float,
    # }
    active = {}

    finished = []

    next_track_id = 0

    # Process slices in order
    for slice_idx, slice_df in group.groupby("Slice", sort=True):
        detections = slice_df.to_dict("records")

        # ----------------------------------------------------
        # Match current detections to existing tracks
        # ----------------------------------------------------

        candidates = []

        for det_idx, det in enumerate(detections):
            x = det["x_center_norm"]

            for track_id, track in active.items():
                gap = slice_idx - track["last_slice"]

                if gap < 1 or gap > MAX_GAP + 1:
                    continue

                distance = abs(x - track["last_x"])

                if distance <= MAX_X_DISTANCE:
                    candidates.append(
                        (
                            distance,
                            det_idx,
                            track_id,
                        )
                    )

        # Greedy closest matching
        candidates.sort()

        matched_detections = set()
        matched_tracks = set()

        assignments = {}

        for distance, det_idx, track_id in candidates:
            if det_idx in matched_detections:
                continue

            if track_id in matched_tracks:
                continue

            assignments[det_idx] = track_id

            matched_detections.add(det_idx)
            matched_tracks.add(track_id)

        # ----------------------------------------------------
        # Create new tracks for unmatched detections
        # ----------------------------------------------------

        for det_idx, det in enumerate(detections):
            if det_idx not in assignments:
                assignments[det_idx] = next_track_id

                next_track_id += 1

        # ----------------------------------------------------
        # Update active tracks
        # ----------------------------------------------------

        new_active = {}

        for det_idx, det in enumerate(detections):
            track_id = assignments[det_idx]

            new_active[track_id] = {
                "last_slice": slice_idx,
                "last_x": det["x_center_norm"],
            }

            finished.append(
                {
                    "FolderName": det["FolderName"],
                    "Side": det["Side"],
                    "Slice": det["Slice"],
                    "TrackID": track_id,
                    "x_center": det["x_center"],
                    "x_center_norm": det["x_center_norm"],
                    "bbox_width": det["bbox_width"],
                    "bbox_height": det["bbox_height"],
                    "mask_area": det["mask_area"],
                    "confidence": det["confidence"],
                }
            )

        # Keep tracks alive if they can survive a gap
        for track_id, track in active.items():
            if track_id not in matched_tracks:
                gap = slice_idx - track["last_slice"]

                if gap <= MAX_GAP:
                    new_active[track_id] = track

        active = new_active

    return finished


# ============================================================
# RUN TRACKING
# ============================================================

all_tracks = []

scan_groups = df.groupby(["FolderName", "Side"], sort=False)

total_scans = len(scan_groups)

for i, ((folder, side), group) in enumerate(scan_groups, start=1):
    if i % 10 == 0 or i == 1:
        print(f"Processing {i:3d}/{total_scans}: {folder} {side}")

    result = track_scan(group)

    all_tracks.extend(result)


tracks_df = pd.DataFrame(all_tracks)


# ============================================================
# TRACK LENGTH
# ============================================================

track_lengths = (
    tracks_df.groupby(["FolderName", "Side", "TrackID"])
    .size()
    .rename("track_length")
    .reset_index()
)

tracks_df = tracks_df.merge(
    track_lengths,
    on=["FolderName", "Side", "TrackID"],
    how="left",
)

# Keep only meaningful tracks
tracks_df = tracks_df[tracks_df["track_length"] >= MIN_TRACK_LENGTH].copy()


# ============================================================
# SAVE
# ============================================================

tracks_df.to_csv(OUTPUT_FILE, index=False)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("TRACKING SUMMARY")
print("=" * 70)

print(f"Track detections retained: {len(tracks_df):,}")

print(
    f"Unique tracks: "
    f"{tracks_df[['FolderName', 'Side', 'TrackID']].drop_duplicates().shape[0]:,}"
)

print()
print("Track length statistics:")

print(
    tracks_df.drop_duplicates(["FolderName", "Side", "TrackID"])[
        "track_length"
    ].describe()
)

print()
print("Longest tracks:")

longest = (
    tracks_df[["FolderName", "Side", "TrackID", "track_length"]]
    .drop_duplicates()
    .sort_values("track_length", ascending=False)
    .head(20)
)

print(longest.to_string(index=False))

print()
print(f"Saved to: {OUTPUT_FILE}")
