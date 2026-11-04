from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.engine.results import Results


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
MODEL_PATH = PROJECT_ROOT / "Models" / "best_full.pt"

OUTPUT_PATH = PROJECT_ROOT / "yolo_slice_features.csv"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.10


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLO FEATURE EXTRACTION")
print("=" * 70)

print("Loading model...")

model = YOLO(str(MODEL_PATH))

print("Model:", MODEL_PATH.name)
print("Task:", model.task)
print("Classes:", model.names)
print("Confidence:", CONFIDENCE)


# ============================================================
# FIND IMAGES
# ============================================================

image_paths = sorted(TRAIN_DIR.glob("*/*.png"))

print()
print("Training images:", len(image_paths))


# ============================================================
# EXTRACT FEATURES
# ============================================================

rows = []

for index, image_path in enumerate(image_paths):
    if index % 250 == 0:
        print(f"Processing {index}/{len(image_paths)}")

    # --------------------------------------------------------
    # Parse filename
    #
    # Example:
    # A2miww5mfx_L_010.png
    # --------------------------------------------------------

    stem = image_path.stem

    parts = stem.split("_")

    if len(parts) != 3:
        print("WARNING: unexpected filename:", stem)
        continue

    folder = parts[0]
    side = parts[1]
    slice_number = int(parts[2])

    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        verbose=False,
    )

    result = cast(
        Results,
        results[0],
    )

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        rows.append(
            {
                "FolderName": folder,
                "Side": side,
                "Slice": slice_number,
                "object_count": 0,
                "total_area": 0.0,
                "mean_area": 0.0,
                "max_area": 0.0,
                "std_area": 0.0,
                "mean_x": np.nan,
                "std_x": np.nan,
                "mean_y": np.nan,
                "std_y": np.nan,
                "mean_width": 0.0,
                "max_width": 0.0,
                "std_width": 0.0,
                "mean_height": 0.0,
                "max_height": 0.0,
                "std_height": 0.0,
                "mean_confidence": 0.0,
                "max_confidence": 0.0,
            }
        )

        continue

    # --------------------------------------------------------
    # Extract confidence
    # --------------------------------------------------------

    confidences = result.boxes.conf.cpu().numpy()

    # --------------------------------------------------------
    # Extract bounding boxes
    #
    # xyxy = x1, y1, x2, y2
    # --------------------------------------------------------

    boxes = result.boxes.xyxy.cpu().numpy()

    widths = boxes[:, 2] - boxes[:, 0]

    heights = boxes[:, 3] - boxes[:, 1]

    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2

    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

    # --------------------------------------------------------
    # Mask areas
    # --------------------------------------------------------

    areas = []

    for polygon in result.masks.xy:
        # Shoelace formula
        x = polygon[:, 0]
        y = polygon[:, 1]

        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        areas.append(area)

    areas = np.asarray(areas, dtype=float)

    # --------------------------------------------------------
    # Store features
    # --------------------------------------------------------

    rows.append(
        {
            "FolderName": folder,
            "Side": side,
            "Slice": slice_number,
            "object_count": len(areas),
            "total_area": areas.sum(),
            "mean_area": areas.mean(),
            "max_area": areas.max(),
            "std_area": areas.std(),
            "mean_x": centers_x.mean(),
            "std_x": centers_x.std(),
            "mean_y": centers_y.mean(),
            "std_y": centers_y.std(),
            "mean_width": widths.mean(),
            "max_width": widths.max(),
            "std_width": widths.std(),
            "mean_height": heights.mean(),
            "max_height": heights.max(),
            "std_height": heights.std(),
            "mean_confidence": confidences.mean(),
            "max_confidence": confidences.max(),
        }
    )


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(rows)

df = df.sort_values(
    [
        "FolderName",
        "Side",
        "Slice",
    ]
)

df.to_csv(OUTPUT_PATH, index=False)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("DONE")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print()
print("Images with detections:", (df["object_count"] > 0).sum())

print("Images without detections:", (df["object_count"] == 0).sum())

print()
print("Mean objects per slice:", df["object_count"].mean())

print("Mean total mask area:", df["total_area"].mean())

print()
print("Saved:")
print(OUTPUT_PATH)
