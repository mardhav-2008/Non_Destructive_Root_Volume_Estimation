from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

TRAIN_DIR = PROJECT_ROOT / "data" / "train"
MODEL_PATH = PROJECT_ROOT / "Models" / "best_full.pt"

OUTPUT_PATH = PROJECT_ROOT / "yolo_detections.csv"


# ============================================================
# Configuration
# ============================================================

CONFIDENCE = 0.10


# ============================================================
# Load model
# ============================================================

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")


# ============================================================
# Extract individual detections
# ============================================================

records = []

image_paths = sorted(TRAIN_DIR.glob("*/*.png"))

print(f"Images found: {len(image_paths)}")


for image_idx, path in enumerate(image_paths, start=1):
    if image_idx % 500 == 0:
        print(
            f"Processed {image_idx}/{len(image_paths)} "
            f"({100 * image_idx / len(image_paths):.1f}%)"
        )

    folder = path.parent.name
    filename = path.stem

    # Expected:
    # FolderName_L_001
    # FolderName_R_001

    parts = filename.split("_")

    if len(parts) < 3:
        continue

    side = parts[-2]
    slice_number = int(parts[-1])

    image = np.array(Image.open(path))

    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    result = model.predict(
        source=image,
        conf=CONFIDENCE,
        verbose=False,
    )[0]

    boxes = result.boxes
    masks = result.masks

    if boxes is None or len(boxes) == 0:
        continue

    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    height, width = image.shape[:2]

    # --------------------------------------------------------
    # Individual detections
    # --------------------------------------------------------

    for object_id in range(len(boxes)):
        confidence = float(boxes.conf[object_id].cpu().numpy())

        # Bounding box:
        # x1, y1, x2, y2

        x1, y1, x2, y2 = boxes.xyxy[object_id].cpu().numpy()

        bbox_width = max(0.0, x2 - x1)
        bbox_height = max(0.0, y2 - y1)

        x_center = (x1 + x2) / 2.0
        y_center = (y1 + y2) / 2.0

        # Normalized horizontal position
        x_center_norm = x_center / width

        # Normalized vertical position
        y_center_norm = y_center / height

        # ----------------------------------------------------
        # Mask area
        # ----------------------------------------------------

        mask_area = np.nan

        if masks is not None and masks.xy is not None:
            polygon = masks.xy[object_id]

            if polygon is not None and len(polygon) >= 3:
                x = polygon[:, 0]
                y = polygon[:, 1]

                # Shoelace formula
                mask_area = 0.5 * abs(
                    np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
                )

        records.append(
            {
                "FolderName": folder,
                "Side": side,
                "Slice": slice_number,
                "ObjectID": object_id,
                "x_center": x_center,
                "x_center_norm": x_center_norm,
                "y_center": y_center,
                "y_center_norm": y_center_norm,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "mask_area": mask_area,
                "confidence": confidence,
            }
        )


# ============================================================
# Save
# ============================================================

detections = pd.DataFrame(records)

detections = detections.sort_values(
    [
        "FolderName",
        "Side",
        "Slice",
        "ObjectID",
    ]
).reset_index(drop=True)


detections.to_csv(
    OUTPUT_PATH,
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Detections: {len(detections):,}")
print(
    f"Images with detections: {detections[['FolderName', 'Side', 'Slice']].drop_duplicates().shape[0]:,}"
)
print(f"Output: {OUTPUT_PATH}")

print()
print(detections.head(20))
