from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

IMAGE_DIR = PROJECT_ROOT / "data" / "train"
LABEL_DIR = PROJECT_ROOT / "train_labels"

rows = []


def get_polygon_features(polygon):
    """Calculate simple geometric features for one polygon."""

    area = cv2.contourArea(polygon)

    _x, _y, w, h = cv2.boundingRect(polygon)

    moments = cv2.moments(polygon)

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = float(np.mean(polygon[:, 0, 0]))
        cy = float(np.mean(polygon[:, 0, 1]))

    return {
        "area": area,
        "centroid_x": cx,
        "centroid_y": cy,
        "bbox_width": w,
        "bbox_height": h,
    }


# --------------------------------------------------
# Process every label file
# --------------------------------------------------

for label_path in sorted(LABEL_DIR.glob("*.txt")):
    image_name = label_path.stem + ".png"

    # Find which scan this belongs to
    parts = label_path.stem.split("_")

    scan_name = "_".join(parts[:-2])
    side = parts[-2]
    slice_number = int(parts[-1])

    image_path = IMAGE_DIR / scan_name / image_name

    if not image_path.exists():
        print(f"Missing image: {image_path}")
        continue

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        print(f"Could not read: {image_path}")
        continue

    height, width = image.shape[:2]

    polygons = []

    with open(label_path) as f:
        for line in f:
            values = list(map(float, line.split()))

            class_id = int(values[0])

            coords = np.array(values[1:], dtype=np.float32).reshape(-1, 2)

            # Normalized YOLO coordinates -> pixels
            coords[:, 0] *= width
            coords[:, 1] *= height

            polygon = coords.astype(np.int32)

            polygons.append((class_id, polygon))

    # --------------------------------------------------
    # Aggregate polygon statistics for this slice
    # --------------------------------------------------

    features = [get_polygon_features(polygon) for _, polygon in polygons]

    if features:
        total_area = sum(f["area"] for f in features)

        max_area = max(f["area"] for f in features)

        mean_area = np.mean([f["area"] for f in features])

        mean_x = np.mean([f["centroid_x"] for f in features])

        mean_y = np.mean([f["centroid_y"] for f in features])

        max_height = max(f["bbox_height"] for f in features)

        max_width = max(f["bbox_width"] for f in features)

    else:
        total_area = 0
        max_area = 0
        mean_area = 0
        mean_x = 0
        mean_y = 0
        max_height = 0
        max_width = 0

    rows.append(
        {
            "scan": scan_name,
            "side": side,
            "slice": slice_number,
            "num_objects": len(polygons),
            "total_area": total_area,
            "max_area": max_area,
            "mean_area": mean_area,
            "mean_x": mean_x,
            "mean_y": mean_y,
            "max_bbox_width": max_width,
            "max_bbox_height": max_height,
        }
    )


# --------------------------------------------------
# Save
# --------------------------------------------------

df = pd.DataFrame(rows)

df = df.sort_values(["scan", "side", "slice"])

output_path = PROJECT_ROOT / "segmentation_features.csv"

df.to_csv(output_path, index=False)

print()
print(f"Processed {len(df)} labelled slices")
print(f"Saved to: {output_path}")
print()

print(df.head(20).to_string(index=False))
