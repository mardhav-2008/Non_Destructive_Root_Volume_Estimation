from pathlib import Path
from typing import cast

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO
from ultralytics.engine.results import Results


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "Models"
LABEL_DIR = PROJECT_ROOT / "train_labels"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"


MODEL_NAMES = [
    "best_early.pt",
    "best_full.pt",
    "best_late.pt",
]

CONFIDENCE = 0.10


# ============================================================
# Polygon -> binary mask
# ============================================================


def polygon_to_mask(polygon, width, height):

    mask = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    polygon = np.asarray(
        polygon,
        dtype=np.int32,
    )

    cv2.fillPoly(
        mask,
        [polygon],
        1,
    )

    return mask


# ============================================================
# IoU
# ============================================================


def calculate_iou(mask_a, mask_b):

    intersection = np.logical_and(
        mask_a,
        mask_b,
    ).sum()

    union = np.logical_or(
        mask_a,
        mask_b,
    ).sum()

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# Read YOLO ground truth
# ============================================================


def load_ground_truth(label_path, width, height):

    polygons = []

    with open(label_path, "r") as f:
        for line in f:
            values = list(map(float, line.split()))

            # First value is class ID
            coordinates = values[1:]

            points = []

            for i in range(
                0,
                len(coordinates),
                2,
            ):
                x = coordinates[i] * width
                y = coordinates[i + 1] * height

                points.append([x, y])

            polygons.append(points)

    return polygons


# ============================================================
# Find labelled images
# ============================================================

label_files = sorted(LABEL_DIR.glob("*.txt"))

print("=" * 70)
print("SEGMENTATION MODEL VALIDATION")
print("=" * 70)

print(
    "Label files:",
    len(label_files),
)

print(
    "Confidence:",
    CONFIDENCE,
)


# ============================================================
# Load models
# ============================================================

models = {}

for model_name in MODEL_NAMES:
    print()
    print(
        "Loading:",
        model_name,
    )

    models[model_name] = YOLO(str(MODEL_DIR / model_name))


# ============================================================
# Evaluate
# ============================================================

all_results = []


for label_path in label_files:
    image_name = label_path.stem + ".png"

    # Search train directories for image.
    matches = list(TRAIN_DIR.glob(f"*/{image_name}"))

    if not matches:
        print(
            "WARNING: image not found:",
            image_name,
        )

        continue

    image_path = matches[0]

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        print(
            "WARNING: could not read:",
            image_path,
        )

        continue

    height, width = image.shape

    # --------------------------------------------------------
    # Ground truth masks
    # --------------------------------------------------------

    gt_polygons = load_ground_truth(
        label_path,
        width,
        height,
    )

    gt_masks = [
        polygon_to_mask(
            polygon,
            width,
            height,
        )
        for polygon in gt_polygons
    ]

    # --------------------------------------------------------
    # Evaluate every model
    # --------------------------------------------------------

    for model_name, model in models.items():
        results = model.predict(
            source=str(image_path),
            conf=CONFIDENCE,
            verbose=False,
        )

        result = cast(
            Results,
            results[0],
        )

        if result.masks is None:
            predicted_masks = []

        else:
            predicted_masks = []

            for polygon in result.masks.xy:
                predicted_masks.append(
                    polygon_to_mask(
                        polygon,
                        width,
                        height,
                    )
                )

        # ----------------------------------------------------
        # Match predictions to GT using best IoU
        # ----------------------------------------------------

        matched_ious = []

        for gt_mask in gt_masks:
            if not predicted_masks:
                matched_ious.append(0.0)

                continue

            ious = [
                calculate_iou(
                    gt_mask,
                    pred_mask,
                )
                for pred_mask in predicted_masks
            ]

            matched_ious.append(max(ious))

        mean_iou = np.mean(matched_ious) if matched_ious else 0.0

        all_results.append(
            {
                "image": image_name,
                "model": model_name,
                "ground_truth_count": len(gt_masks),
                "prediction_count": len(predicted_masks),
                "mean_best_iou": mean_iou,
            }
        )


# ============================================================
# Results
# ============================================================

results_df = pd.DataFrame(all_results)


print()
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

summary = results_df.groupby("model").agg(
    images=("image", "count"),
    mean_gt_count=(
        "ground_truth_count",
        "mean",
    ),
    mean_prediction_count=(
        "prediction_count",
        "mean",
    ),
    mean_best_iou=(
        "mean_best_iou",
        "mean",
    ),
)

print(summary)


# ============================================================
# Save
# ============================================================

output_path = PROJECT_ROOT / "segmentation_model_validation.csv"

results_df.to_csv(
    output_path,
    index=False,
)

print()
print(
    "Saved:",
    output_path,
)
