from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "Models" / "best_full.pt"
LABEL_DIR = PROJECT_ROOT / "train_labels"
TRAIN_DIR = PROJECT_ROOT / "data" / "train"


# ============================================================
# SETTINGS
# ============================================================

CONF_THRESHOLDS = [
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.10,
    0.125,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
]

IOU_MATCH_THRESHOLD = 0.50


# ============================================================
# YOLO LABEL READER
# ============================================================


def load_ground_truth(label_path, width, height):
    """
    Read YOLO segmentation labels.

    Returns:
        list[np.ndarray]
        Each array is an Nx2 polygon in pixel coordinates.
    """

    polygons = []

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()

        if len(values) < 7:
            continue

        # First value is class ID
        coords = np.array([float(x) for x in values[1:]], dtype=np.float32)

        coords = coords.reshape(-1, 2)

        coords[:, 0] *= width
        coords[:, 1] *= height

        polygons.append(coords)

    return polygons


# ============================================================
# POLYGON -> MASK
# ============================================================


def polygon_to_mask(polygon, width, height):
    mask = np.zeros((height, width), dtype=np.uint8)

    points = np.round(polygon).astype(np.int32)

    cv2.fillPoly(mask, [points], 1)

    return mask


# ============================================================
# IOU
# ============================================================


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return intersection / union


# ============================================================
# ONE-TO-ONE MATCHING
# ============================================================


def match_predictions(gt_masks, pred_masks):
    """
    Greedy one-to-one matching.

    Returns:
        matched_ious
        number of matched objects
    """

    if len(gt_masks) == 0 or len(pred_masks) == 0:
        return []

    pairs = []

    for gt_idx, gt_mask in enumerate(gt_masks):
        for pred_idx, pred_mask in enumerate(pred_masks):
            iou = mask_iou(gt_mask, pred_mask)

            pairs.append((iou, gt_idx, pred_idx))

    # Highest IoUs first
    pairs.sort(reverse=True)

    used_gt = set()
    used_pred = set()

    matched_ious = []

    for iou, gt_idx, pred_idx in pairs:
        if gt_idx in used_gt:
            continue

        if pred_idx in used_pred:
            continue

        used_gt.add(gt_idx)
        used_pred.add(pred_idx)

        matched_ious.append(iou)

    return matched_ious


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING MODEL")
print("=" * 70)

model = YOLO(str(MODEL_PATH))

print(f"Model: {MODEL_PATH.name}")
print(f"Task: {model.task}")
print(f"Classes: {model.names}")


# ============================================================
# FIND LABELLED IMAGES
# ============================================================

label_files = sorted(LABEL_DIR.glob("*.txt"))

print()
print(f"Label files: {len(label_files)}")


# ============================================================
# CACHE IMAGE / GT INFORMATION
# ============================================================

samples = []

print()
print("Preparing labelled samples...")

for label_path in label_files:
    # Example:
    # A2miww5mfx_L_010.txt
    stem = label_path.stem

    image_path = TRAIN_DIR / stem.split("_")[0]

    # Folder name is the first part before _L/_R
    parts = stem.split("_")

    if len(parts) < 3:
        continue

    folder = parts[0]
    side = parts[1]
    slice_number = parts[2]

    image_path = TRAIN_DIR / folder / f"{stem}.png"

    if not image_path.exists():
        continue

    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    height, width = image.shape

    gt_polygons = load_ground_truth(label_path, width, height)

    gt_masks = [polygon_to_mask(poly, width, height) for poly in gt_polygons]

    samples.append(
        {
            "image_path": image_path,
            "gt_masks": gt_masks,
            "width": width,
            "height": height,
        }
    )


print(f"Usable samples: {len(samples)}")


# ============================================================
# RUN PREDICTIONS ONCE
# ============================================================

print()
print("=" * 70)
print("RUNNING YOLO")
print("=" * 70)

prediction_cache = []

for i, sample in enumerate(samples):
    if i % 25 == 0:
        print(f"{i}/{len(samples)}")

    results = model.predict(source=str(sample["image_path"]), conf=0.001, verbose=False)

    result = results[0]

    predictions = []

    if result.masks is not None:
        for j, polygon in enumerate(result.masks.xy):
            confidence = float(result.boxes.conf[j])

            mask = polygon_to_mask(polygon, sample["width"], sample["height"])

            predictions.append(
                {
                    "confidence": confidence,
                    "mask": mask,
                }
            )

    prediction_cache.append(
        {
            "gt_masks": sample["gt_masks"],
            "predictions": predictions,
        }
    )


# ============================================================
# THRESHOLD SWEEP
# ============================================================

print()
print("=" * 70)
print("CONFIDENCE THRESHOLD SWEEP")
print("=" * 70)

results_table = []

for threshold in CONF_THRESHOLDS:
    total_gt = 0
    total_pred = 0

    true_positives = 0

    matched_ious = []

    for sample in prediction_cache:
        gt_masks = sample["gt_masks"]

        predictions = [
            p["mask"] for p in sample["predictions"] if p["confidence"] >= threshold
        ]

        total_gt += len(gt_masks)
        total_pred += len(predictions)

        matches = match_predictions(gt_masks, predictions)

        # Count IoU >= threshold as a true positive
        tp = sum(iou >= IOU_MATCH_THRESHOLD for iou in matches)

        true_positives += tp

        matched_ious.extend(matches)

    false_positives = total_pred - true_positives
    false_negatives = total_gt - true_positives

    precision = true_positives / total_pred if total_pred > 0 else 0

    recall = true_positives / total_gt if total_gt > 0 else 0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    mean_iou = np.mean(matched_ious) if matched_ious else 0

    results_table.append(
        {
            "confidence": threshold,
            "predictions": total_pred,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_matched_iou": mean_iou,
        }
    )


# ============================================================
# DISPLAY
# ============================================================

df = pd.DataFrame(results_table)

print()
print(df.to_string(index=False))

print()
print("=" * 70)
print("BEST THRESHOLDS")
print("=" * 70)

best_f1 = df.loc[df["f1"].idxmax()]
best_iou = df.loc[df["mean_matched_iou"].idxmax()]

print()
print("Best F1:")
print(best_f1.to_string())

print()
print("Best mean IoU:")
print(best_iou.to_string())


# ============================================================
# SAVE
# ============================================================

output_path = PROJECT_ROOT / "segmentation_threshold_sweep.csv"

df.to_csv(output_path, index=False)

print()
print(f"Saved: {output_path}")
