from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
from PIL import Image

from ultralytics import YOLO
from ultralytics.engine.results import Results


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "Models" / "best_early.pt"

IMAGE_PATH = PROJECT_ROOT / "data" / "train" / "A2miww5mfx" / "A2miww5mfx_L_010.png"

LABEL_PATH = PROJECT_ROOT / "train_labels" / "A2miww5mfx_L_010.txt"


# ============================================================
# Settings
# ============================================================

CONFIDENCE = 0.10


# ============================================================
# Load image
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

image_width, image_height = image.size

print("=" * 60)
print("IMAGE")
print("=" * 60)

print("Width :", image_width)
print("Height:", image_height)


# ============================================================
# Load ground-truth YOLO segmentation
# ============================================================

ground_truth_polygons = []

with open(LABEL_PATH, "r") as f:
    for line in f:
        values = list(map(float, line.split()))

        class_id = int(values[0])

        coordinates = values[1:]

        points = []

        for i in range(0, len(coordinates), 2):
            x = coordinates[i] * image_width
            y = coordinates[i + 1] * image_height

            points.append((x, y))

        ground_truth_polygons.append(points)


print()
print("=" * 60)
print("GROUND TRUTH")
print("=" * 60)

print("Number of polygons:", len(ground_truth_polygons))


# ============================================================
# Load YOLO model
# ============================================================

model = YOLO(str(MODEL_PATH))


# ============================================================
# Run prediction
# ============================================================

results = model.predict(
    source=str(IMAGE_PATH),
    conf=CONFIDENCE,
    verbose=False,
)

result = cast(Results, results[0])


# ============================================================
# Extract predictions
# ============================================================

predicted_polygons = []

if result.masks is not None:
    predicted_polygons = result.masks.xy


print()
print("=" * 60)
print("YOLO PREDICTION")
print("=" * 60)

print("Confidence threshold:", CONFIDENCE)

print("Number of predicted masks:", len(predicted_polygons))

if result.boxes is not None:
    confidences = result.boxes.conf.cpu().numpy()

    print("Confidences:", confidences)


# ============================================================
# Plot
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 8),
)


# ============================================================
# Ground truth
# ============================================================

axes[0].imshow(image)

for i, polygon in enumerate(ground_truth_polygons):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]

    axes[0].plot(
        xs,
        ys,
        linewidth=2,
    )

    axes[0].text(
        xs[0],
        ys[0],
        f"GT {i}",
        fontsize=10,
    )


axes[0].set_title("Ground-truth segmentation")

axes[0].set_xlabel("Horizontal position")

axes[0].set_ylabel("Vertical position")


# ============================================================
# YOLO prediction
# ============================================================

axes[1].imshow(image)

for i, polygon in enumerate(predicted_polygons):
    xs = polygon[:, 0]
    ys = polygon[:, 1]

    axes[1].plot(
        xs,
        ys,
        linewidth=2,
    )

    axes[1].text(
        xs[0],
        ys[0],
        f"Pred {i}",
        fontsize=10,
    )


axes[1].set_title(f"YOLO prediction (confidence ≥ {CONFIDENCE})")

axes[1].set_xlabel("Horizontal position")

axes[1].set_ylabel("Vertical position")


# ============================================================
# Make the tiny image easier to inspect
# ============================================================

for ax in axes:
    ax.set_xlim(0, image_width)

    ax.set_ylim(image_height, 0)

    ax.set_aspect("auto")


plt.tight_layout()

plt.show()
