from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "Models" / "best_full.pt"

SCAN_NAME = "A2miww5mfx"
SIDE = "L"

IMAGE_DIR = PROJECT_ROOT / "data" / "train" / SCAN_NAME


# ============================================================
# Settings
# ============================================================

CONFIDENCE = 0.10

START_SLICE = 10
END_SLICE = 23


# ============================================================
# Load model
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = YOLO(str(MODEL_PATH))


# ============================================================
# Process slices
# ============================================================

images = []
prediction_polygons = []
prediction_confidences = []


for slice_number in range(
    START_SLICE,
    END_SLICE + 1,
):
    image_path = IMAGE_DIR / f"{SCAN_NAME}_{SIDE}_{slice_number:03d}.png"

    if not image_path.exists():
        print(
            "Missing:",
            image_path.name,
        )

        continue

    image = np.array(Image.open(image_path).convert("RGB"))

    images.append(image)

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
        prediction_polygons.append([])
        prediction_confidences.append([])

        continue

    polygons = result.masks.xy

    if result.boxes is not None:
        confidences = result.boxes.conf.cpu().numpy().tolist()

    else:
        confidences = []

    prediction_polygons.append(polygons)

    prediction_confidences.append(confidences)


# ============================================================
# Create stacked image
# ============================================================

if not images:
    raise RuntimeError("No images were found.")


height, width, _ = images[0].shape

stacked_height = height * len(images)

stacked = np.zeros(
    (
        stacked_height,
        width,
        3,
    ),
    dtype=np.uint8,
)


for i, image in enumerate(images):
    y_start = i * height
    y_end = y_start + height

    stacked[y_start:y_end] = image


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(16, 10))

ax.imshow(
    stacked,
    aspect="auto",
)


# ============================================================
# Overlay predictions
# ============================================================

for i, polygons in enumerate(prediction_polygons):
    y_offset = i * height

    for polygon_index, polygon in enumerate(polygons):
        xs = polygon[:, 0]

        ys = polygon[:, 1] + y_offset

        ax.plot(
            xs,
            ys,
            linewidth=1.5,
        )

        if len(xs) > 0:
            confidence = ""

            if polygon_index < len(prediction_confidences[i]):
                confidence = f" {prediction_confidences[i][polygon_index]:.2f}"

            ax.text(
                xs[0],
                ys[0],
                f"{polygon_index}{confidence}",
                fontsize=7,
            )


# ============================================================
# Slice labels
# ============================================================

for i in range(len(images)):
    y = i * height + height / 2

    slice_number = START_SLICE + i

    ax.text(
        width + 5,
        y,
        f"{slice_number:03d}",
        fontsize=8,
        va="center",
    )


# ============================================================
# Formatting
# ============================================================

ax.set_title(f"{SCAN_NAME} — {SIDE} — YOLO best_full.pt (conf ≥ {CONFIDENCE})")

ax.set_xlabel("Horizontal position")

ax.set_ylabel("Slice")

ax.set_xlim(
    0,
    width,
)

ax.set_ylim(
    stacked_height,
    0,
)

plt.tight_layout()

plt.show()
