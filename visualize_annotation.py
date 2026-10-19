from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------
# Pick one image that we KNOW has a label
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

image_path = PROJECT_ROOT / "data" / "train" / "A2miww5mfx" / "A2miww5mfx_L_010.png"

label_path = PROJECT_ROOT / "train_labels" / "A2miww5mfx_L_010.txt"


# --------------------------------------------------
# Load image
# --------------------------------------------------

image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# OpenCV loads RGBA as BGRA
if image.shape[-1] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
else:
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

height, width = image.shape[:2]

print(f"Image shape: {image.shape}")
print(f"Width: {width}")
print(f"Height: {height}")


# --------------------------------------------------
# Read YOLO segmentation labels
# --------------------------------------------------

with open(label_path, "r") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Number of polygons: {len(lines)}")


# --------------------------------------------------
# Convert normalized YOLO coordinates -> pixels
# --------------------------------------------------

polygons = []

for line in lines:
    values = list(map(float, line.split()))

    class_id = int(values[0])
    coords = np.array(values[1:], dtype=np.float32)

    if len(coords) % 2 != 0:
        raise ValueError("Odd number of polygon coordinates")

    coords = coords.reshape(-1, 2)

    # YOLO coordinates are normalized [0, 1]
    coords[:, 0] *= width
    coords[:, 1] *= height

    polygons.append((class_id, coords.astype(np.int32)))


# --------------------------------------------------
# Draw polygons
# --------------------------------------------------

overlay = image.copy()

for class_id, polygon in polygons:
    cv2.fillPoly(
        overlay,
        [polygon],
        color=(255, 0, 0, 100) if overlay.shape[-1] == 4 else (255, 0, 0),
    )

    cv2.polylines(
        overlay,
        [polygon],
        isClosed=True,
        color=(255, 255, 0),
        thickness=1,
    )


# --------------------------------------------------
# Plot — vertically magnified
# --------------------------------------------------

scale_y = 20

image_big = cv2.resize(
    image,
    (width, height * scale_y),
    interpolation=cv2.INTER_NEAREST,
)

overlay_big = cv2.resize(
    overlay,
    (width, height * scale_y),
    interpolation=cv2.INTER_NEAREST,
)

plt.figure(figsize=(20, 6))
plt.imshow(image_big, aspect="auto")
plt.title("Original — Vertically Magnified")
plt.axis("off")

plt.figure(figsize=(20, 6))
plt.imshow(overlay_big, aspect="auto")
plt.title("YOLO Segmentation — Vertically Magnified")
plt.axis("off")

plt.show()
