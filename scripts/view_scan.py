from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent

scan_name = "A2miww5mfx"
side = "L"

image_dir = PROJECT_ROOT / "data" / "train" / scan_name
label_dir = PROJECT_ROOT / "train_labels"


def load_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise FileNotFoundError(path)

    if image.shape[-1] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def load_polygons(label_path, width, height):
    if not label_path.exists():
        return []

    polygons = []

    with open(label_path) as f:
        for line in f:
            values = list(map(float, line.split()))

            class_id = int(values[0])
            coords = np.array(values[1:], dtype=np.float32)

            coords = coords.reshape(-1, 2)

            coords[:, 0] *= width
            coords[:, 1] *= height

            polygons.append((class_id, coords.astype(np.int32)))

    return polygons


# --------------------------------------------------
# Find all slices
# --------------------------------------------------

images = sorted(image_dir.glob(f"{scan_name}_{side}_*.png"))

print(f"Found {len(images)} images")


# --------------------------------------------------
# Build stacked visualization
# --------------------------------------------------

rows = []

for image_path in images:
    image = load_image(image_path)

    height, width = image.shape[:2]

    # Convert RGBA -> RGB for matplotlib
    if image.shape[-1] == 4:
        image_rgb = image[:, :, :3]
    else:
        image_rgb = image

    # Convert to grayscale
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # Convert grayscale back to RGB
    row = np.stack([gray, gray, gray], axis=-1)

    # Corresponding label
    label_path = label_dir / image_path.name.replace(".png", ".txt")

    polygons = load_polygons(label_path, width, height)

    # Draw segmentation
    for class_id, polygon in polygons:
        cv2.fillPoly(row, [polygon], color=(255, 100, 100))

        cv2.polylines(row, [polygon], isClosed=True, color=(255, 255, 255), thickness=1)

    rows.append(row)


# --------------------------------------------------
# Stack slices vertically
# --------------------------------------------------

stacked = np.vstack(rows)


# --------------------------------------------------
# Display
# --------------------------------------------------

scale_y = 4

stacked_big = cv2.resize(
    stacked,
    (stacked.shape[1], stacked.shape[0] * scale_y),
    interpolation=cv2.INTER_NEAREST,
)


plt.figure(figsize=(20, 12))

plt.imshow(stacked_big)

plt.title(f"{scan_name} — {side} — All {len(images)} slices")

plt.xlabel("Horizontal position")
plt.ylabel("Slice index")

plt.axis("off")

plt.show()
