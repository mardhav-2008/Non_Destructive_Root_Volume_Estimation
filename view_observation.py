from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

df = pd.read_csv(PROJECT_ROOT / "Train.csv")


# ------------------------------------------------------------
# Choose an observation
# ------------------------------------------------------------

ROW = 0

row = df.iloc[ROW]

folder = row["FolderName"]
side = row["Side"]
start = int(row["Start"])
end = int(row["End"])

print("Observation:")
print(row)


# ------------------------------------------------------------
# Load strips
# ------------------------------------------------------------

images = []

for slice_number in range(start, end + 1):
    path = TRAIN_DIR / folder / f"{folder}_{side}_{slice_number:03d}.png"

    if not path.exists():
        print("Missing:", path)
        continue

    image = np.array(Image.open(path))

    # RGBA -> grayscale
    if image.ndim == 3:
        image = image[:, :, :3].mean(axis=2)

    images.append(image)


# ------------------------------------------------------------
# Stack slices vertically
# ------------------------------------------------------------

scan = np.vstack(images)


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

plt.figure(figsize=(16, 6))

plt.imshow(
    scan,
    cmap="gray",
    aspect="auto",
)

plt.title(
    f"{folder} | Plant {row['PlantNumber']} | "
    f"{side} | slices {start}-{end} | "
    f"RootVolume={row['RootVolume']}"
)

plt.xlabel("Horizontal position")
plt.ylabel("Slice")

plt.tight_layout()
plt.show()
