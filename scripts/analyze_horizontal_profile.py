from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

df = pd.read_csv(PROJECT_ROOT / "Train.csv")


# ------------------------------------------------------------
# Choose a scan
# ------------------------------------------------------------

FOLDER = "A6dzrkjqvl"
SIDE = "L"


observations = df[(df["FolderName"] == FOLDER) & (df["Side"] == SIDE)].sort_values(
    "PlantNumber"
)


print(
    observations[
        [
            "PlantNumber",
            "Start",
            "End",
            "RootVolume",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Use the union of all observation ranges
# ------------------------------------------------------------

start = int(observations["Start"].min())
end = int(observations["End"].max())


images = []

for slice_number in range(start, end + 1):
    path = TRAIN_DIR / FOLDER / f"{FOLDER}_{SIDE}_{slice_number:03d}.png"

    image = np.array(Image.open(path))

    if image.ndim == 3:
        image = image[:, :, :3].mean(axis=2)

    image = image.astype(np.float32)

    images.append(image)


scan = np.vstack(images)


# ------------------------------------------------------------
# Horizontal profile
# ------------------------------------------------------------

# Remove the average value of each row.
#
# This emphasizes local GPR responses instead of
# simply measuring overall brightness.

scan_centered = scan - scan.mean(axis=1, keepdims=True)

profile = np.mean(
    np.abs(scan_centered),
    axis=0,
)


# Smooth the profile
window = 15

kernel = np.ones(window) / window

profile_smooth = np.convolve(
    profile,
    kernel,
    mode="same",
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    1,
    figsize=(16, 9),
)


# Image
axes[0].imshow(
    scan,
    cmap="gray",
    aspect="auto",
)

axes[0].set_title(f"{FOLDER} — {SIDE} — slices {start}-{end}")

axes[0].set_ylabel("Slice")


# Profile
axes[1].plot(profile_smooth)

axes[1].set_title("Horizontal GPR response profile")

axes[1].set_xlabel("Horizontal position")

axes[1].set_ylabel("Mean absolute response")


plt.tight_layout()
plt.show()
