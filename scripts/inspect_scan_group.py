from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

df = pd.read_csv(PROJECT_ROOT / "Train.csv")


# ------------------------------------------------------------
# Pick a scan
# ------------------------------------------------------------

FOLDER = "A6dzrkjqvl"
SIDE = "L"


observations = df[(df["FolderName"] == FOLDER) & (df["Side"] == SIDE)].sort_values(
    "PlantNumber"
)


print(
    observations[
        [
            "ID",
            "PlantNumber",
            "Side",
            "Start",
            "End",
            "RootVolume",
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Load the complete side
# ------------------------------------------------------------

images = []

for slice_number in range(1, 83):
    path = TRAIN_DIR / FOLDER / f"{FOLDER}_{SIDE}_{slice_number:03d}.png"

    image = np.array(Image.open(path))

    if image.ndim == 3:
        image = image[:, :, :3].mean(axis=2)

    images.append(image)


scan = np.vstack(images)


# ------------------------------------------------------------
# Plot complete scan
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(16, 10))

ax.imshow(
    scan,
    cmap="gray",
    aspect="auto",
)


# ------------------------------------------------------------
# Mark every observation's slice range
# ------------------------------------------------------------

for _, row in observations.iterrows():
    start = int(row["Start"])
    end = int(row["End"])

    # Each slice is 14 pixels tall.
    y_start = (start - 1) * 14
    y_end = end * 14

    ax.axhline(y_start, linestyle="--")
    ax.axhline(y_end, linestyle="--")

    ax.text(
        5,
        (y_start + y_end) / 2,
        f"Plant {row['PlantNumber']} V={row['RootVolume']}",
        fontsize=10,
        va="center",
    )


ax.set_title(f"{FOLDER} — {SIDE} — complete scan")

ax.set_xlabel("Horizontal position")
ax.set_ylabel("Slice")

plt.tight_layout()
plt.show()
