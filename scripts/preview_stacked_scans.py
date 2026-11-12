from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT / "Train.csv")
image_root = ROOT / "data" / "train"

out = ROOT / "outputs" / "audit" / "stack_previews"
out.mkdir(parents=True, exist_ok=True)

# Select observations with short, typical and long sequences.
lengths = train["End"] - train["Start"] + 1

indices = [
    lengths.idxmin(),
    (lengths - lengths.median()).abs().idxmin(),
    lengths.idxmax(),
]

# Also inspect a few randomly selected observations.
indices += train.sample(n=3, random_state=42).index.tolist()

seen = set()
selected = []

for index in indices:
    if index not in seen:
        selected.append(index)
        seen.add(index)

for index in selected:
    row = train.loc[index]

    folder = row["FolderName"]
    side = row["Side"]

    strips = []

    for number in range(int(row["Start"]), int(row["End"]) + 1):
        path = image_root / folder / f"{folder}_{side}_{number:03d}.png"

        with Image.open(path) as image:
            strips.append(np.asarray(image.convert("L")))

    stacked = np.vstack(strips)

    fig, ax = plt.subplots(figsize=(14, 5))

    ax.imshow(stacked, cmap="gray", aspect="auto")
    ax.set_title(
        f"{row['ID']} | Plant {row['PlantNumber']} | "
        f"{side} | {row['Start']}–{row['End']} | "
        f"Shape: {stacked.shape}"
    )
    ax.set_xlabel("Horizontal pixel position")
    ax.set_ylabel("Stacked slice position")

    fig.tight_layout()

    filename = f"{row['ID']}.png"
    fig.savefig(out / filename, dpi=150)
    plt.close(fig)

    print(
        f"Saved {filename} | "
        f"Shape: {stacked.shape} | "
        f"Pixel range: {stacked.min()}–{stacked.max()}"
    )

print(f"\nPreviews saved to: {out}")
