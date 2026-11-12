from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_ROOT = ROOT / "data" / "train"

train = pd.read_csv(ROOT / "Train.csv")

lengths = Counter()
widths = Counter()
modes = Counter()
alpha_values = Counter()

channel_differences = []
observation_shapes = []

for _, row in train.iterrows():
    folder = row["FolderName"]
    side = row["Side"]

    images = []

    for index in range(int(row["Start"]), int(row["End"]) + 1):
        path = IMAGE_ROOT / folder / f"{folder}_{side}_{index:03d}.png"

        with Image.open(path) as image:
            array = np.asarray(image.convert("RGBA"))

        images.append(array)

        height, width, _ = array.shape
        widths[width] += 1
        modes["RGBA"] += 1

        alpha_values.update(np.unique(array[:, :, 3]).tolist())

        rgb = array[:, :, :3].astype(np.int16)
        channel_differences.append(
            [
                np.mean(np.abs(rgb[:, :, 0] - rgb[:, :, 1])),
                np.mean(np.abs(rgb[:, :, 1] - rgb[:, :, 2])),
            ]
        )

    lengths[len(images)] += 1

    # All strips in one observation should have compatible dimensions.
    unique_shapes = {image.shape for image in images}

    observation_shapes.append(
        {
            "ID": row["ID"],
            "FolderName": folder,
            "PlantNumber": row["PlantNumber"],
            "Side": side,
            "strip_count": len(images),
            "unique_shapes": len(unique_shapes),
            "stack_height": sum(image.shape[0] for image in images),
            "min_width": min(image.shape[1] for image in images),
            "max_width": max(image.shape[1] for image in images),
        }
    )

differences = np.asarray(channel_differences)
shapes = pd.DataFrame(observation_shapes)

print("=" * 65)
print("IMAGE FORMAT ANALYSIS")
print("=" * 65)

print("\nStrips per observation:")
for length, count in sorted(lengths.items()):
    print(f"{length:3d} strips: {count:3d} observations")

print("\nImage widths:")
for width, count in sorted(widths.items()):
    print(f"{width:4d} px: {count:5d} strips")

print("\nAlpha channel values:")
print(sorted(alpha_values))

print("\nRGB channel differences:")
print("Mean |R-G|:", differences[:, 0].mean())
print("Mean |G-B|:", differences[:, 1].mean())

print("\nObservation dimensions:")
print(shapes[["strip_count", "stack_height", "min_width", "max_width"]].describe())

print(
    "\nObservations with inconsistent strip shapes:",
    (shapes["unique_shapes"] > 1).sum(),
)

print("\nLargest width variation within an observation:")
print(
    shapes.assign(width_difference=shapes["max_width"] - shapes["min_width"])
    .sort_values("width_difference", ascending=False)
    .head(10)
    .to_string(index=False)
)

out = ROOT / "outputs" / "audit"
out.mkdir(parents=True, exist_ok=True)

shapes.to_csv(out / "image_observation_dimensions.csv", index=False)

print(f"\nSaved dimension report to: {out}")
