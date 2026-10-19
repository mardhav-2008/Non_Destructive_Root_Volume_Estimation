from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

train = pd.read_csv(PROJECT_ROOT / "Train.csv")


def image_features(image_path):
    """
    Extract simple statistics from one GPR strip.
    """

    image = np.array(Image.open(image_path))

    # RGBA -> grayscale
    if image.ndim == 3:
        image = image[:, :, :3].mean(axis=2)

    image = image.astype(np.float32)

    features = {
        "mean": image.mean(),
        "std": image.std(),
        "min": image.min(),
        "max": image.max(),
        "p10": np.percentile(image, 10),
        "p25": np.percentile(image, 25),
        "median": np.percentile(image, 50),
        "p75": np.percentile(image, 75),
        "p90": np.percentile(image, 90),
        "bright_fraction": np.mean(image > 200),
        "dark_fraction": np.mean(image < 50),
    }

    return features


# ------------------------------------------------------------
# Extract features from every image
# ------------------------------------------------------------

cache = {}

image_paths = list(TRAIN_DIR.rglob("*.png"))

print(f"Found {len(image_paths)} training images.")

for i, path in enumerate(image_paths):
    # Example:
    # A6dzrkjqvl_L_033.png

    filename = path.stem
    parts = filename.rsplit("_", 2)

    folder = parts[0]
    side = parts[1]
    slice_number = int(parts[2])

    features = image_features(path)

    cache[(folder, side, slice_number)] = features

    if (i + 1) % 1000 == 0:
        print(f"Processed {i + 1}/{len(image_paths)}")


# ------------------------------------------------------------
# Aggregate images for each observation
# ------------------------------------------------------------

rows = []

for _, observation in train.iterrows():
    folder = observation["FolderName"]
    side = observation["Side"]
    start = observation["Start"]
    end = observation["End"]

    selected = []

    for slice_number in range(start, end + 1):
        key = (folder, side, slice_number)

        if key in cache:
            selected.append(cache[key])

    if not selected:
        print(f"WARNING: no images found for {folder} {side} {start}-{end}")
        continue

    feature_df = pd.DataFrame(selected)

    row = {
        "ID": observation["ID"],
        "FolderName": folder,
        "PlantNumber": observation["PlantNumber"],
        "Side": side,
        "Start": start,
        "End": end,
        "Genotype": observation["Genotype"],
        "Stage": observation["Stage"],
        "RootVolume": observation["RootVolume"],
        "num_slices": len(selected),
    }

    # Aggregate every pixel feature.
    for column in feature_df.columns:
        row[f"{column}_mean"] = feature_df[column].mean()
        row[f"{column}_std"] = feature_df[column].std()
        row[f"{column}_min"] = feature_df[column].min()
        row[f"{column}_max"] = feature_df[column].max()

    rows.append(row)


output = pd.DataFrame(rows)

output_path = PROJECT_ROOT / "observation_pixel_features.csv"

output.to_csv(output_path, index=False)

print()
print(f"Saved {len(output)} observations.")
print(f"Output: {output_path}")
