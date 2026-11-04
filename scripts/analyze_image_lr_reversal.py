from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

df = pd.read_csv(PROJECT_ROOT / "Train.csv")


def load_region(folder, side, start, end):
    """
    Load the Start-End region and calculate
    a horizontal GPR response profile.
    """

    images = []

    for slice_number in range(start, end + 1):
        path = TRAIN_DIR / folder / f"{folder}_{side}_{slice_number:03d}.png"

        if not path.exists():
            continue

        image = np.array(Image.open(path))

        if image.ndim == 3:
            image = image[:, :, :3].mean(axis=2)

        image = image.astype(np.float32)

        images.append(image)

    if not images:
        return None

    scan = np.vstack(images)

    # Remove each row's average.
    scan = scan - scan.mean(axis=1, keepdims=True)

    # Horizontal response profile.
    profile = np.mean(
        np.abs(scan),
        axis=0,
    )

    return profile


def normalize_profile(profile, length=500):

    # Normalize x-axis so profiles of different widths
    # can be compared.

    old_x = np.linspace(
        0,
        1,
        len(profile),
    )

    new_x = np.linspace(
        0,
        1,
        length,
    )

    profile = np.interp(
        new_x,
        old_x,
        profile,
    )

    # Standardize amplitude.
    profile = (profile - profile.mean()) / (profile.std() + 1e-8)

    return profile


def correlation(a, b):
    return np.corrcoef(a, b)[0, 1]


results = []


# ------------------------------------------------------------
# Find scans with both L and R observations
# ------------------------------------------------------------

for folder, group in df.groupby("FolderName"):
    left = group[group["Side"] == "L"]
    right = group[group["Side"] == "R"]

    if len(left) == 0 or len(right) == 0:
        continue

    # Use the overall observed range for each side.
    l_start = int(left["Start"].min())
    l_end = int(left["End"].max())

    r_start = int(right["Start"].min())
    r_end = int(right["End"].max())

    l_profile = load_region(
        folder,
        "L",
        l_start,
        l_end,
    )

    r_profile = load_region(
        folder,
        "R",
        r_start,
        r_end,
    )

    if l_profile is None or r_profile is None:
        continue

    l_profile = normalize_profile(l_profile)
    r_profile = normalize_profile(r_profile)

    normal = correlation(
        l_profile,
        r_profile,
    )

    reversed_corr = correlation(
        l_profile,
        r_profile[::-1],
    )

    results.append(
        {
            "FolderName": folder,
            "normal_corr": normal,
            "reversed_corr": reversed_corr,
            "improvement": reversed_corr - normal,
        }
    )


results = pd.DataFrame(results)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("=" * 70)
print("IMAGE L/R REVERSAL CHECK")
print("=" * 70)

print(f"Paired scans: {len(results)}")

print()

print(f"Mean normal correlation:   {results['normal_corr'].mean():.4f}")

print(f"Mean reversed correlation: {results['reversed_corr'].mean():.4f}")

print()

print(f"Median normal correlation:   {results['normal_corr'].median():.4f}")

print(f"Median reversed correlation: {results['reversed_corr'].median():.4f}")

print()

print(f"Mean improvement: {results['improvement'].mean():.4f}")

print()

print(
    results.sort_values(
        "improvement",
        ascending=False,
    ).to_string(index=False)
)
