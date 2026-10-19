from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

df = pd.read_csv(PROJECT_ROOT / "Train.csv")


# ------------------------------------------------------------
# Find scans containing both L and R observations
# ------------------------------------------------------------

groups = []

for folder, group in df.groupby("FolderName"):
    left = group[group["Side"] == "L"]
    right = group[group["Side"] == "R"]

    if len(left) >= 2 and len(right) >= 2:
        groups.append(
            {
                "FolderName": folder,
                "L_count": len(left),
                "R_count": len(right),
            }
        )


groups = pd.DataFrame(groups)

print(f"Scans with both L and R observations: {len(groups)}")

print()
print(groups.head(20).to_string(index=False))


# ------------------------------------------------------------
# Compare L/R volume patterns
# ------------------------------------------------------------

normal_correlations = []
reverse_correlations = []


for folder in groups["FolderName"]:
    group = df[df["FolderName"] == folder]

    left = group[group["Side"] == "L"].set_index("PlantNumber")["RootVolume"]

    right = group[group["Side"] == "R"].set_index("PlantNumber")["RootVolume"]

    common = sorted(set(left.index) & set(right.index))

    if len(common) < 3:
        continue

    l_values = np.array([left[p] for p in common])

    r_values = np.array([right[p] for p in common])

    normal_corr = np.corrcoef(l_values, r_values)[0, 1]

    # Compare L against reversed R order
    r_reverse = r_values[::-1]

    reverse_corr = np.corrcoef(l_values, r_reverse)[0, 1]

    normal_correlations.append(normal_corr)

    reverse_correlations.append(reverse_corr)


print()
print("=" * 60)
print("L/R VOLUME CORRELATION")
print("=" * 60)

print(
    f"Normal correlation:  "
    f"{np.mean(normal_correlations):.4f} "
    f"+/- {np.std(normal_correlations):.4f}"
)

print(
    f"Reversed correlation: "
    f"{np.mean(reverse_correlations):.4f} "
    f"+/- {np.std(reverse_correlations):.4f}"
)


# ------------------------------------------------------------
# Show some individual scans
# ------------------------------------------------------------

print()
print("=" * 60)
print("INDIVIDUAL SCANS")
print("=" * 60)

shown = 0

for folder in groups["FolderName"]:
    group = df[df["FolderName"] == folder]

    left = group[group["Side"] == "L"].sort_values("PlantNumber")

    right = group[group["Side"] == "R"].sort_values("PlantNumber")

    common = sorted(set(left["PlantNumber"]) & set(right["PlantNumber"]))

    if len(common) < 3:
        continue

    l_values = [left[left["PlantNumber"] == p]["RootVolume"].iloc[0] for p in common]

    r_values = [right[right["PlantNumber"] == p]["RootVolume"].iloc[0] for p in common]

    print()
    print(folder)
    print("Plants :", common)
    print("L      :", l_values)
    print("R      :", r_values)
    print("R rev  :", r_values[::-1])

    shown += 1

    if shown >= 10:
        break
