from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent

df = pd.read_csv(PROJECT_ROOT / "Train.csv")


exact_normal = 0
exact_reverse = 0
paired_scans = 0

normal_errors = []
reverse_errors = []


for folder, group in df.groupby("FolderName"):
    left = group[group["Side"] == "L"].set_index("PlantNumber")["RootVolume"]

    right = group[group["Side"] == "R"].set_index("PlantNumber")["RootVolume"]

    common = sorted(set(left.index) & set(right.index))

    if len(common) < 2:
        continue

    paired_scans += 1

    l = np.array([left[p] for p in common])
    r = np.array([right[p] for p in common])

    # Same PlantNumber
    normal_error = np.mean(np.abs(l - r))

    # Reverse plant order
    reverse_error = np.mean(np.abs(l - r[::-1]))

    normal_errors.append(normal_error)
    reverse_errors.append(reverse_error)

    if np.allclose(l, r):
        exact_normal += 1

    if np.allclose(l, r[::-1]):
        exact_reverse += 1


print("=" * 60)
print("L/R REVERSAL CHECK")
print("=" * 60)

print(f"Paired scans: {paired_scans}")

print()
print(f"Exact normal matches:   {exact_normal}")
print(f"Exact reversed matches: {exact_reverse}")

print()
print(f"Mean normal error:   {np.mean(normal_errors):.6f}")

print(f"Mean reversed error: {np.mean(reverse_errors):.6f}")

print()
print(f"Median normal error:   {np.median(normal_errors):.6f}")

print(f"Median reversed error: {np.median(reverse_errors):.6f}")
