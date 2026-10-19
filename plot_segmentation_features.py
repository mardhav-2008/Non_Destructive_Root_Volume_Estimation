from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent

df = pd.read_csv(PROJECT_ROOT / "segmentation_features.csv")


# --------------------------------------------------
# One example scan
# --------------------------------------------------

example = df[(df["scan"] == "A2miww5mfx") & (df["side"] == "L")].sort_values("slice")


plt.figure(figsize=(10, 5))

plt.plot(example["slice"], example["total_area"], marker="o")

plt.xlabel("Slice")
plt.ylabel("Total segmentation area")
plt.title("A2miww5mfx — L — Segmentation area")

plt.grid(True)

plt.show()


# --------------------------------------------------
# Number of objects
# --------------------------------------------------

plt.figure(figsize=(10, 5))

plt.plot(example["slice"], example["num_objects"], marker="o")

plt.xlabel("Slice")
plt.ylabel("Number of segmented objects")
plt.title("A2miww5mfx — L — Number of objects")

plt.grid(True)

plt.show()
