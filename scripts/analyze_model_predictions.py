import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import RootVolumeDataset
from models.cnn_regressor import RootVolumeRegressor

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_DIR = ROOT / "outputs" / "splits"
MODEL_DIR = ROOT / "outputs" / "models"
OUT_DIR = ROOT / "outputs" / "analysis"

OUT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(SPLIT_DIR / "train_split.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation_split.csv")

with open(SPLIT_DIR / "category_maps.json") as f:
    category_maps = json.load(f)

category_sizes = [
    len(category_maps[c]) for c in ["PlantNumber", "Genotype", "Stage", "Side"]
]

dataset = RootVolumeDataset(
    val_df,
    ROOT / "data" / "train",
    category_maps,
    augment=False,
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
)

results = []

for mode in ["image", "metadata", "multimodal"]:
    checkpoint_path = MODEL_DIR / f"{mode}_best.pt"

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = RootVolumeRegressor(
        category_sizes=category_sizes,
        mode=mode,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    numeric_mean = checkpoint["numeric_mean"].to(DEVICE)
    numeric_std = checkpoint["numeric_std"].to(DEVICE)

    predictions = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)

            numeric = batch["numeric"].to(DEVICE)
            numeric = (numeric - numeric_mean) / numeric_std

            if mode == "image":
                pred = model(image=image)
            elif mode == "metadata":
                pred = model(
                    categorical=categorical,
                    numeric=numeric,
                )
            else:
                pred = model(
                    image=image,
                    categorical=categorical,
                    numeric=numeric,
                )

            predictions.extend(pred.cpu().numpy())

    results.append(
        pd.DataFrame(
            {
                "ID": val_df["ID"].values,
                "FolderName": val_df["FolderName"].values,
                "PlantNumber": val_df["PlantNumber"].values,
                "RootVolume": val_df["RootVolume"].values,
                "Prediction": predictions,
                "Model": mode,
            }
        )
    )

all_predictions = pd.concat(results, ignore_index=True)

# --------------------------------------------------
# Metrics
# --------------------------------------------------

print("=" * 65)
print("VALIDATION PREDICTION ANALYSIS")
print("=" * 65)

for mode, group in all_predictions.groupby("Model"):
    y = group["RootVolume"].values
    p = group["Prediction"].values

    print(f"\n{mode.upper()}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y, p)):.4f}")
    print(f"MAE:  {mean_absolute_error(y, p):.4f}")
    print(f"R2:   {r2_score(y, p):.4f}")

# Training mean baseline.
baseline = train_df["RootVolume"].mean()
actual = val_df["RootVolume"].values

baseline_predictions = np.full_like(actual, baseline)

print("\nTRAINING MEAN BASELINE")
print(f"Prediction: {baseline:.4f}")
print(
    "RMSE:",
    np.sqrt(mean_squared_error(actual, baseline_predictions)),
)

# --------------------------------------------------
# Multimodal error by PlantNumber
# --------------------------------------------------

multi = all_predictions[all_predictions["Model"] == "multimodal"].copy()

multi["Error"] = multi["Prediction"] - multi["RootVolume"]
multi["AbsoluteError"] = multi["Error"].abs()

plant_metrics = multi.groupby("PlantNumber").agg(
    observations=("RootVolume", "size"),
    actual_mean=("RootVolume", "mean"),
    predicted_mean=("Prediction", "mean"),
    mae=("AbsoluteError", "mean"),
    bias=("Error", "mean"),
)

print("\nMULTIMODAL PERFORMANCE BY PLANT NUMBER")
print(plant_metrics.to_string())

# --------------------------------------------------
# Largest prediction errors
# --------------------------------------------------

print("\nLARGEST MULTIMODAL ERRORS")
print(
    multi.sort_values("AbsoluteError", ascending=False)[
        [
            "ID",
            "FolderName",
            "PlantNumber",
            "RootVolume",
            "Prediction",
            "Error",
            "AbsoluteError",
        ]
    ]
    .head(15)
    .to_string(index=False)
)

# --------------------------------------------------
# Plots
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 7))

for mode, group in all_predictions.groupby("Model"):
    ax.scatter(
        group["RootVolume"],
        group["Prediction"],
        label=mode,
        alpha=0.7,
    )

upper = max(
    all_predictions["RootVolume"].max(),
    all_predictions["Prediction"].max(),
)

ax.plot([0, upper], [0, upper], "k--", label="Ideal prediction")

ax.set_xlabel("Actual RootVolume")
ax.set_ylabel("Predicted RootVolume")
ax.set_title("Actual vs Predicted RootVolume")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(OUT_DIR / "actual_vs_predicted.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(multi["Error"], bins=20)
ax.axvline(0, linestyle="--", color="black")

ax.set_xlabel("Prediction error (prediction - actual)")
ax.set_ylabel("Observations")
ax.set_title("Multimodal Prediction Error Distribution")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(OUT_DIR / "multimodal_residuals.png", dpi=160)
plt.close(fig)

# --------------------------------------------------
# Save results
# --------------------------------------------------

all_predictions.to_csv(
    OUT_DIR / "validation_predictions.csv",
    index=False,
)

plant_metrics.to_csv(
    OUT_DIR / "plant_number_metrics.csv",
)

print("\nSaved analysis to:", OUT_DIR)
