from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "Train.csv"
FEATURES = ROOT / "outputs" / "baseline_comparison" / "image_area_features.csv"

df = pd.read_csv(DATA)
features = pd.read_csv(FEATURES)

df = df.merge(features, on="ID", validate="one_to_one")

target = df["RootVolume"].to_numpy()
groups = df["FolderName"]

area_features = {
    "otsu_dark": ["otsu_dark_area"],
    "otsu_bright": ["otsu_bright_area"],
    "dark_128": ["dark_area_128"],
    "bright_128": ["bright_area_128"],
}

splitter = GroupKFold(n_splits=5)

results = []
predictions = []

for fold, (train_idx, val_idx) in enumerate(
    splitter.split(df, target, groups), start=1
):
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    y_train = target[train_idx]
    y_val = target[val_idx]

    for name, columns in area_features.items():
        model = LinearRegression()

        model.fit(train[columns], y_train)

        pred = model.predict(val[columns])

        metrics = {
            "fold": fold,
            "method": name,
            "rmse": np.sqrt(mean_squared_error(y_val, pred)),
            "mae": mean_absolute_error(y_val, pred),
            "r2": r2_score(y_val, pred),
        }

        results.append(metrics)

        for observation_id, actual, predicted in zip(val["ID"], y_val, pred):
            predictions.append(
                {
                    "ID": observation_id,
                    "fold": fold,
                    "method": name,
                    "actual": actual,
                    "predicted": predicted,
                }
            )

        print(
            f"Fold {fold} | {name:12s} | "
            f"RMSE {metrics['rmse']:.4f} | "
            f"MAE {metrics['mae']:.4f} | "
            f"R2 {metrics['r2']:.4f}"
        )

results = pd.DataFrame(results)
predictions = pd.DataFrame(predictions)

print("\nFOLD-AVERAGED RESULTS")
print(
    results.groupby("method")
    .agg(
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"),
        r2_mean=("r2", "mean"),
    )
    .to_string()
)

print("\nPOOLED OOF RESULTS")

for name, group in predictions.groupby("method"):
    actual = group["actual"]
    predicted = group["predicted"]

    print(
        f"{name:12s} | "
        f"RMSE={np.sqrt(mean_squared_error(actual, predicted)):.4f} | "
        f"MAE={mean_absolute_error(actual, predicted):.4f} | "
        f"R2={r2_score(actual, predicted):.4f}"
    )

out = ROOT / "outputs" / "simple_area"
out.mkdir(parents=True, exist_ok=True)

results.to_csv(out / "fold_metrics.csv", index=False)
predictions.to_csv(out / "oof_predictions.csv", index=False)
