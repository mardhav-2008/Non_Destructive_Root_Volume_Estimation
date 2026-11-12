from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "Train.csv"
FEATURES = ROOT / "outputs" / "baseline_comparison" / "image_area_features.csv"

# We will use the existing cached features for global
# comparisons, and extract spatial features from scans.

IMAGE_DIR = ROOT / "data" / "train"
OUTPUT = ROOT / "outputs" / "spatial_area"

BINS = [2, 4, 8, 16]
THRESHOLD = 128
POLARITY = "bright"


def load_scan(row):
    folder = str(row["FolderName"])
    side = str(row["Side"])

    start = int(row["Start"])
    end = int(row["End"])

    strips = []

    for index in range(start, end + 1):
        path = IMAGE_DIR / folder / f"{folder}_{side}_{index:03d}.png"

        from PIL import Image

        with Image.open(path) as image:
            strips.append(np.asarray(image.convert("L")))

    return np.concatenate(strips, axis=0)


def extract_spatial_features(row):
    image = load_scan(row)

    features = {}

    for n_bins in BINS:
        bands = np.array_split(image, n_bins, axis=0)

        for index, band in enumerate(bands):
            if POLARITY == "bright":
                fraction = np.mean(band >= THRESHOLD)
            else:
                fraction = np.mean(band <= THRESHOLD)

            features[f"bins{n_bins}_band{index}"] = fraction

    return features


def evaluate(X, y, groups, name):
    splitter = GroupKFold(n_splits=5)

    rows = []
    oof = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0)),
            ]
        )

        model.fit(X.iloc[train_idx], y[train_idx])

        pred = model.predict(X.iloc[val_idx])
        oof[val_idx] = pred

        rmse = np.sqrt(mean_squared_error(y[val_idx], pred))

        mae = mean_absolute_error(y[val_idx], pred)

        r2 = r2_score(y[val_idx], pred)

        rows.append(
            {
                "fold": fold,
                "experiment": name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
            }
        )

        print(f"{name:20s} Fold {fold} | RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

    return rows, oof


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA)
    y = df["RootVolume"].to_numpy()
    groups = df["FolderName"]

    feature_path = OUTPUT / "spatial_features.csv"

    if feature_path.exists():
        print("Loading cached spatial features...")
        spatial = pd.read_csv(feature_path)

    else:
        records = []

        for i, (_, row) in enumerate(df.iterrows(), start=1):
            features = extract_spatial_features(row)
            features["ID"] = row["ID"]
            records.append(features)

            if i % 25 == 0:
                print(f"Extracted {i}/{len(df)}")

        spatial = pd.DataFrame(records)

        spatial.to_csv(feature_path, index=False)

    df = df.merge(spatial, on="ID", validate="one_to_one")

    all_metrics = []
    all_oof = []

    for n_bins in BINS:
        columns = [f"bins{n_bins}_band{i}" for i in range(n_bins)]

        rows, predictions = evaluate(df[columns], y, groups, f"spatial_{n_bins}_bins")

        all_metrics.extend(rows)

        all_oof.append(
            pd.DataFrame(
                {
                    "ID": df["ID"],
                    "actual": y,
                    "predicted": predictions,
                    "experiment": f"spatial_{n_bins}_bins",
                }
            )
        )

    metrics_df = pd.DataFrame(all_metrics)
    oof_df = pd.concat(all_oof, ignore_index=True)

    metrics_df.to_csv(OUTPUT / "fold_metrics.csv", index=False)

    oof_df.to_csv(OUTPUT / "oof_predictions.csv", index=False)

    print("\nFOLD-AVERAGED RESULTS")
    print(
        metrics_df.groupby("experiment")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            r2_mean=("r2", "mean"),
        )
        .to_string()
    )

    print("\nPOOLED RESULTS")

    for name, group in oof_df.groupby("experiment"):
        print(
            f"{name:20s} | "
            f"RMSE={np.sqrt(mean_squared_error(group.actual, group.predicted)):.4f} | "
            f"MAE={mean_absolute_error(group.actual, group.predicted):.4f} | "
            f"R2={r2_score(group.actual, group.predicted):.4f}"
        )


if __name__ == "__main__":
    main()
