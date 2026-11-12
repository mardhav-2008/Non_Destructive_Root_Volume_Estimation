from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_CSV = ROOT / "Train.csv"
IMAGE_DIR = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs" / "baseline_comparison"

N_SPLITS = 5
RIDGE_ALPHA = 10.0

# Fixed grayscale thresholds for area measurements.
THRESHOLDS = [32, 64, 96, 128, 160, 192, 224]


# ============================================================
# IMAGE FEATURE EXTRACTION
# ============================================================


def load_stacked_scan(row):
    """
    Load an observation's image strips and stack them vertically
    in ascending slice order.
    """
    folder = str(row["FolderName"])
    side = str(row["Side"])

    start = int(row["Start"])
    end = int(row["End"])

    folder_path = IMAGE_DIR / folder

    strips = []

    for index in range(start, end + 1):
        filename = f"{folder}_{side}_{index:03d}.png"
        image_path = folder_path / filename

        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        with Image.open(image_path) as image:
            strips.append(np.asarray(image.convert("L")))

    return np.concatenate(strips, axis=0)


def otsu_threshold(image):
    """Calculate Otsu's threshold using a grayscale histogram."""
    histogram = np.bincount(image.ravel(), minlength=256).astype(np.float64)

    total = image.size
    levels = np.arange(256)

    weight_background = np.cumsum(histogram)
    sum_background = np.cumsum(histogram * levels)

    weight_foreground = total - weight_background
    sum_total = sum_background[-1]

    denominator = weight_background * weight_foreground

    valid = denominator > 0

    mean_background = np.zeros(256)
    mean_foreground = np.zeros(256)

    mean_background[valid] = sum_background[valid] / weight_background[valid]

    mean_foreground[valid] = (sum_total - sum_background[valid]) / weight_foreground[
        valid
    ]

    between_class_variance = np.zeros(256)

    between_class_variance[valid] = (
        denominator[valid] * (mean_background[valid] - mean_foreground[valid]) ** 2
    )

    return int(np.argmax(between_class_variance))


def extract_area_features(row):
    """
    Extract thresholded foreground-area measurements.

    Both dark and bright area fractions are retained because
    the meaningful foreground polarity is not yet established.
    """
    image = load_stacked_scan(row)

    features = {}

    for threshold in THRESHOLDS:
        features[f"dark_area_{threshold}"] = np.mean(image <= threshold)

        features[f"bright_area_{threshold}"] = np.mean(image >= threshold)

    threshold = otsu_threshold(image)

    features["otsu_dark_area"] = np.mean(image <= threshold)

    features["otsu_bright_area"] = np.mean(image >= threshold)

    features["otsu_threshold"] = threshold / 255.0

    # Basic intensity measurements for diagnostic context.
    features["mean_intensity"] = image.mean() / 255.0
    features["std_intensity"] = image.std() / 255.0

    return features


def build_image_feature_table(df):
    rows = []

    for index, row in df.iterrows():
        features = extract_area_features(row)
        features["ID"] = row["ID"]

        rows.append(features)

        if (len(rows) % 25) == 0:
            print(f"  Extracted {len(rows)}/{len(df)} observations")

    return pd.DataFrame(rows)


# ============================================================
# MODEL PIPELINES
# ============================================================

CATEGORICAL_COLUMNS = [
    "PlantNumber",
    "Genotype",
    "Stage",
    "Side",
]

NUMERIC_COLUMNS = [
    "Start",
    "End",
    "strip_count",
]

AREA_COLUMNS = (
    [f"dark_area_{t}" for t in THRESHOLDS]
    + [f"bright_area_{t}" for t in THRESHOLDS]
    + [
        "otsu_dark_area",
        "otsu_bright_area",
        "otsu_threshold",
        "mean_intensity",
        "std_intensity",
    ]
)


def make_metadata_model():
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("regressor", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def make_area_model():
    preprocessing = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("regressor", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


def make_combined_model():
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    area_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    preprocessing = ColumnTransformer(
        [
            ("categorical", categorical_pipeline, CATEGORICAL_COLUMNS),
            ("numeric", numeric_pipeline, NUMERIC_COLUMNS),
            ("image_area", area_pipeline, AREA_COLUMNS),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("regressor", Ridge(alpha=RIDGE_ALPHA)),
        ]
    )


# ============================================================
# METRICS
# ============================================================


def calculate_metrics(y_true, y_pred):
    return {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("BASELINE COMPARISON")
    print("=" * 65)

    df = pd.read_csv(TRAIN_CSV)

    df["strip_count"] = df["End"].astype(int) - df["Start"].astype(int) + 1

    df["RootVolume"] = pd.to_numeric(df["RootVolume"], errors="raise")

    print(f"Observations: {len(df)}")
    print(f"Folders: {df['FolderName'].nunique()}")

    # --------------------------------------------------------
    # Extract image features once.
    # --------------------------------------------------------

    feature_path = OUTPUT_DIR / "image_area_features.csv"

    if feature_path.exists():
        print("\nLoading cached image features...")
        image_features = pd.read_csv(feature_path)

    else:
        print("\nExtracting image-area features...")

        image_features = build_image_feature_table(df)

        image_features.to_csv(feature_path, index=False)

        print(f"Saved: {feature_path}")

    df = df.merge(image_features, on="ID", how="left", validate="one_to_one")

    if df[AREA_COLUMNS].isna().any().any():
        raise ValueError("Image feature table contains missing values.")

    # --------------------------------------------------------
    # Grouped cross-validation.
    # --------------------------------------------------------

    splitter = GroupKFold(n_splits=N_SPLITS)

    groups = df["FolderName"]
    y = df["RootVolume"].to_numpy()

    experiments = {
        "mean_baseline": None,
        "metadata_ridge": make_metadata_model(),
        "image_area_ridge": make_area_model(),
        "metadata_plus_area": make_combined_model(),
    }

    oof_predictions = []
    fold_metrics = []

    print("\nStarting grouped cross-validation...")
    print(f"Folds: {N_SPLITS}")
    print("Grouping: FolderName")

    for fold, (train_idx, val_idx) in enumerate(splitter.split(df, y, groups), start=1):
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        y_train = y[train_idx]
        y_val = y[val_idx]

        assert set(train_df["FolderName"]).isdisjoint(set(val_df["FolderName"]))

        print("\n" + "-" * 65)
        print(f"FOLD {fold}")
        print(f"Train: {len(train_df)} | Validation: {len(val_df)}")

        for name, model in experiments.items():
            if name == "mean_baseline":
                prediction = np.full(len(val_df), y_train.mean())

            else:
                if name == "metadata_ridge":
                    columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

                elif name == "image_area_ridge":
                    columns = AREA_COLUMNS

                else:
                    columns = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS + AREA_COLUMNS

                model.fit(train_df[columns], y_train)

                prediction = model.predict(val_df[columns])

            metrics = calculate_metrics(y_val, prediction)

            fold_metrics.append(
                {
                    "fold": fold,
                    "experiment": name,
                    **metrics,
                }
            )

            for observation_id, actual, predicted in zip(
                val_df["ID"], y_val, prediction
            ):
                oof_predictions.append(
                    {
                        "ID": observation_id,
                        "fold": fold,
                        "experiment": name,
                        "actual": actual,
                        "predicted": predicted,
                    }
                )

            print(
                f"{name:24s} "
                f"RMSE={metrics['rmse']:.4f} "
                f"MAE={metrics['mae']:.4f} "
                f"R2={metrics['r2']:.4f}"
            )

    # --------------------------------------------------------
    # Save and summarize.
    # --------------------------------------------------------

    fold_df = pd.DataFrame(fold_metrics)
    oof_df = pd.DataFrame(oof_predictions)

    fold_df.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False)

    oof_df.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    summary = (
        fold_df.groupby("experiment")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
        )
        .reset_index()
    )

    pooled_rows = []

    for name, group in oof_df.groupby("experiment"):
        metrics = calculate_metrics(group["actual"], group["predicted"])

        pooled_rows.append(
            {
                "experiment": name,
                **metrics,
            }
        )

    pooled = pd.DataFrame(pooled_rows)

    summary.to_csv(OUTPUT_DIR / "fold_summary.csv", index=False)

    pooled.to_csv(OUTPUT_DIR / "pooled_summary.csv", index=False)

    print("\n" + "=" * 65)
    print("FOLD-AVERAGED RESULTS")
    print("=" * 65)
    print(summary.to_string(index=False))

    print("\n" + "=" * 65)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 65)
    print(pooled.to_string(index=False))

    print("\nResults saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
