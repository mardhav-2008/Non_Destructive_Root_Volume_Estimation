from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# Paths and configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DETECTIONS_PATH = PROJECT_ROOT / "outputs/yolo/yolo_detections.csv"
TRAIN_PATH = PROJECT_ROOT / "Train.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs/yolo/plant_number_predictions.csv"

N_BINS = 20
RANDOM_STATE = 42
N_SPLITS = 5

# ============================================================
# Load data
# ============================================================

detections = pd.read_csv(DETECTIONS_PATH)
train = pd.read_csv(TRAIN_PATH)

detections["Slice"] = pd.to_numeric(detections["Slice"])
train["Start"] = pd.to_numeric(train["Start"])
train["End"] = pd.to_numeric(train["End"])

print("=" * 70)
print("PLANT NUMBER PREDICTION FROM YOLO SPATIAL FEATURES")
print("=" * 70)

# ============================================================
# Feature extraction
# ============================================================


def extract_features(obs, det):
    """Build spatial features for one observation."""

    x = det["x_center_norm"].to_numpy(dtype=float)
    area = det["mask_area"].to_numpy(dtype=float)
    conf = det["confidence"].to_numpy(dtype=float)

    valid = np.isfinite(x) & np.isfinite(area)
    x = x[valid]
    area = area[valid]
    conf = conf[valid]

    features = {}

    # Basic detection statistics
    features["detection_count"] = len(x)
    features["mean_area"] = np.mean(area) if len(area) else 0
    features["max_area"] = np.max(area) if len(area) else 0
    features["std_area"] = np.std(area) if len(area) else 0
    features["mean_confidence"] = np.nanmean(conf) if len(conf) else 0

    # Horizontal distribution
    if len(x):
        features["x_mean"] = np.mean(x)
        features["x_std"] = np.std(x)
        features["x_median"] = np.median(x)
        features["x_q25"] = np.quantile(x, 0.25)
        features["x_q75"] = np.quantile(x, 0.75)
    else:
        for key in ["x_mean", "x_std", "x_median", "x_q25", "x_q75"]:
            features[key] = 0

    # Detection count and mask area in horizontal bins
    count_hist, _ = np.histogram(x, bins=N_BINS, range=(0, 1))

    area_hist, _ = np.histogram(x, bins=N_BINS, range=(0, 1), weights=area)

    for i in range(N_BINS):
        features[f"count_bin_{i}"] = count_hist[i]
        features[f"area_bin_{i}"] = area_hist[i]

    # Normalized spatial profile
    total = area_hist.sum()

    if total > 0:
        normalized = area_hist / total
    else:
        normalized = np.zeros(N_BINS)

    for i, value in enumerate(normalized):
        features[f"normalized_area_bin_{i}"] = value

    # Detection distribution by scan depth
    if len(det):
        depth = det["Slice"].to_numpy(dtype=float)
        depth_span = max(obs["End"] - obs["Start"] + 1, 1)

        relative_depth = (depth - obs["Start"]) / depth_span

        features["mean_relative_depth"] = np.mean(relative_depth)
        features["std_relative_depth"] = np.std(relative_depth)
    else:
        features["mean_relative_depth"] = 0
        features["std_relative_depth"] = 0

    return features


# ============================================================
# Build observation-level dataset
# ============================================================

rows = []

for _, obs in train.iterrows():
    subset = detections[
        (detections["FolderName"] == obs["FolderName"])
        & (detections["Side"] == obs["Side"])
        & (detections["Slice"] >= obs["Start"])
        & (detections["Slice"] <= obs["End"])
    ]

    features = extract_features(obs, subset)

    rows.append(
        {
            "FolderName": obs["FolderName"],
            "PlantNumber": obs["PlantNumber"],
            **features,
        }
    )

dataset = pd.DataFrame(rows)

feature_cols = [
    col for col in dataset.columns if col not in ["FolderName", "PlantNumber"]
]

X = dataset[feature_cols]
y = dataset["PlantNumber"].astype(int)
groups = dataset["FolderName"]

print("\nObservations:", len(dataset))
print("Unique folders:", groups.nunique())
print("Features:", len(feature_cols))
print("\nPlantNumber distribution:")
print(y.value_counts().sort_index())

# ============================================================
# GroupKFold evaluation
# ============================================================

cv = GroupKFold(n_splits=N_SPLITS)

oof_predictions = np.full(len(dataset), -1, dtype=int)
dummy_predictions = np.full(len(dataset), -1, dtype=int)

fold_scores = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), start=1):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Majority-class baseline
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    dummy_pred = dummy.predict(X_val)

    # YOLO spatial classifier
    model = make_pipeline(
        SimpleImputer(strategy="constant", fill_value=0),
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_val)

    oof_predictions[val_idx] = predictions
    dummy_predictions[val_idx] = dummy_pred

    accuracy = accuracy_score(y_val, predictions)
    balanced = balanced_accuracy_score(y_val, predictions)
    dummy_accuracy = accuracy_score(y_val, dummy_pred)

    fold_scores.append(accuracy)

    print(
        f"Fold {fold}: "
        f"YOLO accuracy={accuracy:.4f}, "
        f"balanced={balanced:.4f}, "
        f"dummy={dummy_accuracy:.4f}"
    )

# ============================================================
# Overall results
# ============================================================

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print("YOLO accuracy:", round(accuracy_score(y, oof_predictions), 4))

print("YOLO balanced accuracy:", round(balanced_accuracy_score(y, oof_predictions), 4))

print("Dummy accuracy:", round(accuracy_score(y, dummy_predictions), 4))

print("\nClassification report:")
print(
    classification_report(
        y,
        oof_predictions,
        labels=sorted(y.unique()),
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y,
        oof_predictions,
        labels=sorted(y.unique()),
    )
)

# ============================================================
# Save predictions
# ============================================================

dataset["ActualPlantNumber"] = y
dataset["PredictedPlantNumber"] = oof_predictions

dataset.to_csv(OUTPUT_PATH, index=False)

print("\nSaved:", OUTPUT_PATH)
print("\nANALYSIS COMPLETE")
