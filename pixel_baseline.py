from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor


PROJECT_ROOT = Path(__file__).resolve().parent

df = pd.read_csv(PROJECT_ROOT / "observation_pixel_features.csv")


# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

exclude = [
    "ID",
    "FolderName",
    "RootVolume",
    "Genotype",
    "Side",
    "Stage",
]

X = df.drop(columns=exclude)
y = df["RootVolume"]


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------


def evaluate(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, rmse, r2


# ------------------------------------------------------------
# Grouped validation
# ------------------------------------------------------------

gkf = GroupKFold(n_splits=5)

results = []

for fold, (train_idx, val_idx) in enumerate(
    gkf.split(X, y, groups=df["FolderName"]), start=1
):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Handle missing values
    imputer = SimpleImputer(strategy="median")

    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=3,
    )

    model.fit(X_train, y_train)

    prediction = model.predict(X_val)

    mae, mse, rmse, r2 = evaluate(y_val, prediction)

    results.append([mae, mse, rmse, r2])

    print(f"Fold {fold}: MAE={mae:.4f} MSE={mse:.4f} RMSE={rmse:.4f} R2={r2:.4f}")


results = np.array(results)

print()
print("=" * 60)
print("PIXEL FEATURES — UNSEEN SCANS")
print("=" * 60)

for i, name in enumerate(["MAE", "MSE", "RMSE", "R2"]):
    print(f"{name:5s}: {results[:, i].mean():.4f} +/- {results[:, i].std():.4f}")
