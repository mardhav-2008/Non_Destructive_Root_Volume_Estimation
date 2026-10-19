from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent

train = pd.read_csv(PROJECT_ROOT / "Train.csv")


# ============================================================
# 1. Mean baseline
# ============================================================

y = train["RootVolume"]

mean_prediction = np.full(len(y), y.mean())

mean_rmse = np.sqrt(mean_squared_error(y, mean_prediction))

print("Mean baseline")
print(f"RMSE: {mean_rmse:.4f}")


# ============================================================
# 2. Genotype mean baseline
# ============================================================

# Calculate genotype means using ONLY the training data.
genotype_means = train.groupby("Genotype")["RootVolume"].mean()

genotype_prediction = train["Genotype"].map(genotype_means)

genotype_rmse = np.sqrt(mean_squared_error(y, genotype_prediction))

print()
print("Genotype mean baseline")
print(f"RMSE: {genotype_rmse:.4f}")


# ============================================================
# 3. Metadata regression
# ============================================================

train = train.copy()

train["IntervalLength"] = train["End"] - train["Start"] + 1


features = [
    "PlantNumber",
    "Side",
    "Start",
    "End",
    "IntervalLength",
    "Genotype",
    "Stage",
]

X = train[features]
y = train["RootVolume"]


categorical = [
    "Side",
    "Genotype",
    "Stage",
]

numeric = [
    "PlantNumber",
    "Start",
    "End",
    "IntervalLength",
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
        (
            "numeric",
            "passthrough",
            numeric,
        ),
    ]
)


model = make_pipeline(preprocessor, LinearRegression())


# ============================================================
# 5-fold cross-validation
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model.fit(
        X_train,
        y_train,
    )

    prediction = model.predict(X_val)

    rmse = np.sqrt(
        mean_squared_error(
            y_val,
            prediction,
        )
    )

    scores.append(rmse)

    print(f"Fold {fold}: RMSE = {rmse:.4f}")


print()
print("Metadata regression")
print(f"Mean RMSE: {np.mean(scores):.4f}")
print(f"Std RMSE:  {np.std(scores):.4f}")
