import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Load
# ============================================================

df = pd.read_csv("observation_yolo_features.csv")

TARGET = "RootVolume"
GROUP = "FolderName"

y = df[TARGET]
groups = df[GROUP]


# ============================================================
# Feature groups
# ============================================================

metadata_features = [
    "PlantNumber",
    "Start",
    "End",
]

categorical_features = [
    "Genotype",
    "Stage",
    "Side",
]

yolo_features = [c for c in df.columns if c.startswith("yolo_")]


# ============================================================
# Preprocessors
# ============================================================

metadata_preprocessor = ColumnTransformer(
    [
        (
            "numeric",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            metadata_features,
        ),
        (
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical_features,
        ),
    ]
)


yolo_preprocessor = ColumnTransformer(
    [
        (
            "yolo",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            yolo_features,
        )
    ]
)


hybrid_preprocessor = ColumnTransformer(
    [
        (
            "metadata_numeric",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            metadata_features,
        ),
        (
            "categorical",
            make_pipeline(
                SimpleImputer(strategy="most_frequent"),
                OneHotEncoder(handle_unknown="ignore"),
            ),
            categorical_features,
        ),
        (
            "yolo",
            make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
            yolo_features,
        ),
    ]
)


# ============================================================
# Models
# ============================================================

models = {
    "Metadata": make_pipeline(metadata_preprocessor, Ridge(alpha=10.0)),
    "YOLO": make_pipeline(yolo_preprocessor, Ridge(alpha=10.0)),
    "Hybrid": make_pipeline(hybrid_preprocessor, Ridge(alpha=10.0)),
}


# ============================================================
# GroupKFold
# ============================================================

gkf = GroupKFold(n_splits=5)

all_results = []


for name, model in models.items():
    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, y, groups=groups), 1):
        model.fit(df.iloc[train_idx], y.iloc[train_idx])

        pred = model.predict(df.iloc[val_idx])

        rmse = np.sqrt(mean_squared_error(y.iloc[val_idx], pred))

        mae = mean_absolute_error(y.iloc[val_idx], pred)

        r2 = r2_score(y.iloc[val_idx], pred)

        fold_results.append(rmse)

        print(f"Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    mean_rmse = np.mean(fold_results)
    std_rmse = np.std(fold_results)

    print(f"\nMean RMSE: {mean_rmse:.4f} +/- {std_rmse:.4f}")

    all_results.append(
        {
            "Model": name,
            "RMSE": mean_rmse,
            "RMSE_std": std_rmse,
        }
    )


# ============================================================
# Comparison
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame(all_results)

print(comparison.to_string(index=False))
