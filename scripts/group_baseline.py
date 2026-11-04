from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

PROJECT_ROOT = Path(__file__).resolve().parent
df = pd.read_csv(PROJECT_ROOT / "Train.csv")


def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }


def print_results(name, results):
    print(f"\n{name}")
    print("-" * 60)

    for metric in ["MAE", "MSE", "RMSE", "R2"]:
        values = [r[metric] for r in results]

        print(f"{metric:5s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


# ============================================================
# MODELS
# ============================================================


def predict_overall_mean(train, val):
    return np.full(len(val), train["RootVolume"].mean())


def predict_genotype_mean(train, val):
    means = train.groupby("Genotype")["RootVolume"].mean()

    return val["Genotype"].map(means).fillna(train["RootVolume"].mean()).to_numpy()


def predict_metadata(train, val):

    features = [
        "Genotype",
        "PlantNumber",
        "Side",
        "Stage",
        "Start",
        "End",
    ]

    categorical = [
        "Genotype",
        "Side",
        "Stage",
    ]

    _numerical = [
        "PlantNumber",
        "Start",
        "End",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
        ],
        remainder="passthrough",
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=1.0)),
        ]
    )

    model.fit(
        train[features],
        train["RootVolume"],
    )

    return model.predict(val[features])


# ============================================================
# VALIDATION
# ============================================================


def run_validation(splitter, name):

    results = {
        "Overall Mean": [],
        "Genotype Mean": [],
        "Metadata Ridge": [],
    }

    print("\n")
    print("=" * 70)
    print(name)
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(splitter, start=1):
        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        y_true = val["RootVolume"].to_numpy()

        predictions = {
            "Overall Mean": predict_overall_mean(train, val),
            "Genotype Mean": predict_genotype_mean(train, val),
            "Metadata Ridge": predict_metadata(train, val),
        }

        print(f"\nFold {fold}")

        for model_name, y_pred in predictions.items():
            metrics = evaluate(y_true, y_pred)

            results[model_name].append(metrics)

            print(
                f"{model_name:16s} "
                f"MAE={metrics['MAE']:.4f} "
                f"MSE={metrics['MSE']:.4f} "
                f"RMSE={metrics['RMSE']:.4f} "
                f"R2={metrics['R2']:.4f}"
            )

    for model_name, model_results in results.items():
        print_results(
            model_name,
            model_results,
        )


# ============================================================
# ROW-WISE K-FOLD
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

run_validation(
    kf.split(df),
    "ROW-WISE K-FOLD",
)


# ============================================================
# GROUP K-FOLD
# ============================================================

gkf = GroupKFold(
    n_splits=5,
)

run_validation(
    gkf.split(
        df,
        groups=df["FolderName"],
    ),
    "GROUP K-FOLD — UNSEEN SCANS",
)
