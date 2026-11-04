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

df = pd.read_csv(PROJECT_ROOT / "Train.csv")

df["IntervalLength"] = df["End"] - df["Start"] + 1

y = df["RootVolume"].to_numpy()


# ============================================================
# Cross-validation setup
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)


# Store scores
results = {
    "Overall mean": [],
    "Genotype mean": [],
    "PlantNumber mean": [],
    "Genotype + PlantNumber": [],
    "Metadata regression": [],
}


# ============================================================
# Helper
# ============================================================


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ============================================================
# Cross-validation
# ============================================================

for fold, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
    train = df.iloc[train_idx]
    val = df.iloc[val_idx]

    y_train = train["RootVolume"].to_numpy()
    y_val = val["RootVolume"].to_numpy()

    # --------------------------------------------------------
    # 1. Overall mean
    # --------------------------------------------------------

    prediction = np.full(len(val), y_train.mean())

    results["Overall mean"].append(rmse(y_val, prediction))

    # --------------------------------------------------------
    # 2. Genotype mean
    # --------------------------------------------------------

    genotype_means = train.groupby("Genotype")["RootVolume"].mean()

    global_mean = y_train.mean()

    prediction = val["Genotype"].map(genotype_means).fillna(global_mean)

    results["Genotype mean"].append(rmse(y_val, prediction.to_numpy()))

    # --------------------------------------------------------
    # 3. PlantNumber mean
    # --------------------------------------------------------

    plant_means = train.groupby("PlantNumber")["RootVolume"].mean()

    prediction = val["PlantNumber"].map(plant_means).fillna(global_mean)

    results["PlantNumber mean"].append(rmse(y_val, prediction.to_numpy()))

    # --------------------------------------------------------
    # 4. Genotype + PlantNumber mean
    # --------------------------------------------------------

    combo_means = train.groupby(["Genotype", "PlantNumber"])["RootVolume"].mean()

    prediction = []

    for _, row in val.iterrows():
        key = (row["Genotype"], row["PlantNumber"])

        if key in combo_means.index:
            prediction.append(combo_means.loc[key])
        else:
            prediction.append(global_mean)

    results["Genotype + PlantNumber"].append(rmse(y_val, np.array(prediction)))

    # --------------------------------------------------------
    # 5. Metadata regression
    # --------------------------------------------------------

    features = [
        "PlantNumber",
        "Side",
        "Start",
        "End",
        "IntervalLength",
        "Genotype",
        "Stage",
    ]

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

    model.fit(train[features], y_train)

    prediction = model.predict(val[features])

    results["Metadata regression"].append(rmse(y_val, prediction))

    print(f"Fold {fold}")

    for name in results:
        print(f"  {name:25s}: {results[name][-1]:.4f}")

    print()


# ============================================================
# Final summary
# ============================================================

print("=" * 65)
print("FINAL RESULTS")
print("=" * 65)

for name, scores in results.items():
    print(f"{name:25s} RMSE = {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
