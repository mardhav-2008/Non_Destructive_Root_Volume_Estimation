import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


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

metadata = [
    "PlantNumber",
    "Start",
    "End",
    "Genotype",
    "Stage",
    "Side",
]

persistence = [
    "yolo_n_slices",
    "yolo_detected_slices",
    "yolo_detection_coverage",
    "yolo_longest_detection_run",
    "yolo_first_detection",
    "yolo_last_detection",
]

area = [
    "yolo_area_mean",
    "yolo_area_std",
    "yolo_area_min",
    "yolo_area_max",
    "yolo_area_median",
    "yolo_max_area_mean",
    "yolo_max_area_max",
    "yolo_area_trend",
]

geometry = [
    "yolo_width_mean",
    "yolo_width_std",
    "yolo_width_max",
    "yolo_height_mean",
    "yolo_height_std",
    "yolo_height_max",
]

confidence = [
    "yolo_conf_mean",
    "yolo_conf_max",
]

objects = [
    "yolo_objects_mean",
    "yolo_objects_std",
    "yolo_objects_min",
    "yolo_objects_max",
    "yolo_objects_median",
]


# ============================================================
# Model builder
# ============================================================


def build_model(numeric_features, categorical_features):

    preprocessor = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                numeric_features,
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

    return make_pipeline(preprocessor, Ridge(alpha=10.0))


# ============================================================
# Experiments
# ============================================================

experiments = {
    "Metadata": {
        "numeric": [
            "PlantNumber",
            "Start",
            "End",
        ],
        "categorical": [
            "Genotype",
            "Stage",
            "Side",
        ],
    },
    "YOLO persistence": {
        "numeric": persistence,
        "categorical": [],
    },
    "YOLO area": {
        "numeric": area,
        "categorical": [],
    },
    "YOLO geometry": {
        "numeric": geometry,
        "categorical": [],
    },
    "YOLO confidence": {
        "numeric": confidence,
        "categorical": [],
    },
    "YOLO objects": {
        "numeric": objects,
        "categorical": [],
    },
    "YOLO all": {
        "numeric": (persistence + area + geometry + confidence + objects),
        "categorical": [],
    },
    "Metadata + persistence": {
        "numeric": [
            "PlantNumber",
            "Start",
            "End",
        ]
        + persistence,
        "categorical": [
            "Genotype",
            "Stage",
            "Side",
        ],
    },
    "Metadata + area": {
        "numeric": [
            "PlantNumber",
            "Start",
            "End",
        ]
        + area,
        "categorical": [
            "Genotype",
            "Stage",
            "Side",
        ],
    },
    "Metadata + geometry": {
        "numeric": [
            "PlantNumber",
            "Start",
            "End",
        ]
        + geometry,
        "categorical": [
            "Genotype",
            "Stage",
            "Side",
        ],
    },
    "Metadata + confidence": {
        "numeric": [
            "PlantNumber",
            "Start",
            "End",
        ]
        + confidence,
        "categorical": [
            "Genotype",
            "Stage",
            "Side",
        ],
    },
    "Metadata + objects": {
        "numeric": [
            "PlantNumber",
            "Start",
            "End",
        ]
        + objects,
        "categorical": [
            "Genotype",
            "Stage",
            "Side",
        ],
    },
    "Metadata + YOLO": {
        "numeric": [
            "PlantNumber",
            "Start",
            "End",
        ]
        + persistence
        + area
        + geometry
        + confidence
        + objects,
        "categorical": [
            "Genotype",
            "Stage",
            "Side",
        ],
    },
}


# ============================================================
# Same folds for every experiment
# ============================================================

gkf = GroupKFold(n_splits=5)

results = []

print("=" * 75)
print("YOLO FEATURE ABLATION")
print("=" * 75)


for name, feature_set in experiments.items():
    X = df[feature_set["numeric"] + feature_set["categorical"]]

    model = build_model(
        feature_set["numeric"],
        feature_set["categorical"],
    )

    fold_rmse = []

    for train_idx, val_idx in gkf.split(X, y, groups=groups):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, pred))

        fold_rmse.append(rmse)

    mean_rmse = np.mean(fold_rmse)
    std_rmse = np.std(fold_rmse)

    results.append(
        {
            "Experiment": name,
            "RMSE": mean_rmse,
            "Std": std_rmse,
            "Fold1": fold_rmse[0],
            "Fold2": fold_rmse[1],
            "Fold3": fold_rmse[2],
            "Fold4": fold_rmse[3],
            "Fold5": fold_rmse[4],
        }
    )


# ============================================================
# Results
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values("RMSE")

print("\n" + "=" * 75)
print("RESULTS")
print("=" * 75)

print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


# ============================================================
# Area trend distribution
# ============================================================

print("\n" + "=" * 75)
print("AREA TREND")
print("=" * 75)

print(df["yolo_area_trend"].describe())

print("\nLargest absolute trends:")

print(
    df[
        [
            "ID",
            "FolderName",
            "Side",
            "Start",
            "End",
            "yolo_area_trend",
            "RootVolume",
        ]
    ]
    .assign(abs_trend=lambda x: x["yolo_area_trend"].abs())
    .sort_values("abs_trend", ascending=False)
    .head(10)
    .to_string(index=False)
)
