import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


# ============================================================
# Configuration
# ============================================================

TRAIN_FILE = "Train.csv"
PROFILE_FILE = "observation_yolo_profile_features.csv"

N_SPLITS = 5
RIDGE_ALPHA = 10


# ============================================================
# Load data
# ============================================================

train = pd.read_csv(TRAIN_FILE)
profile = pd.read_csv(PROFILE_FILE)


# ============================================================
# Observation identity
# ============================================================

merge_columns = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
]


# ============================================================
# Feature groups
# ============================================================

metadata_features = [
    "PlantNumber",
    "Start",
    "End",
    "Genotype",
    "Stage",
    "Side",
]

metadata_categorical = [
    "Genotype",
    "Stage",
    "Side",
]

metadata_numerical = [
    "PlantNumber",
    "Start",
    "End",
]


profile_features = [
    column for column in profile.columns if column.startswith("profile_")
]


# ============================================================
# Keep ONLY profile features from the profile table
# ============================================================

profile_subset = profile[merge_columns + profile_features].copy()


# ============================================================
# Keep metadata ONLY from Train.csv
# ============================================================

train_metadata = train[
    merge_columns
    + [
        "RootVolume",
        "Genotype",
        "Stage",
    ]
].copy()


# ============================================================
# Merge
# ============================================================

df = train_metadata.merge(
    profile_subset,
    on=merge_columns,
    how="left",
)


# ============================================================
# Basic information
# ============================================================

print("=" * 70)
print("PROFILE BASELINE EXPERIMENT")
print("=" * 70)

print(f"Train rows:   {len(train)}")
print(f"Profile rows: {len(profile)}")
print(f"Merged rows:  {len(df)}")

print()


# ============================================================
# Display feature groups
# ============================================================

print("Metadata features:")
for feature in metadata_features:
    print(f"  - {feature}")

print()

print("Profile features:")
for feature in profile_features:
    print(f"  - {feature}")

print()


# ============================================================
# Validate columns
# ============================================================

required_columns = merge_columns + ["RootVolume"] + metadata_features + profile_features

missing_columns = [column for column in required_columns if column not in df.columns]

if missing_columns:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(f"  - {column}" for column in missing_columns)
    )


# ============================================================
# Validate merge
# ============================================================

if len(df) != len(train):
    raise ValueError(
        "Merge changed the number of rows!\n"
        f"Train rows:  {len(train)}\n"
        f"Merged rows: {len(df)}"
    )


# ============================================================
# Explicitly enforce data types
# ============================================================

# Metadata categorical columns
for column in metadata_categorical:
    df[column] = df[column].astype(str)


# Metadata numerical columns
for column in metadata_numerical:
    df[column] = pd.to_numeric(
        df[column],
        errors="raise",
    )


# Target
df["RootVolume"] = pd.to_numeric(
    df["RootVolume"],
    errors="raise",
)


# Profile features must all be numeric
for column in profile_features:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce",
    )


# ============================================================
# Check profile missingness
# ============================================================

missing_profile = df[profile_features].isna().sum()

if missing_profile.sum() > 0:
    print("Missing profile values:")

    print(missing_profile[missing_profile > 0])

    print()


# ============================================================
# Evaluation function
# ============================================================


def evaluate(
    features,
    categorical_features,
    numerical_features,
    name,
):

    X = df[features].copy()

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df["RootVolume"].to_numpy(dtype=float)

    # --------------------------------------------------------
    # Groups
    # --------------------------------------------------------

    groups = df["FolderName"].astype(str).to_numpy()

    # --------------------------------------------------------
    # Make sure numerical features are numeric
    # --------------------------------------------------------

    for column in numerical_features:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # Make sure categorical features are strings
    # --------------------------------------------------------

    for column in categorical_features:
        X[column] = X[column].astype(str)

    # --------------------------------------------------------
    # Numerical preprocessing
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    # --------------------------------------------------------
    # Categorical preprocessing
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    # --------------------------------------------------------
    # Column transformer
    # --------------------------------------------------------

    transformers = []

    if numerical_features:
        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features,
            )
        )

    if categorical_features:
        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # --------------------------------------------------------
    # Full model
    # --------------------------------------------------------

    model = Pipeline(
        [
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "ridge",
                Ridge(alpha=RIDGE_ALPHA),
            ),
        ]
    )

    # --------------------------------------------------------
    # GroupKFold
    # --------------------------------------------------------

    gkf = GroupKFold(n_splits=N_SPLITS)

    rmses = []
    maes = []
    r2s = []

    print("=" * 70)
    print(name)
    print("=" * 70)

    # --------------------------------------------------------
    # Cross-validation
    # --------------------------------------------------------

    for fold, (
        train_idx,
        val_idx,
    ) in enumerate(
        gkf.split(
            X,
            y,
            groups,
        ),
        start=1,
    ):
        X_train = X.iloc[train_idx].copy()

        X_val = X.iloc[val_idx].copy()

        y_train = y[train_idx]
        y_val = y[val_idx]

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        pred = model.predict(X_val)

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        rmse = np.sqrt(
            mean_squared_error(
                y_val,
                pred,
            )
        )

        mae = mean_absolute_error(
            y_val,
            pred,
        )

        r2 = r2_score(
            y_val,
            pred,
        )

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        print(f"Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    mean_rmse = np.mean(rmses)
    std_rmse = np.std(rmses)

    mean_mae = np.mean(maes)
    std_mae = np.std(maes)

    mean_r2 = np.mean(r2s)
    std_r2 = np.std(r2s)

    print()

    print(f"{name} summary:")

    print(f"  RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")

    print(f"  MAE:  {mean_mae:.4f} ± {std_mae:.4f}")

    print(f"  R²:   {mean_r2:.4f} ± {std_r2:.4f}")

    print()

    return mean_rmse


# ============================================================
# EXPERIMENT 1
# Metadata only
# ============================================================

metadata_rmse = evaluate(
    features=metadata_features,
    categorical_features=metadata_categorical,
    numerical_features=metadata_numerical,
    name="1. Metadata only",
)


# ============================================================
# EXPERIMENT 2
# Metadata + YOLO profile
# ============================================================

hybrid_features = metadata_features + profile_features

hybrid_categorical = metadata_categorical

hybrid_numerical = metadata_numerical + profile_features

hybrid_rmse = evaluate(
    features=hybrid_features,
    categorical_features=hybrid_categorical,
    numerical_features=hybrid_numerical,
    name="2. Metadata + YOLO Profile",
)


# ============================================================
# EXPERIMENT 3
# YOLO profile only
# ============================================================

profile_rmse = evaluate(
    features=profile_features,
    categorical_features=[],
    numerical_features=profile_features,
    name="3. YOLO Profile only",
)


# ============================================================
# Final comparison
# ============================================================

print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(f"Metadata only:           {metadata_rmse:.4f}")

print(f"Metadata + Profile:      {hybrid_rmse:.4f}")

print(f"Profile only:            {profile_rmse:.4f}")

print()

improvement = metadata_rmse - hybrid_rmse

print(f"Profile improvement over metadata: {improvement:+.4f} RMSE")

print()

if improvement > 0:
    print("RESULT: YOLO profile features improve the metadata baseline.")

elif improvement < 0:
    print("RESULT: YOLO profile features hurt the metadata baseline.")

else:
    print("RESULT: YOLO profile features make no difference.")
