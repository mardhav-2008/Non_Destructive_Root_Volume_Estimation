import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# Load
# ============================================================

df = pd.read_csv("observation_yolo_features.csv")

target = "RootVolume"

# Don't give the model identifiers / target / categorical metadata.
drop_cols = [
    "ID",
    "FolderName",
    "RootVolume",
    "Genotype",
    "Stage",
    "Side",
]

feature_cols = [c for c in df.columns if c not in drop_cols]

X = df[feature_cols]
y = df[target]

groups = df["FolderName"]


# ============================================================
# Model
# ============================================================

model = make_pipeline(
    SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
)


# ============================================================
# GroupKFold
# ============================================================

gkf = GroupKFold(n_splits=5)

results = []

print("=" * 70)
print("YOLO FEATURE BASELINE")
print("=" * 70)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]

    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, pred))
    mae = mean_absolute_error(y_val, pred)
    r2 = r2_score(y_val, pred)

    results.append(
        {
            "fold": fold,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
        }
    )

    print(f"Fold {fold}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")


# ============================================================
# Summary
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"RMSE: {results_df['RMSE'].mean():.4f} +/- {results_df['RMSE'].std():.4f}")

print(f"MAE:  {results_df['MAE'].mean():.4f} +/- {results_df['MAE'].std():.4f}")

print(f"R²:   {results_df['R2'].mean():.4f} +/- {results_df['R2'].std():.4f}")


# ============================================================
# Feature importance
#
# Ridge coefficients tell us which features matter most.
# ============================================================

model.fit(X, y)

ridge = model.named_steps["ridge"]
scaler = model.named_steps["standardscaler"]

coef = pd.Series(ridge.coef_, index=feature_cols)

coef = coef.sort_values(key=abs, ascending=False)

print("\n" + "=" * 70)
print("TOP YOLO FEATURES")
print("=" * 70)

print(coef.head(15))
