import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


# ============================================================
# CONFIG
# ============================================================

TRACK_FILE = "yolo_tracks.csv"
TRAIN_FILE = "Train.csv"

N_X_BINS = 20
TOP_K = 10

RANDOM_STATE = 42


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("TRACK SPATIAL PROFILE FEATURES")
print("=" * 70)

tracks = pd.read_csv(TRACK_FILE)
train = pd.read_csv(TRAIN_FILE)

print(f"Track detections: {len(tracks):,}")
print(f"Train observations: {len(train):,}")


# ============================================================
# BUILD TRACK-LEVEL DATA
# ============================================================

track_cols = [
    "FolderName",
    "Side",
    "TrackID",
]

track_summary = (
    tracks.groupby(track_cols)
    .agg(
        track_length=("Slice", "count"),
        area_sum=("mask_area", "sum"),
        area_mean=("mask_area", "mean"),
        area_max=("mask_area", "max"),
        x_mean=("x_center_norm", "mean"),
        x_min=("x_center_norm", "min"),
        x_max=("x_center_norm", "max"),
        confidence_mean=("confidence", "mean"),
        first_slice=("Slice", "min"),
        last_slice=("Slice", "max"),
    )
    .reset_index()
)

track_summary["x_range"] = track_summary["x_max"] - track_summary["x_min"]

track_summary["depth_span"] = (
    track_summary["last_slice"] - track_summary["first_slice"] + 1
)

track_summary["continuity"] = track_summary["track_length"] / track_summary[
    "depth_span"
].clip(lower=1)

print(f"Tracks: {len(track_summary):,}")


# ============================================================
# JOIN TRACKS TO OBSERVATIONS
#
# Only detections inside the observation's Start-End
# interval are allowed to contribute.
# ============================================================

merged = tracks.merge(
    train[
        [
            "ID",
            "FolderName",
            "PlantNumber",
            "Side",
            "Start",
            "End",
            "RootVolume",
            "Genotype",
            "Stage",
        ]
    ],
    on=["FolderName", "Side"],
    how="inner",
)

merged = merged[
    (merged["Slice"] >= merged["Start"]) & (merged["Slice"] <= merged["End"])
].copy()

print(f"Observation-matched track detections: {len(merged):,}")


# ============================================================
# OBSERVATION-LEVEL TRACK TABLE
# ============================================================

obs_keys = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
    "RootVolume",
    "Genotype",
    "Stage",
]


obs_track_group_keys = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
    "TrackID",
]


obs_tracks = (
    merged.groupby(obs_track_group_keys)
    .agg(
        track_length=("Slice", "count"),
        area_sum=("mask_area", "sum"),
        area_mean=("mask_area", "mean"),
        area_max=("mask_area", "max"),
        x_mean=("x_center_norm", "mean"),
        x_min=("x_center_norm", "min"),
        x_max=("x_center_norm", "max"),
        confidence_mean=("confidence", "mean"),
        first_slice=("Slice", "min"),
        last_slice=("Slice", "max"),
    )
    .reset_index()
)

# Add metadata back to each observation-track row.
obs_metadata = train[
    [
        "ID",
        "FolderName",
        "PlantNumber",
        "Side",
        "Start",
        "End",
        "RootVolume",
        "Genotype",
        "Stage",
    ]
].drop_duplicates()

obs_tracks = obs_tracks.merge(
    obs_metadata,
    on=[
        "ID",
        "FolderName",
        "PlantNumber",
        "Side",
        "Start",
        "End",
    ],
    how="left",
)

obs_tracks["x_range"] = obs_tracks["x_max"] - obs_tracks["x_min"]

obs_tracks["depth_span"] = obs_tracks["last_slice"] - obs_tracks["first_slice"] + 1

obs_tracks["continuity"] = obs_tracks["track_length"] / obs_tracks["depth_span"].clip(
    lower=1
)


# ============================================================
# FEATURE BUILDER
# ============================================================


def percentile_safe(values, q):
    if len(values) == 0:
        return 0.0

    return float(np.percentile(values, q))


def build_features(group):
    row = {}

    areas = group["area_sum"].values
    lengths = group["track_length"].values
    x = group["x_mean"].values
    continuity = group["continuity"].values

    # --------------------------------------------------------
    # Basic track statistics
    # --------------------------------------------------------

    row["track_count"] = len(group)

    row["total_track_area"] = areas.sum() if len(areas) else 0.0

    row["mean_track_area"] = areas.mean() if len(areas) else 0.0

    row["median_track_area"] = np.median(areas) if len(areas) else 0.0

    row["max_track_area"] = areas.max() if len(areas) else 0.0

    row["total_track_length"] = lengths.sum() if len(lengths) else 0.0

    row["mean_track_length"] = lengths.mean() if len(lengths) else 0.0

    row["max_track_length"] = lengths.max() if len(lengths) else 0.0

    row["mean_continuity"] = continuity.mean() if len(continuity) else 0.0

    row["max_continuity"] = continuity.max() if len(continuity) else 0.0

    # --------------------------------------------------------
    # Area distribution
    # --------------------------------------------------------

    for q in [10, 25, 50, 75, 90]:
        row[f"area_q{q}"] = percentile_safe(areas, q)

    # --------------------------------------------------------
    # Top-K tracks by total area
    # --------------------------------------------------------

    sorted_areas = np.sort(areas)[::-1]

    for k in range(1, TOP_K + 1):
        if len(sorted_areas) >= k:
            row[f"top{k}_area"] = sorted_areas[k - 1]
        else:
            row[f"top{k}_area"] = 0.0

    for k in [3, 5, 10]:
        top = sorted_areas[:k]

        row[f"top{k}_area_sum"] = top.sum()

    # --------------------------------------------------------
    # Large-track counts
    # --------------------------------------------------------

    if len(areas):
        med = np.median(areas)

        row["tracks_above_median"] = int(np.sum(areas > med))

        row["tracks_above_2x_median"] = int(np.sum(areas > 2 * med))

        row["tracks_length_ge5"] = int(np.sum(lengths >= 5))

        row["tracks_length_ge10"] = int(np.sum(lengths >= 10))

        row["tracks_length_ge20"] = int(np.sum(lengths >= 20))

    else:
        row["tracks_above_median"] = 0
        row["tracks_above_2x_median"] = 0
        row["tracks_length_ge5"] = 0
        row["tracks_length_ge10"] = 0
        row["tracks_length_ge20"] = 0

    # --------------------------------------------------------
    # Spatial statistics
    # --------------------------------------------------------

    row["mean_track_x"] = x.mean() if len(x) else 0.0

    row["std_track_x"] = x.std() if len(x) > 1 else 0.0

    row["min_track_x"] = x.min() if len(x) else 0.0

    row["max_track_x"] = x.max() if len(x) else 0.0

    # Area-weighted horizontal position
    if len(areas) and areas.sum() > 0:
        row["area_weighted_x"] = np.average(x, weights=areas)

    else:
        row["area_weighted_x"] = 0.0

    # --------------------------------------------------------
    # Spatial area profile
    #
    # 20 horizontal bins.
    #
    # Each track contributes its total area to the bin
    # corresponding to its mean horizontal position.
    # --------------------------------------------------------

    profile = np.zeros(N_X_BINS, dtype=float)

    for area, xpos in zip(areas, x):
        bin_idx = int(xpos * N_X_BINS)

        bin_idx = min(max(bin_idx, 0), N_X_BINS - 1)

        profile[bin_idx] += area

    # Raw profile
    for i, value in enumerate(profile):
        row[f"x_area_{i:02d}"] = value

    # Normalized profile
    total = profile.sum()

    if total > 0:
        normalized = profile / total

    else:
        normalized = profile.copy()

    for i, value in enumerate(normalized):
        row[f"x_area_norm_{i:02d}"] = value

    # --------------------------------------------------------
    # Spatial concentration
    # --------------------------------------------------------

    if total > 0:
        p = normalized[normalized > 0]

        row["spatial_entropy"] = -np.sum(p * np.log(p))

        row["spatial_peak_fraction"] = normalized.max()

        row["spatial_nonzero_bins"] = np.sum(normalized > 0)

    else:
        row["spatial_entropy"] = 0.0
        row["spatial_peak_fraction"] = 0.0
        row["spatial_nonzero_bins"] = 0

    return pd.Series(row)


# ============================================================
# BUILD OBSERVATION FEATURES
# ============================================================

print("\nBuilding observation features...")

feature_rows = []

group_keys = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
]

for keys, group in obs_tracks.groupby(group_keys, dropna=False):
    base = dict(zip(group_keys, keys))

    features = build_features(group)

    base.update(features)

    # Recover target and metadata
    first = group.iloc[0]

    base["RootVolume"] = first["RootVolume"]

    base["Genotype"] = first["Genotype"]

    base["Stage"] = first["Stage"]

    feature_rows.append(base)


features = pd.DataFrame(feature_rows)

print(f"Feature rows before restoring no-track observations: {len(features):,}")


# ============================================================
# RESTORE OBSERVATIONS WITH NO TRACKS
#
# The previous version dropped observations that had no
# qualifying YOLO tracks. We want all 386 observations.
# ============================================================

all_obs = train[
    [
        "ID",
        "FolderName",
        "PlantNumber",
        "Side",
        "Start",
        "End",
        "RootVolume",
        "Genotype",
        "Stage",
    ]
].copy()


merge_keys = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
    "RootVolume",
    "Genotype",
    "Stage",
]


features = all_obs.merge(
    features,
    on=merge_keys,
    how="left",
)


# ============================================================
# FILL TRACK FEATURES FOR NO-TRACK OBSERVATIONS
# ============================================================

base_columns = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
    "RootVolume",
    "Genotype",
    "Stage",
]

generated_feature_cols = [c for c in features.columns if c not in base_columns]

features[generated_feature_cols] = features[generated_feature_cols].fillna(0)


print(f"Feature rows after restoring no-track observations: {len(features):,}")

print(f"Observations with no tracks: {np.sum(features['track_count'] == 0):,}")


# ============================================================
# SAVE FEATURES
# ============================================================

features.to_csv("track_spatial_features.csv", index=False)

print("\nSaved:")
print("track_spatial_features.csv")


# ============================================================
# MODELING
# ============================================================

TARGET = "RootVolume"
GROUP = "FolderName"

META_CATEGORICAL = [
    "Genotype",
    "Stage",
    "Side",
]

META_NUMERIC = [
    "PlantNumber",
    "Start",
    "End",
]


metadata = [
    "PlantNumber",
    "Start",
    "End",
    "Genotype",
    "Stage",
    "Side",
]


# ============================================================
# FEATURE GROUPS
# ============================================================

excluded_columns = [
    "ID",
    "FolderName",
    "RootVolume",
    "Genotype",
    "Stage",
    "Side",
]


all_feature_cols = [c for c in features.columns if c not in excluded_columns]


# Spatial profile features
spatial_cols = [c for c in all_feature_cols if c.startswith("x_area_")]


# Track summary features
#
# IMPORTANT:
# Exclude metadata so we don't accidentally provide the
# same column twice to sklearn.
track_summary_cols = [
    c for c in all_feature_cols if (not c.startswith("x_area_") and c not in metadata)
]


print("\nFeature groups:")
print(f"  Metadata: {len(metadata)}")

print(f"  Track summary: {len(track_summary_cols)}")

print(f"  Spatial profile: {len(spatial_cols)}")

print(f"  All: {len(metadata) + len(track_summary_cols) + len(spatial_cols)}")


# ============================================================
# MODEL EVALUATION
# ============================================================


def evaluate(feature_columns, name):

    print()

    X = features[feature_columns].copy()

    y = features[TARGET].values

    groups = features[GROUP].values

    # Remove duplicate feature names defensively.
    X = X.loc[:, ~X.columns.duplicated()]

    feature_columns = list(X.columns)

    categorical = [c for c in META_CATEGORICAL if c in feature_columns]

    numeric = [c for c in feature_columns if c not in categorical]

    transformers = []

    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    model = Pipeline(
        [
            ("prep", preprocessor),
            ("ridge", Ridge(alpha=10.0)),
        ]
    )

    cv = GroupKFold(n_splits=5)

    scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups), 1):
        model.fit(X.iloc[train_idx], y[train_idx])

        pred = model.predict(X.iloc[val_idx])

        rmse = np.sqrt(mean_squared_error(y[val_idx], pred))

        scores.append(rmse)

        print(f"{name:32s} fold {fold}: {rmse:.4f}")

    mean_score = np.mean(scores)

    std_score = np.std(scores)

    print(f"{name:32s} mean: {mean_score:.4f} +/- {std_score:.4f}")

    return mean_score


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = {}


# ------------------------------------------------------------
# 1. Metadata
# ------------------------------------------------------------

results["Metadata"] = evaluate(metadata, "Metadata")


# ------------------------------------------------------------
# 2. Metadata + track summary
# ------------------------------------------------------------

results["Metadata + track summary"] = evaluate(
    metadata + track_summary_cols, "Metadata + track summary"
)


# ------------------------------------------------------------
# 3. Metadata + spatial profile
# ------------------------------------------------------------

results["Metadata + spatial profile"] = evaluate(
    metadata + spatial_cols, "Metadata + spatial profile"
)


# ------------------------------------------------------------
# 4. Metadata + everything
# ------------------------------------------------------------

results["Metadata + all"] = evaluate(
    metadata + track_summary_cols + spatial_cols, "Metadata + all"
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)

for name, score in results.items():
    print(f"{name:35s}: {score:.4f}")


print("\nPrevious best:")
print("Metadata + YOLO: 1.3994")

print("\nAnalysis complete.")
