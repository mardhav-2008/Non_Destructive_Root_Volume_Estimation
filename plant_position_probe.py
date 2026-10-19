from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "train"

df = pd.read_csv(PROJECT_ROOT / "Train.csv")


def load_observation(row):

    folder = row["FolderName"]
    side = row["Side"]
    start = int(row["Start"])
    end = int(row["End"])

    images = []

    for s in range(start, end + 1):
        path = TRAIN_DIR / folder / f"{folder}_{side}_{s:03d}.png"

        if not path.exists():
            continue

        image = np.array(Image.open(path))

        if image.ndim == 3:
            image = image[:, :, :3].mean(axis=2)

        image = image.astype(np.float32)

        images.append(image)

    if not images:
        return None

    return np.vstack(images)


def extract_bin_features(scan, n_bins):

    width = scan.shape[1]

    features = []

    # Normalize each row first so brightness differences
    # between slices don't dominate.
    scan = scan - scan.mean(axis=1, keepdims=True)

    for i in range(n_bins):
        x0 = int(i * width / n_bins)
        x1 = int((i + 1) * width / n_bins)

        region = scan[:, x0:x1]

        abs_region = np.abs(region)

        features.extend(
            [
                region.mean(),
                region.std(),
                abs_region.mean(),
                np.percentile(abs_region, 75),
                np.percentile(abs_region, 90),
                np.percentile(abs_region, 95),
                abs_region.max(),
            ]
        )

    return features


# ------------------------------------------------------------
# Build several representations
# ------------------------------------------------------------

representations = {
    2: [],
    3: [],
    4: [],
    5: [],
    7: [],
}

targets = []
folders = []


print("Extracting features...")


for idx, row in df.iterrows():
    scan = load_observation(row)

    if scan is None:
        continue

    targets.append(row["RootVolume"])
    folders.append(row["FolderName"])

    for n_bins in representations:
        representations[n_bins].append(
            extract_bin_features(
                scan,
                n_bins,
            )
        )


targets = np.array(targets)
folders = np.array(folders)


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

gkf = GroupKFold(n_splits=5)


for n_bins, X in representations.items():
    X = np.array(X)

    maes = []
    mses = []
    rmses = []
    r2s = []

    print()
    print("=" * 60)
    print(f"{n_bins} HORIZONTAL BINS")
    print("=" * 60)

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X, targets, folders),
        start=1,
    ):
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=10.0),
        )

        model.fit(
            X[train_idx],
            targets[train_idx],
        )

        pred = model.predict(X[val_idx])

        mae = mean_absolute_error(
            targets[val_idx],
            pred,
        )

        mse = mean_squared_error(
            targets[val_idx],
            pred,
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            targets[val_idx],
            pred,
        )

        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        r2s.append(r2)

        print(f"Fold {fold}: MAE={mae:.4f} MSE={mse:.4f} RMSE={rmse:.4f} R2={r2:.4f}")

    print()
    print(f"MAE  : {np.mean(maes):.4f} +/- {np.std(maes):.4f}")
    print(f"MSE  : {np.mean(mses):.4f} +/- {np.std(mses):.4f}")
    print(f"RMSE : {np.mean(rmses):.4f} +/- {np.std(rmses):.4f}")
    print(f"R2   : {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
