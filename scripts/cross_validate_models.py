import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import RootVolumeDataset, build_category_maps
from models.cnn_regressor import RootVolumeRegressor


# ==================================================
# CONFIGURATION
# ==================================================

SEED = 42
FOLDS = 5
EPOCHS = 80
PATIENCE = 12
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_ROOT = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs" / "cross_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CATEGORICAL_COLUMNS = [
    "PlantNumber",
    "Genotype",
    "Stage",
    "Side",
]

NUMERIC_COLUMNS = [
    "Start",
    "End",
    "strip_count",
]

EXPERIMENTS = [
    ("image_smoothl1", "image", "smoothl1"),
    ("metadata_smoothl1", "metadata", "smoothl1"),
    ("multimodal_smoothl1", "multimodal", "smoothl1"),
    ("multimodal_mse", "multimodal", "mse"),
    ("multimodal_log_mse", "multimodal", "log_mse"),
]


# ==================================================
# REPRODUCIBILITY
# ==================================================


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================================================
# METRICS
# ==================================================


def calculate_metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


# ==================================================
# LOSS
# ==================================================


def calculate_loss(prediction, target, loss_name):
    if loss_name == "smoothl1":
        return nn.functional.smooth_l1_loss(
            prediction,
            target,
            beta=0.5,
        )

    if loss_name in {"mse", "log_mse"}:
        return nn.functional.mse_loss(
            prediction,
            target,
        )

    raise ValueError(loss_name)


def transform_target(target, loss_name):
    if loss_name == "log_mse":
        return torch.log1p(target)

    return target


def inverse_prediction(prediction, loss_name):
    if loss_name == "log_mse":
        return torch.expm1(prediction).clamp_min(0)

    return prediction.clamp_min(0)


# ==================================================
# PREDICTION
# ==================================================


def predict(model, loader, mode, loss_name, numeric_mean, numeric_std):
    model.eval()

    actuals = []
    predictions = []
    identifiers = []
    folders = []
    plant_numbers = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)

            numeric = batch["numeric"].to(DEVICE)
            numeric = (numeric - numeric_mean) / numeric_std

            if mode == "image":
                output = model(image=image)

            elif mode == "metadata":
                output = model(
                    categorical=categorical,
                    numeric=numeric,
                )

            else:
                output = model(
                    image=image,
                    categorical=categorical,
                    numeric=numeric,
                )

            output = inverse_prediction(output, loss_name)

            predictions.extend(output.cpu().numpy())
            actuals.extend(batch["target"].numpy())
            identifiers.extend(batch["id"])
            folders.extend(batch["folder"])

            plant_numbers.extend(batch["categorical"][:, 0].numpy())

    return pd.DataFrame(
        {
            "ID": identifiers,
            "FolderName": folders,
            "PlantNumberEncoded": plant_numbers,
            "actual": actuals,
            "prediction": predictions,
        }
    )


# ==================================================
# ONE EXPERIMENT, ONE FOLD
# ==================================================


def run_fold(
    experiment_name,
    mode,
    loss_name,
    train_df,
    val_df,
    fold,
):
    seed_everything(SEED + fold)

    # Build category maps using training data only.
    category_maps = build_category_maps(train_df)

    category_sizes = [len(category_maps[column]) for column in CATEGORICAL_COLUMNS]

    # Fold-specific numeric normalization.
    numeric_mean = torch.tensor(
        train_df[NUMERIC_COLUMNS].mean().values,
        dtype=torch.float32,
        device=DEVICE,
    )

    numeric_std = torch.tensor(
        train_df[NUMERIC_COLUMNS].std().values,
        dtype=torch.float32,
        device=DEVICE,
    ).clamp_min(1e-6)

    train_dataset = RootVolumeDataset(
        train_df,
        IMAGE_ROOT,
        category_maps,
        augment=True,
    )

    val_dataset = RootVolumeDataset(
        val_df,
        IMAGE_ROOT,
        category_maps,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )

    model = RootVolumeRegressor(
        category_sizes=category_sizes,
        mode=mode,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_rmse = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            image = batch["image"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)

            numeric = batch["numeric"].to(DEVICE)
            numeric = (numeric - numeric_mean) / numeric_std

            target = batch["target"].to(DEVICE)
            target = transform_target(target, loss_name)

            optimizer.zero_grad(set_to_none=True)

            if mode == "image":
                output = model(image=image)

            elif mode == "metadata":
                output = model(
                    categorical=categorical,
                    numeric=numeric,
                )

            else:
                output = model(
                    image=image,
                    categorical=categorical,
                    numeric=numeric,
                )

            loss = calculate_loss(
                output,
                target,
                loss_name,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item() * target.size(0)

        # Validate in original RootVolume units.
        validation = predict(
            model,
            val_loader,
            mode,
            loss_name,
            numeric_mean,
            numeric_std,
        )

        metrics = calculate_metrics(
            validation["actual"],
            validation["prediction"],
        )

        scheduler.step(metrics["rmse"])

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint produced.")

    model.load_state_dict(best_state)

    validation = predict(
        model,
        val_loader,
        mode,
        loss_name,
        numeric_mean,
        numeric_std,
    )

    validation["fold"] = fold
    validation["experiment"] = experiment_name

    metrics = calculate_metrics(
        validation["actual"],
        validation["prediction"],
    )

    metrics.update(
        {
            "experiment": experiment_name,
            "fold": fold,
            "best_epoch": best_epoch,
            "train_observations": len(train_df),
            "validation_observations": len(val_df),
            "train_folders": train_df["FolderName"].nunique(),
            "validation_folders": val_df["FolderName"].nunique(),
        }
    )

    # Fold-specific mean baseline.
    mean_prediction = train_df["RootVolume"].mean()

    baseline = calculate_metrics(
        validation["actual"],
        np.full(
            len(validation),
            mean_prediction,
        ),
    )

    metrics["mean_baseline_rmse"] = baseline["rmse"]

    return metrics, validation


# ==================================================
# MAIN
# ==================================================


def main():
    seed_everything(SEED)

    df = pd.read_csv(ROOT / "Train.csv")

    df["strip_count"] = df["End"].astype(int) - df["Start"].astype(int) + 1

    if df.isnull().any().any():
        raise ValueError("Missing values in training dataframe.")

    splitter = GroupKFold(n_splits=FOLDS)

    splits = list(
        splitter.split(
            df,
            groups=df["FolderName"],
        )
    )

    print("=" * 70)
    print("GROUPED CROSS-VALIDATION")
    print("=" * 70)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("Total observations:", len(df))
    print("Total folders:", df["FolderName"].nunique())
    print("Folds:", FOLDS)

    all_metrics = []
    all_predictions = []

    for experiment_name, mode, loss_name in EXPERIMENTS:
        print("\n" + "=" * 70)
        print("EXPERIMENT:", experiment_name)
        print("=" * 70)

        for fold, (train_indices, val_indices) in enumerate(
            splits,
            start=1,
        ):
            train_df = df.iloc[train_indices].copy()
            val_df = df.iloc[val_indices].copy()

            assert set(train_df["FolderName"]).isdisjoint(set(val_df["FolderName"]))

            print(
                f"\nFold {fold}/{FOLDS} | "
                f"Train: {len(train_df)} | "
                f"Validation: {len(val_df)}"
            )

            metrics, predictions = run_fold(
                experiment_name,
                mode,
                loss_name,
                train_df,
                val_df,
                fold,
            )

            all_metrics.append(metrics)
            all_predictions.append(predictions)

            print(
                f"RMSE: {metrics['rmse']:.4f} | "
                f"MAE: {metrics['mae']:.4f} | "
                f"R2: {metrics['r2']:.4f} | "
                f"Best epoch: {metrics['best_epoch']}"
            )

    metrics_df = pd.DataFrame(all_metrics)
    predictions_df = pd.concat(
        all_predictions,
        ignore_index=True,
    )

    metrics_df.to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_DIR / "oof_predictions.csv",
        index=False,
    )

    # Fold-level summary.
    summary = (
        metrics_df.groupby("experiment")
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            baseline_rmse_mean=("mean_baseline_rmse", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False,
    )

    # Pooled out-of-fold metrics.
    pooled_rows = []

    for experiment, group in predictions_df.groupby("experiment"):
        pooled = calculate_metrics(
            group["actual"],
            group["prediction"],
        )

        pooled["experiment"] = experiment
        pooled_rows.append(pooled)

    pooled_df = pd.DataFrame(pooled_rows)

    pooled_df.to_csv(
        OUTPUT_DIR / "pooled_oof_metrics.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("FOLD-AVERAGED RESULTS")
    print("=" * 70)

    print(summary.to_string(index=False))

    print("\n" + "=" * 70)
    print("POOLED OUT-OF-FOLD RESULTS")
    print("=" * 70)

    print(pooled_df.to_string(index=False))

    print("\nSaved results to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
