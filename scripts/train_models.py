import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import RootVolumeDataset
from models.cnn_regressor import RootVolumeRegressor


# --------------------------------------------------
# Configuration
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SPLIT_DIR = ROOT / "outputs" / "splits"
IMAGE_ROOT = ROOT / "data" / "train"
OUTPUT_DIR = ROOT / "outputs" / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# --------------------------------------------------
# Load prepared data
# --------------------------------------------------

train_df = pd.read_csv(SPLIT_DIR / "train_split.csv")
val_df = pd.read_csv(SPLIT_DIR / "validation_split.csv")

with open(SPLIT_DIR / "category_maps.json") as file:
    category_maps = json.load(file)

category_sizes = [
    len(category_maps[column])
    for column in [
        "PlantNumber",
        "Genotype",
        "Stage",
        "Side",
    ]
]

# Normalize numeric features using training statistics only.
numeric_columns = ["Start", "End", "strip_count"]

numeric_mean = torch.tensor(
    train_df[numeric_columns].mean().values,
    dtype=torch.float32,
)

numeric_std = torch.tensor(
    train_df[numeric_columns].std().values,
    dtype=torch.float32,
).clamp_min(1e-6)


# --------------------------------------------------
# Dataset and loaders
# --------------------------------------------------

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


# --------------------------------------------------
# Evaluation
# --------------------------------------------------


def evaluate(model, loader, mode):
    model.eval()

    predictions = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)

            numeric = batch["numeric"].to(DEVICE)
            numeric = (numeric - numeric_mean.to(DEVICE)) / numeric_std.to(DEVICE)

            target = batch["target"].to(DEVICE)

            if mode == "image":
                prediction = model(image=image)

            elif mode == "metadata":
                prediction = model(
                    categorical=categorical,
                    numeric=numeric,
                )

            else:
                prediction = model(
                    image=image,
                    categorical=categorical,
                    numeric=numeric,
                )

            predictions.extend(prediction.cpu().numpy())
            targets.extend(target.cpu().numpy())

    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    return {
        "rmse": np.sqrt(mean_squared_error(targets, predictions)),
        "mae": mean_absolute_error(targets, predictions),
        "r2": r2_score(targets, predictions),
    }


# --------------------------------------------------
# Training
# --------------------------------------------------


def train_model(mode):
    print("\n" + "=" * 65)
    print(f"TRAINING: {mode.upper()}")
    print("=" * 65)

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

    criterion = nn.SmoothL1Loss(beta=0.5)

    best_rmse = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    checkpoint_path = OUTPUT_DIR / f"{mode}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0

        for batch in train_loader:
            image = batch["image"].to(DEVICE)
            categorical = batch["categorical"].to(DEVICE)

            numeric = batch["numeric"].to(DEVICE)
            numeric = (numeric - numeric_mean.to(DEVICE)) / numeric_std.to(DEVICE)

            target = batch["target"].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            if mode == "image":
                prediction = model(image=image)

            elif mode == "metadata":
                prediction = model(
                    categorical=categorical,
                    numeric=numeric,
                )

            else:
                prediction = model(
                    image=image,
                    categorical=categorical,
                    numeric=numeric,
                )

            loss = criterion(prediction, target)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            running_loss += loss.item() * target.size(0)

        train_loss = running_loss / len(train_dataset)

        metrics = evaluate(model, val_loader, mode)

        scheduler.step(metrics["rmse"])

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Loss: {train_loss:.4f} | "
            f"Val RMSE: {metrics['rmse']:.4f} | "
            f"MAE: {metrics['mae']:.4f} | "
            f"R2: {metrics['r2']:.4f}"
        )

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mode": mode,
                    "category_sizes": category_sizes,
                    "numeric_mean": numeric_mean,
                    "numeric_std": numeric_std,
                    "best_epoch": best_epoch,
                    "best_metrics": metrics,
                },
                checkpoint_path,
            )

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation RMSE: {best_rmse:.4f}")

    return {
        "model": mode,
        "best_epoch": best_epoch,
        **evaluate_checkpoint(checkpoint_path, mode),
    }


def evaluate_checkpoint(path, mode):
    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False,
    )

    model = RootVolumeRegressor(
        category_sizes=category_sizes,
        mode=mode,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])

    return evaluate(model, val_loader, mode)


# --------------------------------------------------
# Run experiments
# --------------------------------------------------

if __name__ == "__main__":
    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    results = []

    for mode in ["image", "metadata", "multimodal"]:
        results.append(train_model(mode))

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)

    print(
        results_df[["model", "best_epoch", "rmse", "mae", "r2"]].to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_DIR)
