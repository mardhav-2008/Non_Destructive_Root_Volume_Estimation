import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dataset import RootVolumeDataset, build_category_maps
from models.cnn_regressor import RootVolumeRegressor


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4

COVERAGE = 0.90

# Fixed epochs based on previous GroupKFold experiments.
FIXED_EPOCHS = {
    "image": 12,
    "metadata": 41,
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_ROOT = ROOT / "data" / "train"
LABELS_PATH = ROOT / "Train.csv"

OUTPUT_DIR = ROOT / "outputs" / "uncertainty"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC_COLUMNS = [
    "Start",
    "End",
    "strip_count",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================


def seed_everything(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATA PREPARATION
# ============================================================


def prepare_dataframe(frame):

    frame = frame.copy()

    frame["strip_count"] = frame["End"].astype(int) - frame["Start"].astype(int) + 1

    return frame


def split_by_folder(frame):

    # First split: 80% development, 20% evaluation.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    development_idx, evaluation_idx = next(
        splitter.split(
            frame,
            groups=frame["FolderName"],
        )
    )

    development = frame.iloc[development_idx].reset_index(drop=True)

    evaluation = frame.iloc[evaluation_idx].reset_index(drop=True)

    # Second split: 75% training, 25% calibration
    # within development data.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=SEED + 1,
    )

    training_idx, calibration_idx = next(
        splitter.split(
            development,
            groups=development["FolderName"],
        )
    )

    training = development.iloc[training_idx].reset_index(drop=True)

    calibration = development.iloc[calibration_idx].reset_index(drop=True)

    # Verify folder separation.
    train_groups = set(training["FolderName"])
    calibration_groups = set(calibration["FolderName"])
    evaluation_groups = set(evaluation["FolderName"])

    assert train_groups.isdisjoint(calibration_groups)
    assert train_groups.isdisjoint(evaluation_groups)
    assert calibration_groups.isdisjoint(evaluation_groups)

    return training, calibration, evaluation


def make_loader(
    frame,
    category_maps,
    augment=False,
    shuffle=False,
):

    dataset = RootVolumeDataset(
        dataframe=frame,
        image_root=IMAGE_ROOT,
        category_maps=category_maps,
        augment=augment,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================
# BATCH PREPARATION
# ============================================================


def prepare_batch(
    batch,
    numeric_mean,
    numeric_std,
):

    image = batch["image"].to(DEVICE)
    categorical = batch["categorical"].to(DEVICE)

    numeric = batch["numeric"].to(DEVICE)

    numeric = (numeric - numeric_mean.to(DEVICE)) / numeric_std.to(DEVICE)

    target = batch["target"].to(DEVICE)

    return image, categorical, numeric, target


# ============================================================
# MODEL TRAINING
# ============================================================


def train_model(
    mode,
    train_loader,
    category_sizes,
    numeric_mean,
    numeric_std,
):

    epochs = FIXED_EPOCHS[mode]

    model = RootVolumeRegressor(
        category_sizes=category_sizes,
        mode=mode,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.SmoothL1Loss(beta=0.5)

    print(f"\nTraining {mode.upper()} for {epochs} epochs")

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0
        total_samples = 0

        for batch in train_loader:
            image, categorical, numeric, target = prepare_batch(
                batch,
                numeric_mean,
                numeric_std,
            )

            optimizer.zero_grad(set_to_none=True)

            if mode == "image":
                prediction = model(image=image)

            else:
                prediction = model(
                    categorical=categorical,
                    numeric=numeric,
                )

            loss = criterion(
                prediction,
                target,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            total_loss += loss.item() * target.size(0)
            total_samples += target.size(0)

        average_loss = total_loss / total_samples

        print(f"Epoch {epoch:03d}/{epochs} | Loss: {average_loss:.4f}")

    return model


# ============================================================
# PREDICTION
# ============================================================


@torch.no_grad()
def predict(
    model,
    loader,
    mode,
    numeric_mean,
    numeric_std,
):

    model.eval()

    predictions = []
    targets = []
    ids = []
    folders = []

    for batch in loader:
        image, categorical, numeric, target = prepare_batch(
            batch,
            numeric_mean,
            numeric_std,
        )

        if mode == "image":
            output = model(image=image)

        else:
            output = model(
                categorical=categorical,
                numeric=numeric,
            )

        # Root volume cannot be negative.
        output = output.clamp_min(0)

        predictions.extend(output.cpu().numpy())

        targets.extend(target.cpu().numpy())

        ids.extend(batch["id"])
        folders.extend(batch["folder"])

    return pd.DataFrame(
        {
            "ID": ids,
            "FolderName": folders,
            "actual": targets,
            "prediction": predictions,
        }
    )


# ============================================================
# CONFORMAL CALIBRATION
# ============================================================


def conformal_quantile(scores, coverage):

    scores = np.asarray(scores)

    n = len(scores)

    if n == 0:
        raise ValueError("Calibration set is empty.")

    # Finite-sample conformal rank.
    rank = int(np.ceil((n + 1) * coverage))

    if rank > n:
        return float("inf"), rank

    threshold = np.sort(scores)[rank - 1]

    return float(threshold), rank


def calibrate(calibration_predictions):

    calibration_predictions = calibration_predictions.copy()

    calibration_predictions["absolute_error"] = (
        calibration_predictions["actual"] - calibration_predictions["prediction"]
    ).abs()

    # --------------------------------------------------------
    # Observation-level calibration
    # --------------------------------------------------------

    observation_scores = calibration_predictions["absolute_error"].to_numpy()

    observation_q, observation_rank = conformal_quantile(
        observation_scores,
        COVERAGE,
    )

    # --------------------------------------------------------
    # Folder-maximum calibration
    # --------------------------------------------------------

    folder_scores = (
        calibration_predictions.groupby("FolderName")["absolute_error"].max().to_numpy()
    )

    folder_q, folder_rank = conformal_quantile(
        folder_scores,
        COVERAGE,
    )

    return {
        "observation_q": observation_q,
        "observation_rank": observation_rank,
        "observation_n": len(observation_scores),
        "folder_q": folder_q,
        "folder_rank": folder_rank,
        "folder_n": len(folder_scores),
    }


# ============================================================
# EVALUATION
# ============================================================


def evaluate_point_predictions(predictions):

    actual = predictions["actual"].to_numpy()
    predicted = predictions["prediction"].to_numpy()

    return {
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
        "r2": float(r2_score(actual, predicted)),
    }


def evaluate_intervals(
    predictions,
    threshold,
    method,
):

    result = predictions.copy()

    result["lower"] = np.maximum(
        0,
        result["prediction"] - threshold,
    )

    result["upper"] = result["prediction"] + threshold

    result["covered"] = (result["actual"] >= result["lower"]) & (
        result["actual"] <= result["upper"]
    )

    result["interval_width"] = result["upper"] - result["lower"]

    metrics = {
        "method": method,
        "target_coverage": COVERAGE,
        "empirical_coverage": float(result["covered"].mean()),
        "mean_interval_width": float(result["interval_width"].mean()),
        "median_interval_width": float(result["interval_width"].median()),
        "threshold": float(threshold),
        "evaluation_observations": len(result),
        "evaluation_folders": (result["FolderName"].nunique()),
    }

    return result, metrics


def evaluate_subgroups(interval_predictions):

    frame = interval_predictions.copy()

    frame["target_band"] = pd.cut(
        frame["actual"],
        bins=[
            -np.inf,
            1.0,
            2.0,
            4.0,
            np.inf,
        ],
        labels=[
            "<=1",
            "(1,2]",
            "(2,4]",
            ">4",
        ],
    )

    results = []

    for band, group in frame.groupby(
        "target_band",
        observed=False,
    ):
        if len(group) == 0:
            continue

        results.append(
            {
                "group": f"Volume {band}",
                "n": len(group),
                "coverage": group["covered"].mean(),
                "mean_width": group["interval_width"].mean(),
            }
        )

    for plant, group in frame.groupby("PlantNumber"):
        results.append(
            {
                "group": f"Plant {plant}",
                "n": len(group),
                "coverage": group["covered"].mean(),
                "mean_width": group["interval_width"].mean(),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================


def main():

    seed_everything(SEED)

    print("Device:", DEVICE)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    frame = pd.read_csv(LABELS_PATH)

    frame = prepare_dataframe(frame)

    training, calibration, evaluation = split_by_folder(frame)

    print("\nDATA SPLIT")

    for name, subset in [
        ("Training", training),
        ("Calibration", calibration),
        ("Evaluation", evaluation),
    ]:
        print(
            f"{name:12s}: "
            f"{len(subset):3d} observations | "
            f"{subset['FolderName'].nunique():2d} folders"
        )

    # Save split membership.
    split_dir = OUTPUT_DIR / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    training.to_csv(
        split_dir / "training.csv",
        index=False,
    )

    calibration.to_csv(
        split_dir / "calibration.csv",
        index=False,
    )

    evaluation.to_csv(
        split_dir / "evaluation.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Training-only preprocessing
    # --------------------------------------------------------

    category_maps = build_category_maps(training)

    category_sizes = [
        len(category_maps[column])
        for column in [
            "PlantNumber",
            "Genotype",
            "Stage",
            "Side",
        ]
    ]

    numeric_mean = torch.tensor(
        training[NUMERIC_COLUMNS].mean().values,
        dtype=torch.float32,
    )

    numeric_std = torch.tensor(
        training[NUMERIC_COLUMNS].std().values,
        dtype=torch.float32,
    ).clamp_min(1e-6)

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    train_loader = make_loader(
        training,
        category_maps,
        augment=True,
        shuffle=True,
    )

    calibration_loader = make_loader(
        calibration,
        category_maps,
        augment=False,
    )

    evaluation_loader = make_loader(
        evaluation,
        category_maps,
        augment=False,
    )

    all_metrics = []

    # --------------------------------------------------------
    # Train and calibrate both models
    # --------------------------------------------------------

    for mode in ["image", "metadata"]:
        seed_everything(SEED)

        model = train_model(
            mode,
            train_loader,
            category_sizes,
            numeric_mean,
            numeric_std,
        )

        # Save checkpoint.
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "mode": mode,
                "category_sizes": category_sizes,
                "category_maps": category_maps,
                "numeric_mean": numeric_mean,
                "numeric_std": numeric_std,
                "epochs": FIXED_EPOCHS[mode],
            },
            OUTPUT_DIR / f"{mode}_fixed_epoch.pt",
        )

        # Calibration predictions.
        calibration_predictions = predict(
            model,
            calibration_loader,
            mode,
            numeric_mean,
            numeric_std,
        )

        # Independent evaluation predictions.
        evaluation_predictions = predict(
            model,
            evaluation_loader,
            mode,
            numeric_mean,
            numeric_std,
        )

        # Attach metadata for subgroup analysis.
        metadata = evaluation[
            [
                "ID",
                "PlantNumber",
                "Genotype",
                "Stage",
                "Side",
            ]
        ]

        evaluation_predictions = evaluation_predictions.merge(
            metadata,
            on="ID",
            how="left",
            validate="one_to_one",
        )

        calibration_predictions.to_csv(
            OUTPUT_DIR / f"{mode}_calibration_predictions.csv",
            index=False,
        )

        # ----------------------------------------------------
        # Estimate calibration thresholds
        # ----------------------------------------------------

        thresholds = calibrate(calibration_predictions)

        print(f"\n{'=' * 60}")
        print(f"{mode.upper()} MODEL")
        print("=" * 60)

        print("\nPoint prediction metrics:")

        point_results = evaluate_point_predictions(evaluation_predictions)

        print(
            json.dumps(
                point_results,
                indent=2,
            )
        )

        # ----------------------------------------------------
        # Evaluate both interval methods
        # ----------------------------------------------------

        methods = [
            (
                "observation_split_conformal",
                "observation_q",
            ),
            (
                "folder_max_conformal",
                "folder_q",
            ),
        ]

        for method, threshold_key in methods:
            threshold = thresholds[threshold_key]

            interval_predictions, interval_metrics = evaluate_intervals(
                evaluation_predictions,
                threshold,
                method,
            )

            interval_predictions.to_csv(
                OUTPUT_DIR / f"{mode}_{method}_evaluation.csv",
                index=False,
            )

            subgroup_results = evaluate_subgroups(interval_predictions)

            subgroup_results.to_csv(
                OUTPUT_DIR / f"{mode}_{method}_subgroups.csv",
                index=False,
            )

            print(f"\n{method}")

            print(
                json.dumps(
                    interval_metrics,
                    indent=2,
                )
            )

            print("\nSubgroup coverage:")
            print(subgroup_results.to_string(index=False))

            all_metrics.append(
                {
                    "model": mode,
                    **interval_metrics,
                    **point_results,
                }
            )

        # Save model-level metrics.
        with open(
            OUTPUT_DIR / f"{mode}_metrics.json",
            "w",
        ) as file:
            json.dump(
                {
                    "point_metrics": point_results,
                    "calibration_thresholds": thresholds,
                },
                file,
                indent=2,
            )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = pd.DataFrame(all_metrics)

    summary.to_csv(
        OUTPUT_DIR / "uncertainty_summary.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("UNCERTAINTY SUMMARY")
    print("=" * 70)

    print(summary.to_string(index=False))

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
