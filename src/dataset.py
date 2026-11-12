from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from preprocessing import load_observation_image


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


def build_category_maps(frame: pd.DataFrame):
    """
    Build categorical vocabularies.

    Index 0 is reserved for unknown categories.
    """

    maps = {}

    for column in CATEGORICAL_COLUMNS:
        values = sorted(frame[column].dropna().astype(str).unique())

        maps[column] = {value: index + 1 for index, value in enumerate(values)}

    return maps


class RootVolumeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_root: Path,
        category_maps: dict,
        augment: bool = False,
    ):
        self.frame = dataframe.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.category_maps = category_maps
        self.augment = augment

    def __len__(self):
        return len(self.frame)

    def _augment_image(self, image: torch.Tensor):
        """
        Conservative intensity augmentation.

        The validity mask is not modified.
        """

        grayscale = image[0]
        mask = image[1]

        # Small contrast adjustment.
        contrast = torch.empty(1).uniform_(0.90, 1.10).item()
        grayscale = grayscale * contrast

        # Small brightness adjustment.
        brightness = torch.empty(1).uniform_(-0.05, 0.05).item()
        grayscale = grayscale + brightness

        # Small Gaussian noise, only inside valid pixels.
        noise = torch.randn_like(grayscale) * 0.01
        grayscale = grayscale + noise * mask

        grayscale = grayscale.clamp(0.0, 1.0)

        return torch.stack([grayscale, mask], dim=0)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image = load_observation_image(
            image_root=self.image_root,
            folder=str(row["FolderName"]),
            side=str(row["Side"]),
            start=int(row["Start"]),
            end=int(row["End"]),
        )

        if self.augment:
            image = self._augment_image(image)

        categorical = []

        for column in CATEGORICAL_COLUMNS:
            value = str(row[column])
            category_index = self.category_maps[column].get(value, 0)
            categorical.append(category_index)

        categorical = torch.tensor(
            categorical,
            dtype=torch.long,
        )

        numeric = torch.tensor(
            [
                float(row["Start"]),
                float(row["End"]),
                float(row["End"] - row["Start"] + 1),
            ],
            dtype=torch.float32,
        )

        target = torch.tensor(
            float(row["RootVolume"]),
            dtype=torch.float32,
        )

        return {
            "image": image,
            "categorical": categorical,
            "numeric": numeric,
            "target": target,
            "id": str(row["ID"]),
            "folder": str(row["FolderName"]),
        }
