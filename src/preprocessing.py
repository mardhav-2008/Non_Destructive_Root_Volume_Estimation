from pathlib import Path

import numpy as np
import torch
from PIL import Image


IMAGE_WIDTH = 512
IMAGE_HEIGHT = 256


def load_observation_image(
    image_root: Path,
    folder: str,
    side: str,
    start: int,
    end: int,
) -> torch.Tensor:
    """
    Load an observation's image strips, stack vertically,
    resize while preserving aspect ratio, and pad vertically.

    Returns:
        Tensor of shape (2, 256, 512).

        Channel 0: grayscale image, range [0, 1].
        Channel 1: validity mask, 1 for image pixels and 0 for padding.
    """

    strips = []

    for index in range(start, end + 1):
        image_path = image_root / folder / f"{folder}_{side}_{index:03d}.png"

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            strips.append(np.asarray(grayscale, dtype=np.uint8))

    # Stack slices in ascending index order.
    stacked = np.vstack(strips)

    original_height, original_width = stacked.shape

    # Preserve aspect ratio while fixing width.
    resized_height = round(original_height * IMAGE_WIDTH / original_width)

    if resized_height > IMAGE_HEIGHT:
        raise ValueError(
            f"Resized height {resized_height} exceeds "
            f"target height {IMAGE_HEIGHT}. "
            f"Original shape: {stacked.shape}"
        )

    resized = Image.fromarray(stacked).resize(
        (IMAGE_WIDTH, resized_height),
        resample=Image.Resampling.BILINEAR,
    )

    resized_array = np.asarray(resized, dtype=np.float32) / 255.0

    # Pad only at the bottom.
    image_tensor = np.zeros(
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        dtype=np.float32,
    )

    mask_tensor = np.zeros_like(image_tensor)

    image_tensor[:resized_height] = resized_array
    mask_tensor[:resized_height] = 1.0

    result = np.stack(
        [image_tensor, mask_tensor],
        axis=0,
    )

    return torch.from_numpy(result)
