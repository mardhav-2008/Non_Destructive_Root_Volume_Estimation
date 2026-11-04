from pathlib import Path
from typing import cast

from ultralytics import YOLO
from ultralytics.engine.results import Results


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "Models" / "best_early.pt"

IMAGE_PATH = PROJECT_ROOT / "data" / "train" / "A2miww5mfx" / "A2miww5mfx_L_010.png"


# ============================================================
# Load model
# ============================================================

print("=" * 60)
print("LOADING MODEL")
print("=" * 60)

model = YOLO(str(MODEL_PATH))

print("Model:", MODEL_PATH.name)
print("Task:", model.task)
print("Names:", model.names)


# ============================================================
# Model configuration
# ============================================================

print()
print("=" * 60)
print("MODEL INFORMATION")
print("=" * 60)

print("Model class:", type(model.model).__name__)

print()
print("Model arguments:")

try:
    for key, value in model.model.args.items():
        print(f"  {key}: {value}")
except AttributeError:
    print("Could not read model arguments.")


# ============================================================
# Image information
# ============================================================

print()
print("=" * 60)
print("IMAGE INFORMATION")
print("=" * 60)

print("Image:", IMAGE_PATH.name)
print("Exists:", IMAGE_PATH.exists())


# ============================================================
# Run predictions at several confidence thresholds
# ============================================================

for conf in [0.25, 0.10, 0.05, 0.01]:
    print()
    print("=" * 60)
    print(f"CONFIDENCE = {conf}")
    print("=" * 60)

    results = model.predict(
        source=str(IMAGE_PATH),
        conf=conf,
        verbose=False,
    )

    # Tell Pylance that this is definitely an Ultralytics Results
    # object rather than a Tensor.
    result = cast(Results, results[0])

    # --------------------------------------------------------
    # Basic information
    # --------------------------------------------------------

    print("Original image shape:", result.orig_shape)

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    if result.masks is None:
        print("Masks: NONE")

    else:
        num_masks = len(result.masks.xy)

        print("Number of masks:", num_masks)

        for i, polygon in enumerate(result.masks.xy):
            print(f"  Mask {i}: {len(polygon)} polygon points")

    # --------------------------------------------------------
    # Bounding boxes
    # --------------------------------------------------------

    if result.boxes is None:
        print("Boxes: NONE")

    else:
        num_boxes = len(result.boxes)

        print("Number of boxes:", num_boxes)

        if num_boxes > 0:
            confidences = result.boxes.conf.cpu().numpy()

            classes = result.boxes.cls.cpu().numpy()

            print("Confidences:", confidences)
            print("Classes:", classes)

            for i, (confidence, cls) in enumerate(zip(confidences, classes)):
                print(f"  Detection {i}: class={int(cls)}, confidence={confidence:.6f}")
