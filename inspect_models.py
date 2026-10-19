from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "Models"


for model_path in sorted(MODEL_DIR.glob("*.pt")):
    print("=" * 70)
    print(model_path.name)
    print("=" * 70)

    model = YOLO(str(model_path))

    print("Task:", model.task)
    print("Model:", type(model.model).__name__)

    print("Names:", model.names)

    print("Number of classes:", len(model.names))

    print()
