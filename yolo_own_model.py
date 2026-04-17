from pathlib import Path

import torch
from ultralytics import YOLO


# The block below is completely mandatory for PyTorch on Windows
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_yaml = base_dir / "yolo_dataset" / "dataset.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"

    # 1. Load the model
    model = YOLO("yolov8s.pt")

    # 2. Train the model (must be indented under the if statement!)
    results = model.train(
        data=str(dataset_yaml),
        epochs=50,
        imgsz=640,
        device=device,
    )
