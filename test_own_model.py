import os
import argparse
from pathlib import Path

from pothole_pipeline import load_dataset, run_pipeline, save_outputs


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Test the locally trained YOLOv8s pothole model on the dataset folder."
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(base_dir / "dataset"),
        help="Dataset directory containing dataset.json, images/, and masks/.",
    )
    parser.add_argument(
        "--weights",
        default=str(base_dir / "runs" / "detect" / "train2" / "weights" / "best.pt"),
        help="Path to the local YOLOv8s weights to test.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold used when a segmentation mask is returned.",
    )
    parser.add_argument(
        "--pixel-scale-m-per-px",
        type=float,
        default=None,
        help="Override the dataset pixel scale in meters per pixel for size conversion.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of dataset items to test for a quick smoke run.",
    )
    parser.add_argument(
        "--output-json",
        default=str(base_dir / "test_own_model_results.json"),
        help="Where to save per-image structured results.",
    )
    parser.add_argument(
        "--output-summary",
        default=str(base_dir / "test_own_model_summary.txt"),
        help="Where to save a readable summary report.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    matplotlib_dir = base_dir / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    from ultralytics.yolo.utils import set_settings

    set_settings({"sync": False})

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Could not find YOLOv8s weights at {weights_path}. "
            "Expected the trained 8s model at runs/detect/train2/weights/best.pt."
        )

    _, items = load_dataset(args.dataset_dir)
    if args.limit is not None:
        items = items[: args.limit]

    results = run_pipeline(
        items=items,
        mode="yolo",
        weights=str(weights_path),
        conf=args.conf,
        imgsz=args.imgsz,
        mask_threshold=args.mask_threshold,
        pixel_scale_m_per_px=args.pixel_scale_m_per_px,
    )
    summary_text = save_outputs(results, args.output_json, args.output_summary)

    print(f"Tested weights: {weights_path.resolve()}")
    print(f"Dataset: {Path(args.dataset_dir).resolve()}")
    print(summary_text, end="")
    print(f"Saved JSON results to: {Path(args.output_json).resolve()}")
    print(f"Saved summary to: {Path(args.output_summary).resolve()}")


if __name__ == "__main__":
    main()
