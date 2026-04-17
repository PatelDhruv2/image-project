import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run pothole detection across the keremberke dataset and measure pothole size. "
            "Default mode uses dataset masks/metadata. YOLO mode can be enabled later with local weights."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(Path(__file__).resolve().parent / "dataset"),
        help="Dataset directory containing dataset.json, images/, and masks/.",
    )
    parser.add_argument(
        "--mode",
        choices=("annotations", "yolo", "online"),
        default="annotations",
        help="Use annotation masks, a local YOLO model, or the online Keremberke model.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to local YOLO weights. Required only for --mode yolo.",
    )
    parser.add_argument(
        "--online-model",
        default="keremberke/yolov8n-pothole-segmentation",
        help="Model id for ultralyticsplus when --mode online is used.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for YOLO predictions.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for YOLO predictions.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Threshold used to binarize predicted masks in YOLO mode.",
    )
    parser.add_argument(
        "--pixel-scale-m-per-px",
        type=float,
        default=None,
        help="Override the dataset pixel scale in meters per pixel for size conversion.",
    )
    parser.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parent / "pipeline_results.json"),
        help="Where to save per-image structured results.",
    )
    parser.add_argument(
        "--output-summary",
        default=str(Path(__file__).resolve().parent / "pipeline_summary.txt"),
        help="Where to save a readable summary report.",
    )
    return parser.parse_args()


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir)
    dataset_json = dataset_dir / "dataset.json"
    if not dataset_json.exists():
        raise FileNotFoundError(f"Missing dataset metadata file: {dataset_json}")
    data = json.loads(dataset_json.read_text(encoding="utf-8"))
    for item in data:
        item["image_path"] = str((dataset_dir / item["image_file"]).resolve())
        item["mask_path"] = str((dataset_dir / item["mask_file"]).resolve())
    return dataset_dir, data


def load_mask(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return (mask > 0).astype(np.uint8)


def resolve_scale_m_per_px(item, override_scale_m_per_px=None):
    if override_scale_m_per_px is not None:
        return override_scale_m_per_px
    return item.get("pixel_scale_m_per_px")


def measure_binary_mask(binary_mask, scale_m_per_px=None):
    if binary_mask is None or int(binary_mask.sum()) == 0:
        return None

    ys, xs = np.where(binary_mask > 0)
    x_min = int(xs.min())
    x_max = int(xs.max())
    y_min = int(ys.min())
    y_max = int(ys.max())
    width_px = int(x_max - x_min + 1)
    height_px = int(y_max - y_min + 1)
    area_px = int(binary_mask.sum())

    contours, _ = cv2.findContours(binary_mask * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(largest)
    length_px = float(max(rect_w, rect_h))
    short_px = float(min(rect_w, rect_h))

    measurement = {
        "mask_area_px": area_px,
        "mask_bounds_px": {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
        },
        "axis_aligned_width_px": width_px,
        "axis_aligned_height_px": height_px,
        "rotated_length_px": round(length_px, 4),
        "rotated_width_px": round(short_px, 4),
        "rotated_angle_deg": round(float(angle), 4),
    }

    if scale_m_per_px is not None:
        measurement["pixel_scale_m_per_px"] = scale_m_per_px
        measurement["mask_area_m2"] = round(area_px * (scale_m_per_px ** 2), 6)
        measurement["axis_aligned_width_m"] = round(width_px * scale_m_per_px, 4)
        measurement["axis_aligned_height_m"] = round(height_px * scale_m_per_px, 4)
        measurement["rotated_length_m"] = round(length_px * scale_m_per_px, 4)
        measurement["rotated_width_m"] = round(short_px * scale_m_per_px, 4)

    return measurement


def result_from_annotations(item, override_scale_m_per_px=None):
    binary_mask = load_mask(item["mask_path"])
    has_pothole = bool(binary_mask.sum() > 0)
    scale_m_per_px = resolve_scale_m_per_px(item, override_scale_m_per_px)
    measurement = measure_binary_mask(binary_mask, scale_m_per_px) if has_pothole else None
    return {
        "image_id": item["image_id"],
        "image_path": item["image_path"],
        "state": item.get("state"),
        "mode": "annotations",
        "detected_pothole": has_pothole,
        "confidence": 1.0 if has_pothole else 0.0,
        "ground_truth_has_pothole": item.get("has_pothole"),
        "measurement": measurement,
    }


def make_rect_mask(image_shape, xyxy):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    x1, y1, x2, y2 = xyxy
    x1 = max(0, int(np.floor(x1)))
    y1 = max(0, int(np.floor(y1)))
    x2 = min(mask.shape[1], int(np.ceil(x2)))
    y2 = min(mask.shape[0], int(np.ceil(y2)))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def find_cached_hf_weights(model_id, filename="best.pt"):
    model_cache_dir = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}"
    ref_file = model_cache_dir / "refs" / "main"
    if not ref_file.exists():
        return None
    revision = ref_file.read_text(encoding="utf-8").strip()
    candidate = model_cache_dir / "snapshots" / revision / filename
    return candidate if candidate.exists() else None


def result_from_yolo(item, model, conf, imgsz, mask_threshold, override_scale_m_per_px=None):
    image_path = item["image_path"]
    result = model.predict(
        source=image_path,
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )[0]

    confidences = result.boxes.conf.tolist() if result.boxes is not None and result.boxes.conf is not None else []
    best_index = int(np.argmax(confidences)) if confidences else None
    has_pothole = best_index is not None
    measurement = None
    detection = None

    if has_pothole:
        image = np.array(Image.open(image_path))
        if result.masks is not None and result.masks.data is not None:
            predicted_mask = result.masks.data[best_index].cpu().numpy()
            binary_mask = (predicted_mask > mask_threshold).astype(np.uint8)
            measurement_source = "segmentation_mask"
        else:
            xyxy = result.boxes.xyxy[best_index].cpu().numpy().tolist()
            binary_mask = make_rect_mask(image.shape, xyxy)
            measurement_source = "bounding_box"

        scale_m_per_px = resolve_scale_m_per_px(item, override_scale_m_per_px)
        measurement = measure_binary_mask(binary_mask, scale_m_per_px)
        detection = {
            "confidence": round(float(confidences[best_index]), 6),
            "class_id": int(result.boxes.cls[best_index].item()) if result.boxes.cls is not None else None,
            "measurement_source": measurement_source,
        }
        if result.boxes.xyxy is not None:
            xyxy = result.boxes.xyxy[best_index].cpu().numpy().tolist()
            detection["bbox_xyxy"] = [round(float(v), 3) for v in xyxy]

    return {
        "image_id": item["image_id"],
        "image_path": image_path,
        "state": item.get("state"),
        "mode": "yolo",
        "detected_pothole": has_pothole,
        "confidence": round(float(confidences[best_index]), 6) if has_pothole else 0.0,
        "ground_truth_has_pothole": item.get("has_pothole"),
        "detection": detection,
        "measurement": measurement,
    }


def run_pipeline(
    items,
    mode,
    weights=None,
    online_model="keremberke/yolov8n-pothole-segmentation",
    conf=0.25,
    imgsz=640,
    mask_threshold=0.5,
    pixel_scale_m_per_px=None,
):
    if mode == "annotations":
        return [
            result_from_annotations(item, override_scale_m_per_px=pixel_scale_m_per_px)
            for item in items
        ]

    if mode == "online":
        cached_weights = find_cached_hf_weights(online_model)
        if cached_weights is not None:
            from ultralytics import YOLO

            model = YOLO(str(cached_weights))
        else:
            from ultralyticsplus import YOLO

            model = YOLO(online_model)

        model.overrides["conf"] = conf
        model.overrides["iou"] = 0.45
        model.overrides["agnostic_nms"] = False
        model.overrides["max_det"] = 1000
        return [
            result_from_yolo(
                item=item,
                model=model,
                conf=conf,
                imgsz=imgsz,
                mask_threshold=mask_threshold,
                override_scale_m_per_px=pixel_scale_m_per_px,
            )
            for item in items
        ]

    if mode != "yolo":
        raise ValueError(f"Unsupported mode: {mode}")

    if not weights:
        raise ValueError("--weights is required for --mode yolo")

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

    from ultralytics import YOLO

    model = YOLO(str(weights_path))
    return [
        result_from_yolo(
            item=item,
            model=model,
            conf=conf,
            imgsz=imgsz,
            mask_threshold=mask_threshold,
            override_scale_m_per_px=pixel_scale_m_per_px,
        )
        for item in items
    ]


def build_summary(results):
    total_images = len(results)
    positives = [result for result in results if result["detected_pothole"]]
    gt_positives = [result for result in results if result.get("ground_truth_has_pothole")]

    lines = [
        "Pothole Detection Pipeline Summary",
        f"Total images scanned: {total_images}",
        f"Detected pothole images: {len(positives)}",
        f"Ground-truth pothole images in metadata: {len(gt_positives)}",
    ]

    if positives:
        lines.append("")
        lines.append("Detected pothole entries:")
        for result in positives:
            measurement = result.get("measurement") or {}
            size_m = "size_m=not available"
            if "rotated_length_m" in measurement and "rotated_width_m" in measurement:
                size_m = (
                    f"size_m={measurement['rotated_length_m']:.2f} x "
                    f"{measurement['rotated_width_m']:.2f}"
                )
            area_m2 = measurement.get("mask_area_m2")
            area_text = f" area_m2={area_m2:.4f}" if area_m2 is not None else ""
            lines.append(
                f"- image_id={result['image_id']} confidence={result['confidence']:.4f} "
                f"{size_m}{area_text}"
            )
    else:
        lines.append("")
        lines.append("No potholes were detected.")

    if any(result.get("ground_truth_has_pothole") is not None for result in results):
        tp = sum(result["detected_pothole"] and result.get("ground_truth_has_pothole") for result in results)
        fp = sum(result["detected_pothole"] and not result.get("ground_truth_has_pothole") for result in results)
        fn = sum((not result["detected_pothole"]) and result.get("ground_truth_has_pothole") for result in results)
        tn = sum((not result["detected_pothole"]) and (not result.get("ground_truth_has_pothole")) for result in results)
        lines.extend(
            [
                "",
                "Comparison against dataset labels:",
                f"- TP={tp}",
                f"- FP={fp}",
                f"- FN={fn}",
                f"- TN={tn}",
            ]
        )

    return "\n".join(lines) + "\n"


def save_outputs(results, output_json, output_summary):
    output_json = Path(output_json)
    output_summary = Path(output_summary)
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary_text = build_summary(results)
    output_summary.write_text(summary_text, encoding="utf-8")
    return summary_text


def main():
    args = parse_args()
    _, items = load_dataset(args.dataset_dir)
    results = run_pipeline(
        items=items,
        mode=args.mode,
        weights=args.weights,
        online_model=args.online_model,
        conf=args.conf,
        imgsz=args.imgsz,
        mask_threshold=args.mask_threshold,
        pixel_scale_m_per_px=args.pixel_scale_m_per_px,
    )
    summary_text = save_outputs(results, args.output_json, args.output_summary)
    print(summary_text, end="")
    print(f"Saved JSON results to: {Path(args.output_json).resolve()}")
    print(f"Saved summary to: {Path(args.output_summary).resolve()}")


if __name__ == "__main__":
    main()
