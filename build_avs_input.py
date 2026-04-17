#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO
from ultralytics.yolo.utils import set_settings


TARGET_NAME = "pothole"


def parse_args():
    base_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Build an AVS-compatible input folder from the keremberke pothole dataset "
            "using the trained YOLOv8s weights."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(base_dir / "dataset"),
        help="Dataset directory containing dataset.json, images/, masks/, and transitions.json.",
    )
    parser.add_argument(
        "--weights",
        default=str(base_dir / "runs" / "detect" / "train2" / "weights" / "best.pt"),
        help="Path to the trained YOLOv8s weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(base_dir / "input_pomp_be_pd_yolov8s"),
        help="Output folder to create in AVS format.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detector outputs.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for YOLO predictions.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=90.0,
        help="Horizontal FOV used for the geometric observability approximation.",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Max visible cell distance. Defaults to the inferred grid step.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian sigma for GaussObsvMatrix. Defaults to the inferred grid step.",
    )
    parser.add_argument(
        "--listview-mode",
        choices=("absolute", "relative", "name"),
        default="absolute",
        help="How image paths are stored in ListView.txt.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    dataset_path = dataset_dir / "dataset.json"
    transitions_path = dataset_dir / "transitions.json"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {dataset_path}")
    if not transitions_path.exists():
        raise FileNotFoundError(f"Missing transitions metadata: {transitions_path}")

    items = load_json(dataset_path)
    transitions = load_json(transitions_path)

    normalized = []
    for item in items:
        entry = dict(item)
        entry["image_path"] = (dataset_dir / item["image_file"]).resolve()
        entry["mask_path"] = (dataset_dir / item["mask_file"]).resolve()
        normalized.append(entry)

    return dataset_dir, normalized, transitions


def listview_entry(item, output_dir, mode):
    image_path = Path(item["image_path"]).resolve()
    if mode == "absolute":
        return str(image_path)
    if mode == "relative":
        return os.path.relpath(image_path, output_dir)
    return image_path.name


def sorted_unique_cells(items):
    coords = {
        (float(item["state"]["x"]), float(item["state"]["y"]))
        for item in items
    }
    return sorted(coords)


def infer_grid_step(cells):
    if len(cells) < 2:
        return 1.0

    min_dist = None
    for i, (x1, y1) in enumerate(cells):
        for x2, y2 in cells[i + 1 :]:
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist <= 1e-9:
                continue
            if min_dist is None or dist < min_dist:
                min_dist = dist

    return min_dist if min_dist is not None else 1.0


def write_listview(items, output_dir, mode):
    lines = [listview_entry(item, output_dir, mode) for item in items]
    (output_dir / "ListView.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def write_photo_map(items, listview_lines, output_dir):
    lines = ["pose_index,image_id,listview_entry,image_path,x,y,yaw_deg,has_pothole"]
    for pose_index, (item, listview_line) in enumerate(zip(items, listview_lines)):
        state = item["state"]
        lines.append(
            ",".join(
                [
                    str(pose_index),
                    str(item["image_id"]),
                    listview_line,
                    str(item["image_path"]),
                    str(state["x"]),
                    str(state["y"]),
                    str(state["yaw_deg"]),
                    str(bool(item.get("has_pothole", False))).lower(),
                ]
            )
        )
    (output_dir / "photo_map.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_instance_map(output_dir):
    (output_dir / "instance_id_map.txt").write_text("0 pothole\n", encoding="utf-8")


def write_matrix_g(items, transitions, output_dir):
    image_id_to_index = {str(item["image_id"]): idx for idx, item in enumerate(items)}
    rows = []

    for item in items:
        image_id = str(item["image_id"])
        transition = transitions.get(image_id, {})

        def lookup(key):
            next_image_id = transition.get(key)
            if next_image_id is None:
                return -1
            return image_id_to_index.get(str(next_image_id), -1)

        rows.append([lookup("ccw"), lookup("cw"), lookup("forward"), lookup("backward")])

    np.savetxt(output_dir / "MatrixG.txt", np.array(rows, dtype=int), fmt="%d")


def mask_has_foreground(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return int(np.any(mask > 0))


def write_matrix_l_gt(items, output_dir):
    values = [[mask_has_foreground(item["mask_path"])] for item in items]
    matrix = np.array(values, dtype=int)
    np.savetxt(output_dir / "MatrixL_gt.txt", matrix, fmt="%d")
    return matrix


def load_model(weights_path):
    weights_path = Path(weights_path).resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing YOLO weights: {weights_path}")
    matplotlib_dir = Path("/tmp") / "matplotlib"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    set_settings({"sync": False})
    return YOLO(str(weights_path))


def detect_pothole(model, image_path, conf, imgsz):
    result = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    if result.boxes is None or result.boxes.cls is None or len(result.boxes.cls) == 0:
        return 0

    names = model.names
    for cls in result.boxes.cls.tolist():
        cls_name = str(names[int(cls)]).lower()
        if cls_name == TARGET_NAME:
            return 1
    return 0


def write_matrix_l_dt(items, output_dir, model, conf, imgsz):
    images = [np.array(Image.open(item["image_path"]).convert("RGB")) for item in items]
    results = model.predict(source=images, conf=conf, imgsz=imgsz, verbose=False)
    values = []

    for idx, result in enumerate(results, start=1):
        detected = 0
        if result.boxes is not None and result.boxes.cls is not None and len(result.boxes.cls) > 0:
            for cls in result.boxes.cls.tolist():
                cls_name = str(model.names[int(cls)]).lower()
                if cls_name == TARGET_NAME:
                    detected = 1
                    break
        values.append([detected])
        if idx % 20 == 0 or idx == len(results):
            print(f"  detector progress: {idx}/{len(results)}")

    matrix = np.array(values, dtype=int)
    np.savetxt(output_dir / "MatrixL_dt.txt", matrix, fmt="%d")
    return matrix


def detector_stats(matrix_dt, matrix_gt):
    dt = matrix_dt.flatten()
    gt = matrix_gt.flatten()

    tp = int(((dt == 1) & (gt == 1)).sum())
    fp = int(((dt == 1) & (gt == 0)).sum())
    fn = int(((dt == 0) & (gt == 1)).sum())
    tn = int(((dt == 0) & (gt == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "0": {
            "name": TARGET_NAME,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "false_positive_rate": round(false_positive_rate, 6),
            "false_negative_rate": round(false_negative_rate, 6),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }
    }


def write_detector_stats(matrix_dt, matrix_gt, output_dir):
    stats = detector_stats(matrix_dt, matrix_gt)
    with open(output_dir / "detector_stat.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def write_particles_inside_home(cells, output_dir):
    values = np.ones((len(cells), 1), dtype=int)
    np.savetxt(output_dir / "particles_that_are_inside_home.txt", values, fmt="%d")


def heading_vector(yaw_deg):
    theta = math.radians(float(yaw_deg))
    return math.sin(theta), math.cos(theta)


def write_observ_matrix(items, cells, output_dir, fov_deg, max_range):
    observ = np.zeros((len(items), len(cells)), dtype=int)
    half_fov = math.radians(fov_deg) / 2.0

    for row_idx, item in enumerate(items):
        px = float(item["state"]["x"])
        py = float(item["state"]["y"])
        fx, fy = heading_vector(item["state"]["yaw_deg"])

        for col_idx, (cx, cy) in enumerate(cells):
            dx = cx - px
            dy = cy - py
            dist = math.hypot(dx, dy)
            if dist < 1e-9 or dist > max_range:
                continue

            dot = (dx * fx + dy * fy) / dist
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            if angle <= half_fov:
                observ[row_idx, col_idx] = 1

    with open(output_dir / "ObservMatrix.txt", "w", encoding="utf-8") as handle:
        for row in observ:
            handle.write("".join(str(int(value)) for value in row) + "\n")


def write_gauss_matrix(items, cells, output_dir, sigma):
    gauss = np.zeros((len(items), len(cells)), dtype=float)
    eps = 1e-12

    for row_idx, item in enumerate(items):
        px = float(item["state"]["x"])
        py = float(item["state"]["y"])
        for col_idx, (cx, cy) in enumerate(cells):
            dist = math.hypot(px - cx, py - cy)
            gauss[row_idx, col_idx] = math.exp(-(dist * dist) / (2.0 * sigma * sigma))

        row_sum = gauss[row_idx].sum()
        if row_sum > eps:
            gauss[row_idx] /= row_sum

    np.savetxt(output_dir / "GaussObsvMatrix.txt", gauss, fmt="%.6f")


def main():
    args = parse_args()
    dataset_dir, items, transitions = load_dataset(args.dataset_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = sorted_unique_cells(items)
    inferred_step = infer_grid_step(cells)
    max_range = args.max_range if args.max_range is not None else inferred_step
    sigma = args.sigma if args.sigma is not None else inferred_step

    print(f"Dataset dir: {dataset_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Images/poses: {len(items)}")
    print(f"Unique cells: {len(cells)}")
    print(f"Inferred grid step: {inferred_step:.4f}")
    print(f"Using max_range={max_range:.4f}, sigma={sigma:.4f}")

    listview_lines = write_listview(items, output_dir, args.listview_mode)
    write_photo_map(items, listview_lines, output_dir)
    write_instance_map(output_dir)
    write_matrix_g(items, transitions, output_dir)
    matrix_gt = write_matrix_l_gt(items, output_dir)

    model = load_model(args.weights)
    print(f"Using detector weights: {Path(args.weights).resolve()}")
    matrix_dt = write_matrix_l_dt(items, output_dir, model, args.conf, args.imgsz)

    write_detector_stats(matrix_dt, matrix_gt, output_dir)
    write_particles_inside_home(cells, output_dir)
    write_observ_matrix(items, cells, output_dir, args.fov_deg, max_range)
    write_gauss_matrix(items, cells, output_dir, sigma)

    print("Wrote files:")
    print(f"  {output_dir / 'ListView.txt'}")
    print(f"  {output_dir / 'MatrixG.txt'}")
    print(f"  {output_dir / 'MatrixL_dt.txt'}")
    print(f"  {output_dir / 'MatrixL_gt.txt'}")
    print(f"  {output_dir / 'ObservMatrix.txt'}")
    print(f"  {output_dir / 'GaussObsvMatrix.txt'}")
    print(f"  {output_dir / 'particles_that_are_inside_home.txt'}")
    print(f"  {output_dir / 'detector_stat.json'}")
    print(f"  {output_dir / 'instance_id_map.txt'}")
    print(f"  {output_dir / 'photo_map.csv'}")


if __name__ == "__main__":
    main()
