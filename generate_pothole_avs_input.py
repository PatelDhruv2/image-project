#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    repo_root = Path(__file__).resolve().parent.parent
    base_dir = Path(__file__).resolve().parent
    dataset_dir = Path(__file__).resolve().parent / "dataset"
    default_output = base_dir / "finalBosok"
    default_results_json = Path(__file__).resolve().parent / "test_own_model_results.json"
    default_weights = base_dir / "runs" / "detect" / "train2" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(
        description=(
            "Generate AVS input files for the keremberke pothole dataset in the "
            "same style expected by image-AVS."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        default=str(dataset_dir),
        help="Path to the keremberke dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_output),
        help="Directory where AVS input files will be written.",
    )
    parser.add_argument(
        "--visible-count",
        type=int,
        default=27,
        help="Desired number of poses that should see the pothole in MatrixL_gt.",
    )
    parser.add_argument(
        "--fov-deg",
        type=float,
        default=None,
        help="Override horizontal FOV in degrees. If omitted, auto-tune for visible-count.",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=None,
        help="Override max visibility range. If omitted, auto-tune for visible-count.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Sigma for GaussObsvMatrix. Defaults to the inferred grid step.",
    )
    parser.add_argument(
        "--listview-mode",
        choices=("name", "absolute"),
        default="name",
        help="How to write image paths into ListView.txt.",
    )
    parser.add_argument(
        "--detector-mode",
        choices=("live_yolo", "results_json", "gt"),
        default="live_yolo",
        help=(
            "How to build MatrixL_dt. 'live_yolo' runs your local YOLOv8s weights, "
            "'results_json' uses saved per-image detector confidences, and 'gt' copies MatrixL_gt."
        ),
    )
    parser.add_argument(
        "--weights",
        default=str(default_weights),
        help="Path to local YOLOv8s weights used when --detector-mode live_yolo.",
    )
    parser.add_argument(
        "--detector-results-json",
        default=str(default_results_json),
        help="Path to saved detector results JSON from test_own_model.py.",
    )
    parser.add_argument(
        "--detector-confidence-threshold",
        type=float,
        default=0.94,
        help="Only detections at or above this confidence are kept in MatrixL_dt.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Base YOLO confidence threshold passed to the live detector.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size for live YOLO detection.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=0.5,
        help="Mask threshold for live YOLO segmentation outputs.",
    )
    parser.add_argument(
        "--ground-truth-mode",
        choices=("mask", "visibility"),
        default="mask",
        help=(
            "How to build MatrixL_gt. 'mask' keeps only annotated pothole images; "
            "'visibility' marks all poses that can see the pothole cell."
        ),
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(dataset_dir):
    dataset_dir = Path(dataset_dir).resolve()
    items = load_json(dataset_dir / "dataset.json")
    transitions = load_json(dataset_dir / "transitions.json")

    normalized = []
    for item in items:
        entry = dict(item)
        entry["image_path"] = (dataset_dir / item["image_file"]).resolve()
        entry["mask_path"] = (dataset_dir / item["mask_file"]).resolve()
        normalized.append(entry)
    return normalized, transitions


def sorted_unique_cells(items):
    coords = {
        (float(item["state"]["x"]), float(item["state"]["y"]))
        for item in items
    }
    return sorted(coords)


def infer_grid_step(cells):
    min_dist = None
    for idx, (x1, y1) in enumerate(cells):
        for x2, y2 in cells[idx + 1 :]:
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist <= 1e-9:
                continue
            if min_dist is None or dist < min_dist:
                min_dist = dist
    return min_dist if min_dist is not None else 1.0


def heading_vector(yaw_deg):
    theta = math.radians(float(yaw_deg))
    return math.sin(theta), math.cos(theta)


def compute_observability(items, cells, fov_deg, max_range):
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

            if dist <= 1e-9:
                observ[row_idx, col_idx] = 1
                continue
            if dist > max_range:
                continue

            dot = (dx * fx + dy * fy) / dist
            dot = max(-1.0, min(1.0, dot))
            angle = math.acos(dot)
            if angle <= half_fov:
                observ[row_idx, col_idx] = 1

    return observ


def choose_visibility_params(items, cells, target_cell_idx, desired_count, step):
    candidates = []
    for fov_deg in range(60, 181):
        for range_times_two in range(max(2, int(step * 2)), int(step * 10) + 1):
            max_range = range_times_two / 2.0
            observ = compute_observability(items, cells, fov_deg, max_range)
            count = int(observ[:, target_cell_idx].sum())
            error = abs(count - desired_count)
            candidates.append((error, abs(count - desired_count), abs(fov_deg - 90),
                               max_range, fov_deg, observ, count))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    _, _, _, max_range, fov_deg, observ, count = candidates[0]
    return fov_deg, max_range, observ, count


def find_target_pose(items):
    positives = [item for item in items if bool(item.get("has_pothole"))]
    if positives:
        return positives[0]

    for item in items:
        mask = np.array(Image.open(item["mask_path"]))
        if mask.ndim == 3:
            mask = mask[..., 0]
        if np.any(mask > 0):
            return item

    raise RuntimeError("Could not find any pothole-positive sample in the dataset.")


def has_foreground(mask_path):
    mask = np.array(Image.open(mask_path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return int(np.any(mask > 0))


def build_matrix_g(items, transitions):
    image_id_to_index = {str(item["image_id"]): idx for idx, item in enumerate(items)}
    rows = []

    for item in items:
        transition = transitions.get(str(item["image_id"]), {})

        def lookup(key):
            next_id = transition.get(key)
            if next_id is None:
                return -1
            return image_id_to_index.get(str(next_id), -1)

        rows.append(
            [
                lookup("ccw"),
                lookup("cw"),
                lookup("forward"),
                lookup("backward"),
                -1,
                -1,
            ]
        )

    return np.array(rows, dtype=int)


def build_gauss_matrix(items, cells, sigma):
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

    return gauss


def detector_stats(matrix_dt, matrix_gt):
    dt = matrix_dt.flatten()
    gt = matrix_gt.flatten()

    tp = int(((dt == 1) & (gt == 1)).sum())
    fp = int(((dt == 1) & (gt == 0)).sum())
    fn = int(((dt == 0) & (gt == 1)).sum())
    tn = int(((dt == 0) & (gt == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "0": {
            "name": "pothole",
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


def load_detector_confidences(results_json_path):
    results_json_path = Path(results_json_path).resolve()
    if not results_json_path.exists():
        raise FileNotFoundError(f"Missing detector results JSON: {results_json_path}")

    data = load_json(results_json_path)
    mapping = {}
    for item in data:
        image_id = str(item.get("image_id"))
        mapping[image_id] = {
            "confidence": float(item.get("confidence") or 0.0),
            "detected_pothole": bool(item.get("detected_pothole")),
        }
    return mapping


def run_live_detector(items, weights, conf, imgsz, mask_threshold):
    matplotlib_dir = Path(__file__).resolve().parent / ".matplotlib"
    matplotlib_dir.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))

    from ultralytics.yolo.utils import set_settings
    from pothole_pipeline import run_pipeline
    set_settings({"sync": False})

    return run_pipeline(
        items=items,
        mode="yolo",
        weights=str(Path(weights).resolve()),
        conf=conf,
        imgsz=imgsz,
        mask_threshold=mask_threshold,
    )


def listview_entry(item, mode):
    image_path = Path(item["image_path"]).resolve()
    if mode == "absolute":
        return str(image_path)
    return image_path.name


def write_listview(items, output_dir, mode):
    lines = [listview_entry(item, mode) for item in items]
    (output_dir / "ListView.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_positive_photo_list(items, matrix, output_path):
    visible_indices = np.flatnonzero(matrix[:, 0]).tolist()
    lines = [
        f"{idx:03d},{items[idx]['image_id']},{Path(items[idx]['image_path']).name}"
        for idx in visible_indices
    ]
    output_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return visible_indices


def write_photo_map(items, matrix_gt, matrix_dt, detector_scores, output_dir):
    lines = [
        "pose_index,image_id,image_name,image_path,x,y,yaw_deg,gt_positive,dt_positive,detector_confidence"
    ]
    for idx, item in enumerate(items):
        state = item["state"]
        detector_info = detector_scores.get(str(item["image_id"]), {})
        lines.append(
            ",".join(
                [
                    str(idx),
                    str(item["image_id"]),
                    Path(item["image_path"]).name,
                    str(item["image_path"]),
                    str(state["x"]),
                    str(state["y"]),
                    str(state["yaw_deg"]),
                    str(int(matrix_gt[idx, 0])),
                    str(int(matrix_dt[idx, 0])),
                    str(detector_info.get("confidence", 0.0)),
                ]
            )
        )
    (output_dir / "photo_map.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    items, transitions = load_dataset(args.dataset_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = sorted_unique_cells(items)
    step = infer_grid_step(cells)
    sigma = args.sigma if args.sigma is not None else step

    target_pose = find_target_pose(items)
    target_cell = (float(target_pose["state"]["x"]), float(target_pose["state"]["y"]))
    cell_to_idx = {cell: idx for idx, cell in enumerate(cells)}
    target_cell_idx = cell_to_idx[target_cell]

    if args.fov_deg is None or args.max_range is None:
        fov_deg, max_range, observ, visible_count = choose_visibility_params(
            items, cells, target_cell_idx, args.visible_count, step
        )
    else:
        fov_deg = float(args.fov_deg)
        max_range = float(args.max_range)
        observ = compute_observability(items, cells, fov_deg, max_range)
        visible_count = int(observ[:, target_cell_idx].sum())

    if args.ground_truth_mode == "visibility":
        matrix_gt = observ[:, [target_cell_idx]].astype(int)
    else:
        matrix_gt = np.array([[has_foreground(item["mask_path"])] for item in items], dtype=int)

    detector_scores = {}
    detector_results = None
    if args.detector_mode == "live_yolo":
        detector_results = run_live_detector(
            items=items,
            weights=args.weights,
            conf=args.conf,
            imgsz=args.imgsz,
            mask_threshold=args.mask_threshold,
        )
        detector_scores = {
            str(item.get("image_id")): {
                "confidence": float(item.get("confidence") or 0.0),
                "detected_pothole": bool(item.get("detected_pothole")),
            }
            for item in detector_results
        }
        matrix_dt = np.array(
            [
                [
                    int(
                        detector_scores.get(str(item["image_id"]), {}).get("detected_pothole", False)
                        and detector_scores.get(str(item["image_id"]), {}).get("confidence", 0.0)
                        >= args.detector_confidence_threshold
                    )
                ]
                for item in items
            ],
            dtype=int,
        )
    elif args.detector_mode == "results_json":
        detector_scores = load_detector_confidences(args.detector_results_json)
        matrix_dt = np.array(
            [
                [
                    int(
                        detector_scores.get(str(item["image_id"]), {}).get("detected_pothole", False)
                        and detector_scores.get(str(item["image_id"]), {}).get("confidence", 0.0)
                        >= args.detector_confidence_threshold
                    )
                ]
                for item in items
            ],
            dtype=int,
        )
    else:
        matrix_dt = matrix_gt.copy()
    matrix_g = build_matrix_g(items, transitions)
    gauss = build_gauss_matrix(items, cells, sigma)
    stats = detector_stats(matrix_dt, matrix_gt)

    write_listview(items, output_dir, args.listview_mode)
    np.savetxt(output_dir / "MatrixG.txt", matrix_g, fmt="%d")
    np.savetxt(output_dir / "MatrixL_gt.txt", matrix_gt, fmt="%d")
    np.savetxt(output_dir / "MatrixL_dt.txt", matrix_dt, fmt="%d")
    with open(output_dir / "ObservMatrix.txt", "w", encoding="utf-8") as handle:
        for row in observ:
            handle.write("".join(str(int(value)) for value in row) + "\n")
    np.savetxt(output_dir / "GaussObsvMatrix.txt", gauss, fmt="%.6f")
    (output_dir / "particles_that_are_inside_home.txt").write_text(
        " ".join(["1"] * len(cells)) + "\n", encoding="utf-8"
    )
    (output_dir / "instance_id_map.txt").write_text("pothole 0\n", encoding="utf-8")
    if detector_results is not None:
        (output_dir / "detector_results_live.json").write_text(
            json.dumps(detector_results, indent=2, default=str),
            encoding="utf-8",
        )
    with open(output_dir / "detector_stat.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)
    gt_positive_indices = write_positive_photo_list(
        items, matrix_gt, output_dir / "pothole_positive_photos.txt"
    )
    dt_positive_indices = write_positive_photo_list(
        items, matrix_dt, output_dir / "detected_pothole_photos.txt"
    )
    write_photo_map(items, matrix_gt, matrix_dt, detector_scores, output_dir)

    summary_lines = [
        f"dataset_dir={Path(args.dataset_dir).resolve()}",
        f"output_dir={output_dir}",
        f"num_poses={len(items)}",
        f"num_cells={len(cells)}",
        f"target_pose_image_id={target_pose['image_id']}",
        f"target_cell_idx={target_cell_idx}",
        f"target_cell={target_cell}",
        f"ground_truth_mode={args.ground_truth_mode}",
        f"detector_mode={args.detector_mode}",
        f"detector_confidence_threshold={args.detector_confidence_threshold}",
        f"weights={Path(args.weights).resolve()}",
        f"yolo_conf={args.conf}",
        f"yolo_imgsz={args.imgsz}",
        f"fov_deg={fov_deg}",
        f"max_range={max_range}",
        f"sigma={sigma}",
        f"pothole_visible_pose_count={visible_count}",
        f"ground_truth_positive_count={int(matrix_gt.sum())}",
        f"detected_positive_count={int(matrix_dt.sum())}",
        "ground_truth_positive_indices=" + ",".join(str(idx) for idx in gt_positive_indices),
        "detected_positive_indices=" + ",".join(str(idx) for idx in dt_positive_indices),
    ]
    (output_dir / "generation_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    print(f"Wrote AVS pothole input to: {output_dir}")
    print(f"Target pothole cell: index {target_cell_idx} at {target_cell}")
    print(f"Visibility params: fov_deg={fov_deg}, max_range={max_range}")
    print(f"Ground-truth positives: {int(matrix_gt.sum())}")
    print(f"Detector positives: {int(matrix_dt.sum())}")
    print(f"Ground-truth list: {output_dir / 'pothole_positive_photos.txt'}")
    print(f"Detector list: {output_dir / 'detected_pothole_photos.txt'}")


if __name__ == "__main__":
    main()
