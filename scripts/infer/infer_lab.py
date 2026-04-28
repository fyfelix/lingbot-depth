#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join("/tmp", f"matplotlib-{os.getuid() if hasattr(os, 'getuid') else 'cache'}"),
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LingBot-Depth lab RGB-D batch inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Path or Hugging Face ID")
    parser.add_argument("--dataset", required=True, help="Path to lab JSONL index")
    parser.add_argument("--output", required=True, help="Directory for predictions")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--depth-scale", type=float, default=1000.0)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument(
        "--vis-max-depth",
        type=float,
        default=2.0,
        help="Fallback depth visualization max in meters when no valid percentile can be computed",
    )
    parser.add_argument(
        "--vis-percentile",
        type=float,
        default=99.0,
        help="Valid-depth percentile used as the visualization upper bound",
    )
    parser.add_argument("--pc-rot-x-deg", type=float, default=25.0)
    parser.add_argument("--pc-rot-y-deg", type=float, default=15.0)
    parser.add_argument("--pc-knn-k", type=int, default=16)
    parser.add_argument("--pc-knn-std-ratio", type=float, default=2.0)
    parser.add_argument("--disable-pc-knn-filter", action="store_true")
    parser.add_argument("--save-vis", action="store_true", help="Save analysis grids")
    return parser.parse_args()


if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
    parse_arguments()


import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw
from loguru import logger
from scipy.spatial import cKDTree
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from mdm.model.v2 import MDMModel

matplotlib.use("Agg")


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class LabDataset(Dataset):
    """Dataset for lab-captured RGB-D samples indexed by a JSONL file."""

    REQUIRED_FIELDS = ("sample_id", "rgb", "raw_depth")
    OPTIONAL_PATH_FIELDS = ("raw_depth_u16", "meta")

    def __init__(self, jsonl_path):
        self.jsonl_path = Path(jsonl_path)
        self.root = self.jsonl_path.parent
        self.data = []

        with self.jsonl_path.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, start=1):
                line = line.strip()
                if not line:
                    continue

                item = json.loads(line)
                self._validate_item(item, line_no)
                self.data.append(self._resolve_item(item))

        if not self.data:
            raise ValueError(f"Lab dataset index is empty: {self.jsonl_path}")

    def _validate_item(self, item, line_no):
        missing_fields = [field for field in self.REQUIRED_FIELDS if not item.get(field)]
        if missing_fields:
            raise ValueError(
                f"Missing required fields at {self.jsonl_path}:{line_no}: {missing_fields}"
            )

    def _resolve_path(self, value):
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        return path

    def _resolve_item(self, item):
        resolved = dict(item)

        for field in ("rgb", "raw_depth", *self.OPTIONAL_PATH_FIELDS):
            if field not in item or not item[field]:
                continue
            path = self._resolve_path(item[field])
            if not path.exists():
                raise FileNotFoundError(
                    f"Lab dataset sample {item['sample_id']} references missing "
                    f"{field} file: {path}"
                )
            resolved[field] = str(path)

        return resolved

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "sample_id": item["sample_id"],
            "rgb": item["rgb"],
            "raw_depth": item["raw_depth"],
            "raw_depth_u16": item.get("raw_depth_u16", ""),
            "meta": item.get("meta", ""),
            "width": int(item.get("width", 0)),
            "height": int(item.get("height", 0)),
            "aligned_to": item.get("aligned_to", ""),
        }


def validate_inputs(args):
    if args.batch_size < 1:
        raise ValueError("--batch-size must be greater than 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.depth_scale <= 0:
        raise ValueError("--depth-scale must be greater than 0")
    if args.max_depth <= 0:
        raise ValueError("--max-depth must be greater than 0")
    if args.vis_max_depth <= 0:
        raise ValueError("--vis-max-depth must be greater than 0")
    if not (0.0 < args.vis_percentile <= 100.0):
        raise ValueError("--vis-percentile must be in the range (0, 100]")
    if args.pc_knn_k < 1:
        raise ValueError("--pc-knn-k must be greater than 0")
    if args.pc_knn_std_ratio < 0:
        raise ValueError("--pc-knn-std-ratio must be non-negative")
    if args.model_path.endswith(".pt") and not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    if not Path(args.dataset).exists():
        raise FileNotFoundError(f"Dataset index does not exist: {args.dataset}")
    Path(args.output).mkdir(parents=True, exist_ok=True)


def load_rgbd(rgb_path, depth_path, depth_scale, max_depth):
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")
    rgb = rgb_bgr[:, :, ::-1]

    raw_depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if raw_depth is None:
        raise ValueError(f"Could not load raw depth image: {depth_path}")
    if raw_depth.ndim == 3:
        raw_depth = raw_depth[:, :, 0]

    depth = np.asarray(raw_depth).astype(np.float32) / depth_scale
    invalid = (~np.isfinite(depth)) | (depth < 0) | (depth > max_depth)
    depth[invalid] = 0.0

    target_hw = rgb.shape[:2]
    if depth.shape != target_hw:
        logger.warning(
            f"Resizing raw depth from {depth.shape} to RGB shape {target_hw}: {depth_path}"
        )
        depth = cv2.resize(depth, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_NEAREST)

    return rgb, depth.astype(np.float32, copy=False)


def load_lab_intrinsics(meta_path):
    if not meta_path:
        raise ValueError("Lab visualization requires a meta JSON path with intrinsics")

    with open(meta_path, "r", encoding="utf-8") as file:
        meta = json.load(file)

    intrinsics_meta = meta.get("color_intrinsics") or meta.get("depth_intrinsics")
    if not intrinsics_meta:
        raise ValueError(f"Meta file does not contain intrinsics: {meta_path}")

    if isinstance(intrinsics_meta, list):
        intrinsics = np.asarray(intrinsics_meta, dtype=np.float32)
        if intrinsics.shape != (3, 3):
            raise ValueError(f"Intrinsics matrix must be 3x3: {meta_path}")
        return intrinsics

    fx = float(intrinsics_meta["fx"])
    fy = float(intrinsics_meta["fy"])
    cx = float(intrinsics_meta.get("ppx", intrinsics_meta.get("cx")))
    cy = float(intrinsics_meta.get("ppy", intrinsics_meta.get("cy")))
    if fx == 0.0 or fy == 0.0:
        raise ValueError(f"Intrinsics fx/fy must be non-zero: {meta_path}")
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def colorize_depth(depth, fallback_max_depth, percentile=99.0):
    depth = np.asarray(depth, dtype=np.float32)
    valid = (depth > 0) & np.isfinite(depth)
    if valid.any():
        max_depth = float(np.nanpercentile(depth[valid], percentile))
    else:
        max_depth = float(fallback_max_depth)
    if not np.isfinite(max_depth) or max_depth <= 0:
        max_depth = float(fallback_max_depth)

    norm = np.clip(depth / max_depth, 0.0, 1.0)
    colored = matplotlib.colormaps["Spectral_r"](norm)[..., :3]
    colored = (colored * 255).astype(np.uint8)
    colored[~valid] = 0
    return colored


def filter_pointcloud_knn(points, colors, k=16, std_ratio=2.0):
    if k < 1 or points.shape[0] <= k:
        return points, colors

    neighbor_count = min(k + 1, points.shape[0])
    try:
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=neighbor_count, workers=-1)
    except Exception:
        return points, colors

    if distances.ndim == 1:
        return points, colors

    mean_distances = distances[:, 1:].mean(axis=1)
    finite = np.isfinite(mean_distances)
    if not finite.any():
        return points, colors

    valid_mean_distances = mean_distances[finite]
    threshold = valid_mean_distances.mean() + std_ratio * valid_mean_distances.std()
    keep = finite & (mean_distances <= threshold)
    if not keep.any():
        return points, colors
    return points[keep], colors[keep]


def render_pointcloud_reproject(
    depth_map,
    intrinsics,
    rgb_img,
    rot_x_deg=25.0,
    rot_y_deg=15.0,
    bg_color=(255, 255, 255),
    knn_filter=True,
    knn_k=16,
    knn_std_ratio=2.0,
):
    height, width = depth_map.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    valid = (depth_map > 1e-8) & np.isfinite(depth_map)
    if not valid.any():
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=-1).astype(np.float32, copy=False)
    colors = np.clip(rgb_img, 0, 255).astype(np.uint8)[valid]
    if knn_filter:
        points, colors = filter_pointcloud_knn(
            points,
            colors,
            k=knn_k,
            std_ratio=knn_std_ratio,
        )

    center = points.mean(axis=0)
    points_centered = points - center

    rx = np.radians(rot_x_deg)
    ry = np.radians(rot_y_deg)
    cos_x, sin_x = np.cos(rx), np.sin(rx)
    cos_y, sin_y = np.cos(ry), np.sin(ry)

    x1 = points_centered[:, 0]
    y1 = points_centered[:, 1] * cos_x - points_centered[:, 2] * sin_x
    z1 = points_centered[:, 1] * sin_x + points_centered[:, 2] * cos_x
    x2 = x1 * cos_y + z1 * sin_y
    y2 = y1
    z2 = -x1 * sin_y + z1 * cos_y
    points_rot = np.stack([x2, y2, z2], axis=-1) + center
    z_new = points_rot[:, 2]
    keep = z_new > 1e-4
    if not keep.any():
        return np.full((height, width, 3), bg_color, dtype=np.uint8)

    u_proj = points_rot[keep, 0] * fx / z_new[keep] + cx
    v_proj = points_rot[keep, 1] * fy / z_new[keep] + cy
    z_buf = z_new[keep]
    c_buf = colors[keep]

    pad = int(max(height, width) * 0.3)
    canvas_h, canvas_w = height + 2 * pad, width + 2 * pad
    ui = np.round(u_proj + pad).astype(np.int32)
    vi = np.round(v_proj + pad).astype(np.int32)

    in_bounds = (ui >= 0) & (ui < canvas_w) & (vi >= 0) & (vi < canvas_h)
    ui = ui[in_bounds]
    vi = vi[in_bounds]
    z_buf = z_buf[in_bounds]
    c_buf = c_buf[in_bounds]

    order = np.argsort(-z_buf)
    ui = ui[order]
    vi = vi[order]
    c_buf = c_buf[order]

    canvas = np.full((canvas_h, canvas_w, 3), bg_color, dtype=np.uint8)
    canvas[vi, ui] = c_buf

    filled = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    filled[vi, ui] = 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    filled_dilated = cv2.dilate(filled, kernel, iterations=1)
    holes = (filled_dilated > 0) & (filled == 0)
    if holes.any():
        for channel_idx in range(3):
            blurred = cv2.blur(canvas[:, :, channel_idx].astype(np.float32), (3, 3))
            canvas[:, :, channel_idx][holes] = blurred[holes].astype(np.uint8)

    rows = np.any(filled_dilated > 0, axis=1)
    cols = np.any(filled_dilated > 0, axis=0)
    if rows.any() and cols.any():
        row_min, row_max = np.where(rows)[0][[0, -1]]
        col_min, col_max = np.where(cols)[0][[0, -1]]
        margin = 10
        row_min = max(0, row_min - margin)
        row_max = min(canvas_h - 1, row_max + margin)
        col_min = max(0, col_min - margin)
        col_max = min(canvas_w - 1, col_max + margin)
        canvas = canvas[row_min : row_max + 1, col_min : col_max + 1]

    return canvas


def add_title(image, title, height=28):
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + height), color=(255, 255, 255))
    canvas.paste(Image.fromarray(image), (0, height))
    drawer = ImageDraw.Draw(canvas)
    drawer.text((8, 6), title, fill=(0, 0, 0))
    return np.asarray(canvas)


def blank_panel(height, width, color=(255, 255, 255)):
    return np.full((height, width, 3), color, dtype=np.uint8)


def resize_to(image, target_hw):
    target_h, target_w = target_hw
    if image.shape[0] == target_h and image.shape[1] == target_w:
        return image
    return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)


def resize_depth_to(depth, target_hw):
    target_h, target_w = target_hw
    if depth.shape == target_hw:
        return depth.astype(np.float32)
    return cv2.resize(depth.astype(np.float32), (target_w, target_h), interpolation=cv2.INTER_NEAREST)


def make_grid(images, nrow):
    if len(images) % nrow != 0:
        raise ValueError("images length must be divisible by nrow")
    rows = []
    for index in range(0, len(images), nrow):
        rows.append(np.concatenate(images[index : index + nrow], axis=1))
    return np.concatenate(rows, axis=0)


def make_lab_visualization(
    rgb,
    raw_depth,
    pred_depth,
    intrinsics,
    vis_max_depth,
    vis_percentile,
    pc_rot_x_deg,
    pc_rot_y_deg,
    pc_knn_k,
    pc_knn_std_ratio,
    disable_pc_knn_filter,
):
    target_hw = rgb.shape[:2]
    raw_depth = resize_depth_to(raw_depth, target_hw)
    pred_depth = resize_depth_to(pred_depth, target_hw)

    raw_vis = colorize_depth(raw_depth, vis_max_depth, vis_percentile)
    pred_vis = colorize_depth(pred_depth, vis_max_depth, vis_percentile)
    raw_pc = render_pointcloud_reproject(
        raw_depth,
        intrinsics,
        rgb,
        rot_x_deg=pc_rot_x_deg,
        rot_y_deg=pc_rot_y_deg,
        knn_filter=False,
    )
    pred_pc = render_pointcloud_reproject(
        pred_depth,
        intrinsics,
        rgb,
        rot_x_deg=pc_rot_x_deg,
        rot_y_deg=pc_rot_y_deg,
        knn_filter=not disable_pc_knn_filter,
        knn_k=pc_knn_k,
        knn_std_ratio=pc_knn_std_ratio,
    )

    raw_pc = resize_to(raw_pc, target_hw)
    pred_pc = resize_to(pred_pc, target_hw)
    empty = blank_panel(*target_hw)

    panels = [
        add_title(np.clip(rgb, 0, 255).astype(np.uint8), "RGB"),
        add_title(raw_vis, "Raw Depth"),
        add_title(pred_vis, "Pred Depth"),
        add_title(empty, ""),
        add_title(raw_pc, "Raw Point Cloud"),
        add_title(pred_pc, "Pred Point Cloud"),
    ]

    return Image.fromarray(make_grid(panels, nrow=3))


def batch_to_paths(batch):
    return {
        "sample_id": list(batch["sample_id"]),
        "rgb": list(batch["rgb"]),
        "raw_depth": list(batch["raw_depth"]),
        "meta": list(batch["meta"]),
    }


def infer_group(model, image_tensors, depth_tensors, use_fp16):
    images = torch.stack(image_tensors).to(device=model.device, dtype=model.dtype)
    depths = torch.stack(depth_tensors).to(device=model.device, dtype=model.dtype)

    start_time = time.time()
    output = model.infer(
        images,
        depth_in=depths,
        apply_mask=False,
        use_fp16=use_fp16,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.time() - start_time) * 1000.0

    pred_depths = output["depth"].float().cpu().numpy()
    if pred_depths.ndim == 2:
        pred_depths = pred_depths[None, ...]
    logger.info(
        f"Forward pass: {elapsed_ms:.2f} ms for batch_size={len(image_tensors)} "
        f"({elapsed_ms / len(image_tensors):.2f} ms per image)"
    )
    return [np.nan_to_num(pred.astype(np.float32), posinf=0.0, neginf=0.0) for pred in pred_depths]


@torch.no_grad()
def infer_depths(model, rgb_images, depth_images, use_fp16=True):
    image_tensors = []
    depth_tensors = []
    shape_groups = defaultdict(list)

    for index, (rgb, depth) in enumerate(zip(rgb_images, depth_images)):
        image_tensor = torch.from_numpy(rgb / 255.0).float().permute(2, 0, 1)
        depth_tensor = torch.from_numpy(depth).float()
        image_tensors.append(image_tensor)
        depth_tensors.append(depth_tensor)
        shape_groups[(image_tensor.shape, depth_tensor.shape)].append(index)

    results = [None] * len(rgb_images)
    for indices in shape_groups.values():
        predictions = infer_group(
            model,
            [image_tensors[index] for index in indices],
            [depth_tensors[index] for index in indices],
            use_fp16=use_fp16,
        )
        for index, prediction in zip(indices, predictions):
            results[index] = prediction

    return results


def save_numpy(output_dir, sample_id, pred):
    path = Path(output_dir) / f"{sample_id}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pred)


def save_visualization(output_dir, sample_id, visualization):
    path = Path(output_dir) / f"{sample_id}_analysis_vis.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    visualization.save(path)


def run_inference(args):
    validate_inputs(args)

    logger.info(f"Loading MDMModel from {args.model_path} on {DEVICE}...")
    model = MDMModel.from_pretrained(args.model_path).to(DEVICE).eval()
    logger.info("Model loaded successfully.")

    run_args = vars(args).copy()
    run_args["device"] = str(DEVICE)
    with open(Path(args.output) / "args.json", "w", encoding="utf-8") as file:
        json.dump(run_args, file, indent=2)

    dataset = LabDataset(args.dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    for batch in tqdm(dataloader, desc="Processing lab batches"):
        batch_paths = batch_to_paths(batch)
        rgb_images = []
        raw_depths = []
        intrinsics_list = []

        for rgb_path, depth_path, meta_path in zip(
            batch_paths["rgb"],
            batch_paths["raw_depth"],
            batch_paths["meta"],
        ):
            rgb, raw_depth = load_rgbd(
                rgb_path,
                depth_path,
                depth_scale=args.depth_scale,
                max_depth=args.max_depth,
            )
            rgb_images.append(rgb)
            raw_depths.append(raw_depth)
            if args.save_vis:
                intrinsics_list.append(load_lab_intrinsics(meta_path))

        predictions = infer_depths(model, rgb_images, raw_depths)

        for index, (sample_id, rgb, raw_depth, pred) in enumerate(
            zip(batch_paths["sample_id"], rgb_images, raw_depths, predictions)
        ):
            save_numpy(args.output, sample_id, pred)

            if args.save_vis:
                visualization = make_lab_visualization(
                    rgb,
                    raw_depth,
                    pred,
                    intrinsics=intrinsics_list[index],
                    vis_max_depth=args.vis_max_depth,
                    vis_percentile=args.vis_percentile,
                    pc_rot_x_deg=args.pc_rot_x_deg,
                    pc_rot_y_deg=args.pc_rot_y_deg,
                    pc_knn_k=args.pc_knn_k,
                    pc_knn_std_ratio=args.pc_knn_std_ratio,
                    disable_pc_knn_filter=args.disable_pc_knn_filter,
                )
                save_visualization(args.output, sample_id, visualization)


if __name__ == "__main__":
    run_inference(parse_arguments())
