#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import load_test_dataset, sample_name_for_dataset
from mdm.model.v2 import MDMModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LingBot-Depth HAMMER/ClearPose/DREDS inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Local LingBot-Depth checkpoint path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="HAMMER, ClearPose, or DREDS JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/output",
        help="Directory for args, predictions, and optional visualizations",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw depth type. ClearPose only supports d435; DREDS ignores this value.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="Scale factor used to convert uint depth to meters",
    )
    parser.add_argument(
        "--max-depth",
        type=float,
        default=None,
        help="Maximum raw input depth in meters. Defaults to dataset depth-range max.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Dataloader batch size. The model is still run one image at a time.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device",
    )
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=9,
        help="LingBot-Depth resolution level passed to MDMModel.infer",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Enable autocast inside MDMModel.infer on CUDA",
    )
    parser.add_argument(
        "--apply-mask",
        action="store_true",
        help="Apply the model-predicted mask before saving depth",
    )
    parser.add_argument(
        "--save-vis",
        action="store_true",
        help="Save optional RGB/raw/pred/GT visualization images",
    )
    parser.add_argument(
        "--image-min",
        type=float,
        default=0.1,
        help="Minimum depth for optional visualization",
    )
    parser.add_argument(
        "--image-max",
        type=float,
        default=5.0,
        help="Maximum depth for optional visualization",
    )
    return parser.parse_args()


def select_device(device_name):
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(device_name)


def validate_inputs(args):
    if not os.path.exists(args.dataset):
        raise FileNotFoundError(f"Dataset file not found: {args.dataset}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    os.makedirs(args.output, exist_ok=True)


def squeeze_depth(depth):
    depth = np.asarray(depth)
    if depth.ndim == 3:
        if depth.shape[-1] == 1:
            depth = depth[..., 0]
        else:
            depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected 2D depth map, got shape {depth.shape}")
    return depth


def load_rgb(rgb_path, device):
    image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return image_rgb, image_tensor


def load_raw_depth(depth_path, depth_scale, max_depth, device):
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to read raw depth: {depth_path}")

    depth = squeeze_depth(depth).astype(np.float32) / depth_scale
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    depth = np.where(valid, depth, 0.0).astype(np.float32)

    depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device)
    return depth, depth_tensor


def load_gt_depth(depth_path, depth_scale, max_depth, min_depth):
    depth_gt = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_gt is None:
        raise ValueError(f"Could not load GT depth from {depth_path}")

    depth_gt = squeeze_depth(depth_gt).astype(np.float32) / depth_scale
    valid = np.isfinite(depth_gt) & (depth_gt >= min_depth) & (depth_gt <= max_depth)
    return np.where(valid, depth_gt, 0.0).astype(np.float32)


def normalize_prediction(pred_depth, target_shape):
    pred = np.asarray(pred_depth, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected HxW prediction, got shape {pred.shape}")
    if pred.shape != target_shape:
        pred = cv2.resize(
            pred,
            (target_shape[1], target_shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred.astype(np.float32, copy=False)


def colorize_depth(depth, vmin, vmax):
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    scaled = (scaled * 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def resize_for_visualization(depth, target_shape):
    if depth.shape == target_shape:
        return depth
    return cv2.resize(
        depth.astype(np.float32, copy=False),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )


def save_visualization(
    output_dir,
    name,
    rgb,
    raw_depth,
    pred_depth,
    gt_depth,
    image_min,
    image_max,
):
    target_shape = rgb.shape[:2]
    raw_depth = resize_for_visualization(raw_depth, target_shape)
    pred_depth = resize_for_visualization(pred_depth, target_shape)
    gt_depth = resize_for_visualization(gt_depth, target_shape)

    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raw_vis = colorize_depth(raw_depth, image_min, image_max)
    pred_vis = colorize_depth(pred_depth, image_min, image_max)
    gt_vis = colorize_depth(gt_depth, image_min, image_max)
    grid = np.concatenate([rgb_bgr, raw_vis, pred_vis, gt_vis], axis=1)
    cv2.imwrite(str(Path(output_dir) / f"{name}_vis.jpg"), grid)


@torch.inference_mode()
def run_inference(args):
    validate_inputs(args)
    device = select_device(args.device)

    dataset, dataset_kind = load_test_dataset(args.dataset, args.raw_type)
    args.dataset_kind = dataset_kind
    if hasattr(dataset, "depth_scale"):
        args.depth_scale = dataset.depth_scale

    min_depth, dataset_max_depth = dataset.depth_range
    max_depth = args.max_depth if args.max_depth is not None else dataset_max_depth

    predictions_dir = Path(args.output) / "predictions"
    visualizations_dir = Path(args.output) / "visualizations"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    if args.save_vis:
        visualizations_dir.mkdir(parents=True, exist_ok=True)

    model = MDMModel.from_pretrained(args.model_path).to(device).eval()

    args.resolved_model_module = "mdm.model.v2"
    args.resolved_model_class = "MDMModel"
    args.output_kind = "metric_depth_meter"
    args.alignment = "none"
    args.dataset_min_depth = min_depth
    args.dataset_max_depth = dataset_max_depth
    args.effective_max_depth = max_depth
    args.device_resolved = str(device)

    with open(Path(args.output) / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    use_fp16 = args.use_fp16 and device.type == "cuda"

    for batch_items in tqdm(dataloader, desc="LingBot-Depth inference"):
        rgb_paths, raw_depth_paths, gt_depth_paths = batch_items
        for rgb_path, raw_depth_path, gt_depth_path in zip(
            rgb_paths,
            raw_depth_paths,
            gt_depth_paths,
        ):
            rgb_path = str(rgb_path)
            raw_depth_path = str(raw_depth_path)
            gt_depth_path = str(gt_depth_path)
            name = sample_name_for_dataset(dataset_kind, rgb_path)

            rgb, image_tensor = load_rgb(rgb_path, device)
            raw_depth, depth_tensor = load_raw_depth(
                raw_depth_path,
                depth_scale=args.depth_scale,
                max_depth=max_depth,
                device=device,
            )

            if raw_depth.shape != rgb.shape[:2]:
                raise ValueError(
                    f"RGB/depth shape mismatch for {name}: "
                    f"rgb={rgb.shape[:2]}, depth={raw_depth.shape}"
                )

            output = model.infer(
                image_tensor,
                depth_in=depth_tensor,
                intrinsics=None,
                resolution_level=args.resolution_level,
                apply_mask=args.apply_mask,
                use_fp16=use_fp16,
            )
            pred_depth = output["depth"].detach().cpu().numpy()
            pred_depth = normalize_prediction(pred_depth, raw_depth.shape)

            np.save(predictions_dir / f"{name}.npy", pred_depth)

            if args.save_vis:
                gt_depth = load_gt_depth(
                    gt_depth_path,
                    depth_scale=args.depth_scale,
                    max_depth=max_depth,
                    min_depth=min_depth,
                )
                save_visualization(
                    visualizations_dir,
                    name,
                    rgb,
                    raw_depth,
                    pred_depth,
                    gt_depth,
                    args.image_min,
                    args.image_max,
                )


if __name__ == "__main__":
    run_inference(parse_arguments())
