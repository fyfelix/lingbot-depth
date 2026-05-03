#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset import ClearPoseDataset, HAMMERDataset
from mdm.model.v2 import MDMModel


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="LingBot-Depth HAMMER/ClearPose inference",
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
        help="HAMMER or ClearPose JSONL path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation/output",
        help="Directory for .npy predictions and args.json",
    )
    parser.add_argument(
        "--raw-type",
        type=str,
        required=True,
        choices=["d435", "l515", "tof"],
        help="Raw depth type. ClearPose only supports d435.",
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
        help="Save optional RGB/raw/pred visualization images",
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


def load_dataset_for_eval(dataset_path, raw_type):
    dataset_lower = dataset_path.lower()
    if "clearpose" in dataset_lower:
        if raw_type != "d435":
            raise ValueError("ClearPose dataset only supports raw-type=d435")
        return ClearPoseDataset(dataset_path)
    if "hammer" in dataset_lower:
        return HAMMERDataset(dataset_path, raw_type)
    raise ValueError(f"Invalid dataset: {dataset_path}")


def resolve_sample_name(rgb_path, dataset_path):
    parts = Path(rgb_path).parts
    dataset_lower = dataset_path.lower()

    if "hammer" in dataset_lower:
        if len(parts) < 4:
            raise ValueError(f"Unexpected HAMMER rgb path: {rgb_path}")
        return f"{parts[-4]}#{Path(rgb_path).stem}"

    if "clearpose" in dataset_lower:
        if len(parts) < 3:
            raise ValueError(f"Unexpected ClearPose rgb path: {rgb_path}")
        return f"{'#'.join(parts[-3:-1])}#{Path(rgb_path).stem}"

    raise ValueError(f"Invalid dataset: {dataset_path}")


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

    depth = depth.astype(np.float32) / depth_scale
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is not None:
        valid &= depth <= max_depth
    depth = np.where(valid, depth, 0.0).astype(np.float32)

    depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device)
    return depth, depth_tensor


def normalize_prediction(pred_depth, target_shape):
    pred = np.asarray(pred_depth, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected HxW prediction, got shape {pred.shape}")
    if pred.shape != target_shape:
        pred = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    return pred.astype(np.float32, copy=False)


def colorize_depth(depth, vmin, vmax):
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.clip((depth - vmin) / max(vmax - vmin, 1e-6), 0, 1)
    scaled = (scaled * 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def save_visualization(output_dir, name, rgb, raw_depth, pred_depth, image_min, image_max):
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    raw_vis = colorize_depth(raw_depth, image_min, image_max)
    pred_vis = colorize_depth(pred_depth, image_min, image_max)
    grid = np.concatenate([rgb_bgr, raw_vis, pred_vis], axis=1)
    cv2.imwrite(str(Path(output_dir) / f"{name}_vis.jpg"), grid)


@torch.inference_mode()
def run_inference(args):
    validate_inputs(args)
    device = select_device(args.device)

    dataset = load_dataset_for_eval(args.dataset, args.raw_type)
    min_depth, dataset_max_depth = dataset.depth_range
    max_depth = args.max_depth if args.max_depth is not None else dataset_max_depth

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
        rgb_paths, raw_depth_paths, _gt_depth_paths = batch_items
        for rgb_path, raw_depth_path in zip(rgb_paths, raw_depth_paths):
            rgb_path = str(rgb_path)
            raw_depth_path = str(raw_depth_path)
            name = resolve_sample_name(rgb_path, args.dataset)

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

            np.save(Path(args.output) / f"{name}.npy", pred_depth)

            if args.save_vis:
                save_visualization(
                    args.output,
                    name,
                    rgb,
                    raw_depth,
                    pred_depth,
                    args.image_min,
                    args.image_max,
                )


if __name__ == "__main__":
    run_inference(parse_arguments())
