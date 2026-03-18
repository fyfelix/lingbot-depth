#!/usr/bin/env python3
import argparse
import os
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader, Dataset
import json
import time

from mdm.model.v2 import MDMModel

# Automatically select the best available device for inference
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_arguments():
    parser = argparse.ArgumentParser(description="LingBot-Depth Batch Inference")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model directory or .pt file")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset file")
    parser.add_argument("--output", type=str, default="output_dir", help="Output directory")
    parser.add_argument("--raw-type", type=str, required=True, choices=["d435", "l515", "tof"], help="Camera raw type")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Scale factor for depth values")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--max-depth", type=float, default=6.0, help="Max depth limit for raw truncation")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Min valid depth")
    return parser.parse_args()


from scripts.utils.test_datasets import HAMMERDataset, ClearPoseDataset, load_images

def batch_collate(batch):
    rgb_paths = [item[0] for item in batch]
    raw_depth_paths = [item[1] for item in batch]
    gt_depth_paths = [item[2] for item in batch]
    return rgb_paths, raw_depth_paths, gt_depth_paths

@torch.no_grad()
def inference(args):
    os.makedirs(args.output, exist_ok=True)
    
    with open(os.path.join(args.output, 'args.json'), 'w') as f:
        json.dump(vars(args), f)

    logger.info(f"Loading MDMModel from {args.model_path}...")
    # Supports loading from a local .pt file or directory
    model = MDMModel.from_pretrained(args.model_path).to(DEVICE)
    model.eval()
    logger.info("Model loaded successfully.")

    if 'clearpose' in args.dataset.lower():
        dataset = ClearPoseDataset(args.dataset)
    elif 'hammer' in args.dataset.lower():
        dataset = HAMMERDataset(args.dataset, args.raw_type)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
        logger.error(f"Failed to load any samples from {args.dataset}. Please check the JSONL format.")
        sys.exit(1)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=batch_collate,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    for batch_idx, (rgb_paths, raw_depth_paths, gt_depth_paths) in enumerate(tqdm(dataloader, desc="Processing batches")):
        
        batch_rgb_tensors = []
        batch_depth_tensors = []
        names = []
        valid_paths = []

        # Load and preprocess data
        for rgb_path, raw_depth_path in zip(rgb_paths, raw_depth_paths):
            tmp = rgb_path.split('/')
            # Use HAMMER naming convention
            if len(tmp) >= 4:
                scene_name = tmp[-4]
                name = scene_name + '#' + tmp[-1].split('.')[0]
            else:
                name = Path(rgb_path).stem
            names.append(name)

            try:
                rgb_src, depth_src, _ = load_images(rgb_path, raw_depth_path, args.depth_scale, args.max_depth)
                
                # Image to float32 [0, 1] tensor shape (3, H, W)
                img_tensor = torch.from_numpy(rgb_src / 255.0).float().permute(2, 0, 1)
                
                # Depth to float32 tensor shape (H, W)
                depth_tensor = torch.from_numpy(depth_src).float()
                
                batch_rgb_tensors.append(img_tensor)
                batch_depth_tensors.append(depth_tensor)
                valid_paths.append(True)
            except Exception as e:
                logger.error(f"Error loading {rgb_path}: {e}")
                valid_paths.append(False)

        # Filter out failed loads
        batch_rgb_tensors = [t for t, v in zip(batch_rgb_tensors, valid_paths) if v]
        batch_depth_tensors = [t for t, v in zip(batch_depth_tensors, valid_paths) if v]
        names = [n for n, v in zip(names, valid_paths) if v]
        
        if not batch_rgb_tensors:
            continue

        # Stack to batch (Assuming all images in a batch have the same dimensions, 
        # which is typically true for HAMMER benchmark. If not, batch size must be 1)
        try:
            batch_images = torch.stack(batch_rgb_tensors).to(DEVICE)
            batch_depths = torch.stack(batch_depth_tensors).to(DEVICE)
        except RuntimeError as e:
            logger.warning(f"Images in batch have different sizes, processing one by one. Error: {e}")
            # Fallback to batch size 1 processing
            for img, dep, name in zip(batch_rgb_tensors, batch_depth_tensors, names):
                output = model.infer(
                    img.unsqueeze(0).to(DEVICE),
                    depth_in=dep.unsqueeze(0).to(DEVICE),
                    apply_mask=False # User requested no intrinsics/points
                )
                pred_depth = output['depth'].squeeze().cpu().numpy()
                np.save(os.path.join(args.output, f"{name}.npy"), pred_depth)
            continue
            
        # Run MDMModel inference
        output = model.infer(
            batch_images,
            depth_in=batch_depths,
            apply_mask=False # Ignore mask/points as requested
        )
        
        pred_depths = output['depth'].cpu().numpy() # [B, H, W]

        # Save predictions
        for i, name in enumerate(names):
            if pred_depths.ndim == 3:
                pred = pred_depths[i]
            else:
                pred = pred_depths # batch size 1 fallback
            np.save(os.path.join(args.output, f"{name}.npy"), pred)

if __name__ == "__main__":
    args = parse_arguments()
    inference(args)
