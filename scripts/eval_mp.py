import os
import sys
import argparse
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from os.path import join, exists
from datetime import datetime
import json
from tqdm import tqdm
from loguru import logger

# Add project root to sys.path to import from external_files
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.metric import (
    abs_relative_difference,
    rmse_linear,
    delta1_acc,
    mae_linear,
    delta4_acc_105,
    delta5_acc110
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate Depth Metrics")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the JSONL dataset")
    parser.add_argument("--output", type=str, required=True, help="Prediction output directory")
    parser.add_argument("--raw-type", type=str, default="d435", help="Camera raw type")
    parser.add_argument("--depth-scale", type=float, default=1000.0, help="Scale factor for depth")
    parser.add_argument("--min-depth", type=float, default=0.1, help="Min valid depth")
    parser.add_argument("--max-depth", type=float, default=5.0, help="Max valid depth")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--align", action="store_true", help="Enable scaling to align pred and gt")
    return parser.parse_args()

from scripts.utils.test_datasets import HAMMERDataset, ClearPoseDataset

def load_gt_depth(depth_path, depth_scale, max_depth, min_depth):
    depth_GT = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_GT is None:
        raise ValueError(f"Could not load GT depth from {depth_path}")
    depth_GT = np.asarray(depth_GT).astype(np.float32) / depth_scale
    valid_mask = (depth_GT >= min_depth) & (depth_GT <= max_depth)
    depth_GT[~valid_mask] = min_depth
    return depth_GT, valid_mask

class EvalDataset(Dataset):
    def __init__(self, jsonl_path, prediction_path, args):
        self.samples = []
        self.prediction_path = prediction_path
        self.args = args
        
        if 'clearpose' in self.args.dataset.lower():
            dataset = ClearPoseDataset(self.args.dataset)
        elif 'hammer' in self.args.dataset.lower():
            dataset = HAMMERDataset(self.args.dataset, self.args.raw_type)
        else:
            raise ValueError(f"Invalid dataset: {self.args.dataset}")
        # Default to args if dataset doesn't provide depth_range
        self.min_depth = getattr(dataset, 'depth_range', [args.min_depth, args.max_depth])[0]
        self.max_depth = getattr(dataset, 'depth_range', [args.min_depth, args.max_depth])[1]
        
        # Override args for passing into loading function later
        self.args.min_depth = self.min_depth
        self.args.max_depth = self.max_depth
        
        self.dataset = dataset
        self.align = getattr(args, 'align', False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        rgb_path = sample[0]
        gt_path = sample[2]
        
        # Consistent naming with infer_dataset_bs.py
        tmp = rgb_path.split('/')
        if len(tmp) >= 4:
            scene_name = tmp[-4]
            name = scene_name + '#' + tmp[-1].split('.')[0]
        else:
            name = Path(rgb_path).stem
            
        pred_file = join(self.prediction_path, f"{name}.npy")
        if not exists(pred_file):
            raise FileNotFoundError(f"Prediction not found: {pred_file}")
            
        pred = np.load(pred_file)
        depth_GT, valid_mask = load_gt_depth(gt_path, self.args.depth_scale, self.args.max_depth, self.args.min_depth)
        
        pred_invalid_mask = np.logical_or(np.isnan(pred), np.isinf(pred))
        if pred_invalid_mask.sum() > 0:
            valid_mask = valid_mask & ~pred_invalid_mask

        if self.align:
            depth_GT_reshaped = depth_GT[valid_mask].reshape((-1, 1))
            pred_reshaped = pred[valid_mask].reshape((-1, 1))

            _ones = np.ones_like(pred_reshaped)
            A = np.concatenate([pred_reshaped, _ones], axis=-1)
            X = np.linalg.lstsq(A, depth_GT_reshaped, rcond=None)[0]
            scale, shift = X 
            pred_reshaped = scale * pred_reshaped + shift
            pred_reshaped = np.clip(pred_reshaped, a_min=self.args.min_depth, a_max=None) 
            
            return {
                'name': name,
                'pred': pred_reshaped.astype(np.float32),
                'gt': depth_GT_reshaped.astype(np.float32),
                'mask': np.ones_like(pred_reshaped, dtype=bool),
                'is_aligned': True
            }
        else:
            return {
                'name': name,
                'pred': pred.astype(np.float32),
                'gt': depth_GT.astype(np.float32),
                'mask': valid_mask.astype(bool),
                'is_aligned': False
            }

def main():
    args = parse_arguments()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    with open(join(args.output, 'eval_args.json'), 'w') as f:
        json.dump(vars(args), f)
        
    eval_dataset = EvalDataset(args.dataset, args.output, args)
    if len(eval_dataset) == 0:
        logger.error("No samples found in dataset for evaluation.")
        return

    logger.info(f"Updated depth range from dataset: min={eval_dataset.min_depth}, max={eval_dataset.max_depth}")
        
    loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size if not getattr(args, 'align', False) else 1,  
        shuffle=False, 
        num_workers=args.num_workers, 
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    all_metrics = []
    
    # Evaluate
    for batch in tqdm(loader, desc="Evaluating Metrics"):
        names = batch['name']
        
        # Move to GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pred_depth_ts = batch['pred'].to(device)
        gt_depth_ts = batch['gt'].to(device)
        mask_ts = batch['mask'].to(device)
        
        # Compute metrics per sample
        l1 = mae_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        rmse = rmse_linear(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        abs_rel = abs_relative_difference(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        d4 = delta4_acc_105(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        d5 = delta5_acc110(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        d1 = delta1_acc(pred_depth_ts, gt_depth_ts, mask_ts, reduction='none')
        
        # To CPU
        l1_cpu = l1.detach().cpu().numpy()
        rmse_cpu = rmse.detach().cpu().numpy()
        abs_rel_cpu = abs_rel.detach().cpu().numpy()
        d4_cpu = d4.detach().cpu().numpy()
        d5_cpu = d5.detach().cpu().numpy()
        d1_cpu = d1.detach().cpu().numpy()
        
        batch_len = len(names)
        for i in range(batch_len):
            all_metrics.append({
                'name': names[i],
                'L1': l1_cpu[i],
                'rmse_linear': rmse_cpu[i],
                'abs_relative_difference': abs_rel_cpu[i],
                'delta4_acc_105': d4_cpu[i],
                'delta5_acc110': d5_cpu[i],
                'delta1_acc': d1_cpu[i],
            })
            
    df_metrics = pd.DataFrame(all_metrics)
    mean_metrics = df_metrics.mean(numeric_only=True).to_frame().T
    
    ALIGN = getattr(args, 'align', False)
    
    csv_out = join(args.output, f'all_metrics_{current_time}_{ALIGN}.csv')
    json_out = join(args.output, f'mean_metrics_{current_time}_{ALIGN}.json')
    
    df_metrics.to_csv(csv_out, index=False)
    mean_metrics.to_json(json_out, orient='records', lines=True, force_ascii=False)
    
    logger.info(f"Evaluation completed. Metrics saved to: {args.output}")
    print("="*50)
    print(mean_metrics.to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    main()
