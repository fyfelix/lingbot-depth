from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from evaluation.core.io import (
    normalize_asdepth_prediction,
    normalize_prediction,
    read_gt_depth,
    read_raw_depth,
    read_rgb,
)
from evaluation.core.output import RunLayout, save_prediction
from evaluation.core.types import EvaluationSample, LoadedSample, RunConfig
from evaluation.core.visualization import save_kitti_visualizations, save_visualization
from evaluation.datasets.base import DatasetCollection
from mdm.model.v2 import MDMModel

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - the evaluation extra installs tqdm

    def tqdm(iterable, **_kwargs):
        return iterable


class InferenceInputDataset(Dataset):
    def __init__(self, samples: Sequence[EvaluationSample], load_gt: bool):
        self.samples = list(samples)
        self.load_gt = load_gt

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> LoadedSample:
        sample = self.samples[index]
        rgb = read_rgb(sample.rgb_path)
        raw_depth = read_raw_depth(
            sample.raw_depth_path,
            sample.depth_scale,
            sample.min_depth,
            sample.raw_max_depth if sample.raw_max_depth is not None else sample.max_depth,
        )
        gt_depth = None
        if self.load_gt and sample.gt_depth_path is not None:
            gt_depth = read_gt_depth(
                sample.gt_depth_path,
                sample.depth_scale,
                sample.min_depth,
                sample.max_depth,
            )
        return LoadedSample(sample=sample, rgb=rgb, raw_depth=raw_depth, gt_depth=gt_depth)


def collate_loaded_samples(items: List[LoadedSample]) -> List[LoadedSample]:
    return items


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    return torch.device(name)


def load_model(model_path: str, device: torch.device) -> MDMModel:
    return MDMModel.from_pretrained(model_path).to(device).eval()


@torch.inference_mode()
def run_inference(
    collection: DatasetCollection,
    config: RunConfig,
    layout: RunLayout,
) -> Dict[str, object]:
    if not config.model_path:
        raise ValueError("--model-path is required for inference")
    if config.batch_size < 1:
        raise ValueError("--batch-size must be greater than zero")
    if config.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 0 <= config.resolution_level <= 9:
        raise ValueError("--resolution-level must be between 0 and 9")

    device = select_device(config.device)
    model = load_model(config.model_path, device)
    if collection.name == "kitti" and config.save_visualizations:
        for sample in collection.samples:
            intrinsics_path = sample.metadata.get("intrinsics_path") or config.intrinsics_path
            if not intrinsics_path:
                raise ValueError(
                    "KITTI visualization requires an intrinsics path in every manifest row "
                    "or --intrinsics-path"
                )
    dataset = InferenceInputDataset(
        collection.samples,
        load_gt=config.save_visualizations,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_loaded_samples,
    )

    written = 0
    use_fp16 = config.use_fp16 and device.type == "cuda"
    for batch in tqdm(loader, desc=f"{collection.name} inference"):
        for item in batch:
            sample = item.sample
            if item.raw_depth.shape != item.rgb.shape[:2]:
                raise ValueError(
                    f"RGB/raw-depth shape mismatch for {sample.sample_id}: "
                    f"rgb={item.rgb.shape[:2]}, raw={item.raw_depth.shape}"
                )
            if sample.expected_shape is not None and item.raw_depth.shape != sample.expected_shape:
                raise ValueError(
                    f"Unexpected input shape for {sample.sample_id}: "
                    f"got {item.raw_depth.shape}, expected {sample.expected_shape}"
                )

            image_tensor = (
                torch.from_numpy(item.rgb.astype(np.float32) / 255.0)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )
            depth_tensor = torch.from_numpy(item.raw_depth).unsqueeze(0).to(device)
            output = model.infer(
                image_tensor,
                depth_in=depth_tensor,
                intrinsics=None,
                resolution_level=config.resolution_level,
                apply_mask=config.apply_mask,
                use_fp16=use_fp16,
            )
            if "depth" not in output:
                raise ValueError("MDMModel.infer did not return a depth prediction")

            prediction_array = output["depth"].detach().cpu().numpy()
            if collection.name == "kitti":
                prediction = normalize_asdepth_prediction(prediction_array, item.raw_depth.shape)
            else:
                prediction = normalize_prediction(prediction_array, item.raw_depth.shape)
            if sample.expected_shape is not None and prediction.shape != sample.expected_shape:
                raise ValueError(
                    f"Unexpected prediction shape for {sample.sample_id}: "
                    f"got {prediction.shape}, expected {sample.expected_shape}"
                )

            save_prediction(layout.prediction_path(sample), prediction)
            if config.save_visualizations:
                save_visualization(
                    layout.visualization_path(sample),
                    item.rgb,
                    item.raw_depth,
                    prediction,
                    item.gt_depth,
                    config.visualization_min_depth,
                    config.visualization_max_depth,
                )
                if collection.name == "kitti":
                    intrinsics_value = sample.metadata.get("intrinsics_path")
                    intrinsics_path = (
                        Path(intrinsics_value)
                        if intrinsics_value
                        else config.intrinsics_path
                    )
                    assert intrinsics_path is not None
                    save_kitti_visualizations(
                        layout.kitti_prediction_visualization_path(sample),
                        layout.kitti_pointcloud_visualization_path(sample),
                        item.rgb,
                        prediction,
                        intrinsics_path,
                        config.visualization_min_depth,
                        config.visualization_max_depth,
                        config.pointcloud_rot_x_deg,
                        config.pointcloud_rot_y_deg,
                        config.pointcloud_knn_k,
                        config.pointcloud_knn_std_ratio,
                        config.disable_pointcloud_knn_filter,
                    )
            written += 1

    return {
        "num_predictions": written,
        "device": str(device),
        "use_fp16": use_fp16,
        "model_class": "mdm.model.v2.MDMModel",
    }
