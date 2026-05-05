#!/usr/bin/env python3
"""Run LingBot-Depth iBims inference and save official *_results.mat files."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.io import savemat
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mdm.model.v2 import MDMModel  # noqa: E402


IBIMS_DEPTH_MAX_M = 50.0
IBIMS_DEPTH_SCALE = 65535.0 / IBIMS_DEPTH_MAX_M
SYNTHETIC_RAW_DIR_NAME = "ibims1_synthetic_raw_depth"
EXPECTED_SHAPE = (480, 640)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LingBot-Depth inference for iBims and write official MAT files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--manifest", required=True, help="iBims JSONL manifest path")
    parser.add_argument(
        "--model-path",
        default="ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt",
        help="Local LingBot-Depth checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Prediction directory; defaults to evaluation_ibims/output/ibims_<model>_<timestamp>/predictions/<level>",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Manifest batch size")
    parser.add_argument(
        "--device",
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
        "--max-depth",
        type=float,
        default=None,
        help="Depth clamp for raw input; defaults to manifest depth-range max",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=None,
        help="Raw depth scale; defaults to each manifest row depth_scale",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Use only the first N manifest rows for smoke testing",
    )
    return parser.parse_args()


def resolve_root(path: str | Path) -> Path:
    path_obj = Path(path).expanduser()
    if not path_obj.is_absolute():
        path_obj = Path.cwd() / path_obj
    return path_obj.resolve()


def resolve_path(base: Path, value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def select_device(device_name: str) -> torch.device:
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


def load_model(model_path: str | Path, device: torch.device) -> MDMModel:
    model_path = resolve_root(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    return MDMModel.from_pretrained(model_path).to(device).eval()


def load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(manifest_path, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("dataset") != "ibims":
                raise ValueError(f"{manifest_path}:{line_number} is not an iBims row")
            for key in ("sample_id", "rgb", "raw_depth"):
                if key not in row:
                    raise ValueError(f"{manifest_path}:{line_number} missing required key: {key}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def infer_difficulty(manifest_path: Path, rows: list[dict[str, Any]]) -> str:
    difficulty = rows[0].get("difficulty")
    if difficulty:
        return str(difficulty)
    stem = manifest_path.stem
    return stem[len("ibims_") :] if stem.startswith("ibims_") else stem


def default_output_dir(manifest_path: Path, rows: list[dict[str, Any]], model_path: Path) -> Path:
    difficulty = infer_difficulty(manifest_path, rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_stem = model_path.stem
    return (
        PROJECT_ROOT
        / "evaluation_ibims"
        / "output"
        / f"ibims_{model_stem}_{timestamp}"
        / "predictions"
        / difficulty
    )


def row_depth_scale(row: dict[str, Any], cli_depth_scale: float | None) -> float:
    if cli_depth_scale is not None:
        return cli_depth_scale
    return float(row.get("depth_scale", IBIMS_DEPTH_SCALE))


def row_max_depth(row: dict[str, Any], cli_max_depth: float | None) -> float:
    if cli_max_depth is not None:
        return cli_max_depth
    depth_range = row.get("depth-range", [0.01, IBIMS_DEPTH_MAX_M])
    return float(depth_range[1])


def read_single_channel(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    if image.ndim == 3:
        image = image[:, :, 0]
    return image


def load_rgb(rgb_path: Path, device: torch.device) -> tuple[np.ndarray, torch.Tensor]:
    image_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read RGB image: {rgb_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb.astype(np.float32) / 255.0)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    return image_rgb, image_tensor


def load_raw_depth(
    raw_depth_path: Path,
    depth_scale: float,
    max_depth: float,
    device: torch.device,
) -> tuple[np.ndarray, torch.Tensor]:
    depth = read_single_channel(raw_depth_path).astype(np.float32) / depth_scale
    valid = np.isfinite(depth) & (depth > 0.0) & (depth <= max_depth)
    depth = np.where(valid, depth, 0.0).astype(np.float32)
    depth_tensor = torch.from_numpy(depth).unsqueeze(0).to(device)
    return depth, depth_tensor


def normalize_prediction(pred_depth: Any, target_shape: tuple[int, int]) -> np.ndarray:
    pred = np.asarray(pred_depth, dtype=np.float32)
    if pred.ndim == 3 and pred.shape[0] == 1:
        pred = pred[0]
    if pred.ndim != 2:
        raise ValueError(f"Expected HxW prediction, got shape {pred.shape}")
    if pred.shape != target_shape:
        pred = cv2.resize(pred, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    pred = pred.astype(np.float32, copy=False)
    invalid = ~np.isfinite(pred) | (pred <= 0.0)
    pred[invalid] = np.nan
    return pred


def iter_batches(rows: list[dict[str, Any]], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


@torch.inference_mode()
def run_manifest_inference(
    manifest_path: str | Path,
    output_dir: str | Path,
    model: MDMModel,
    device: torch.device,
    *,
    batch_size: int = 1,
    resolution_level: int = 9,
    use_fp16: bool = False,
    apply_mask: bool = False,
    depth_scale: float | None = None,
    max_depth: float | None = None,
    max_samples: int | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be greater than 0")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be greater than 0")

    manifest_path = resolve_root(manifest_path)
    output_dir = resolve_root(output_dir)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    rows = load_manifest(manifest_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    difficulty = infer_difficulty(manifest_path, rows)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    cuda_fp16 = use_fp16 and device.type == "cuda"
    progress = tqdm(total=len(rows), desc=f"iBims {difficulty} inference")
    try:
        for batch_rows in iter_batches(rows, batch_size):
            for row in batch_rows:
                sample_id = str(row["sample_id"])
                rgb_path = resolve_path(manifest_path.parent, row["rgb"])
                raw_depth_path = resolve_path(manifest_path.parent, row["raw_depth"])

                rgb, image_tensor = load_rgb(rgb_path, device)
                raw_depth, depth_tensor = load_raw_depth(
                    raw_depth_path,
                    row_depth_scale(row, depth_scale),
                    row_max_depth(row, max_depth),
                    device,
                )
                if raw_depth.shape != rgb.shape[:2]:
                    raise ValueError(
                        f"RGB/depth shape mismatch for {sample_id}: "
                        f"rgb={rgb.shape[:2]}, depth={raw_depth.shape}"
                    )

                output = model.infer(
                    image_tensor,
                    depth_in=depth_tensor,
                    intrinsics=None,
                    resolution_level=resolution_level,
                    apply_mask=apply_mask,
                    use_fp16=cuda_fp16,
                )
                pred_depth = output["depth"].detach().cpu().numpy()
                pred_depth = normalize_prediction(pred_depth, raw_depth.shape)
                if pred_depth.shape != EXPECTED_SHAPE:
                    raise ValueError(
                        f"{sample_id}: expected prediction shape {EXPECTED_SHAPE}, got {pred_depth.shape}"
                    )

                savemat(
                    output_dir / f"{sample_id}_results.mat",
                    {"pred_depths": pred_depth.astype(np.float32, copy=False)},
                )
                written += 1
                progress.update(1)
    finally:
        progress.close()

    stats = {
        "difficulty": difficulty,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "num_predictions": written,
    }
    metadata = dict(run_metadata or {})
    metadata.update(stats)
    with open(output_dir / "infer_args.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False, sort_keys=True, default=str)

    return stats


def main() -> None:
    args = parse_args()
    manifest_path = resolve_root(args.manifest)
    model_path = resolve_root(args.model_path)
    rows = load_manifest(manifest_path)
    output_dir = (
        resolve_root(args.output_dir)
        if args.output_dir
        else default_output_dir(manifest_path, rows, model_path)
    )
    device = select_device(args.device)
    model = load_model(model_path, device)

    stats = run_manifest_inference(
        manifest_path,
        output_dir,
        model,
        device,
        batch_size=args.batch_size,
        resolution_level=args.resolution_level,
        use_fp16=args.use_fp16,
        apply_mask=args.apply_mask,
        depth_scale=args.depth_scale,
        max_depth=args.max_depth,
        max_samples=args.max_samples,
        run_metadata={
            **vars(args),
            "model_path": str(model_path),
            "device_resolved": str(device),
            "resolved_model_module": "mdm.model.v2",
            "resolved_model_class": "MDMModel",
            "output_kind": "metric_depth_meter",
            "alignment": "none",
        },
    )
    print(f"Wrote {stats['num_predictions']} official iBims predictions to: {output_dir}")


if __name__ == "__main__":
    main()
