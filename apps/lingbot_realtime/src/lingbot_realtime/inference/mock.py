from __future__ import annotations

import time

import cv2
import numpy as np

from lingbot_realtime.domain import InferenceResult, RGBDFrame
from lingbot_realtime.inference.preprocessing import sanitize_metric_depth


def depth_to_points(depth_m: np.ndarray, frame: RGBDFrame) -> np.ndarray:
    height, width = depth_m.shape
    intr = frame.intrinsics
    sx = width / max(1, intr.width)
    sy = height / max(1, intr.height)
    fx = intr.fx * sx
    fy = intr.fy * sy
    ppx = intr.ppx * sx
    ppy = intr.ppy * sy
    yy, xx = np.mgrid[0:height, 0:width]
    z = depth_m.astype(np.float32, copy=False)
    x = ((xx.astype(np.float32) + 0.5 - ppx) / fx) * z
    y = ((yy.astype(np.float32) + 0.5 - ppy) / fy) * z
    points = np.stack([x, y, z], axis=-1).astype(np.float32)
    invalid = ~np.isfinite(z) | (z <= 0)
    points[invalid] = np.inf
    return points


class MockInferenceEngine:
    """Fast deterministic depth completion used by macOS and CI tests."""

    def __init__(self, max_depth_m: float = 6.0) -> None:
        self.max_depth_m = float(max_depth_m)

    @property
    def name(self) -> str:
        return "mock"

    @property
    def device_name(self) -> str:
        return "cpu"

    def load(self) -> None:
        return None

    def infer(self, frame: RGBDFrame) -> InferenceResult:
        started = time.perf_counter()
        raw = sanitize_metric_depth(frame.depth_m, self.max_depth_m)
        valid = np.isfinite(raw) & (raw > 0)
        if valid.any():
            filled = np.where(valid, raw, float(np.median(raw[valid]))).astype(np.float32)
        else:
            filled = np.ones_like(raw, dtype=np.float32)
        smooth = cv2.bilateralFilter(filled, 7, 0.05, 5.0)
        pred = np.where(np.isfinite(smooth) & (smooth > 0), smooth, 0.0).astype(np.float32)
        points = depth_to_points(pred, frame)
        return InferenceResult(
            pred_depth_m=np.ascontiguousarray(pred),
            points=np.ascontiguousarray(points),
            elapsed_sec=time.perf_counter() - started,
        )

    def close(self) -> None:
        return None
