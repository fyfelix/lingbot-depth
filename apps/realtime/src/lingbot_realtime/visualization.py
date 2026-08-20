"""Depth visualization helpers shared by the web protocol and persistence layer.

The numeric depth arrays remain metric values and are never normalized in place.  The
visualization path follows the AS-Depth realtime convention: raw sensor depth uses a
stable configured range, while predicted depth uses a robust percentile range that is
insensitive to invalid pixels and a small number of outliers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DepthRange:
    min_m: float
    max_m: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.min_m) or not np.isfinite(self.max_m):
            raise ValueError("Depth visualization range must be finite")
        if self.max_m <= self.min_m:
            raise ValueError("Depth visualization max must be greater than min")

    def to_dict(self) -> dict[str, float]:
        return {"min_m": float(self.min_m), "max_m": float(self.max_m)}


@dataclass(frozen=True)
class DepthVisualizationConfig:
    """Display-only policy; it does not change inference or measurement depth."""

    min_depth_m: float = 0.1
    max_depth_m: float = 5.0
    valid_max_depth_m: float = 6.0
    pred_percentile_min: float = 1.0
    pred_percentile_max: float = 99.0
    colormap_name: str = "turbo"

    def __post_init__(self) -> None:
        if self.min_depth_m < 0 or self.max_depth_m <= self.min_depth_m:
            raise ValueError("Invalid depth visualization range")
        if self.valid_max_depth_m <= 0:
            raise ValueError("valid_max_depth_m must be positive")
        if not 0 <= self.pred_percentile_min < self.pred_percentile_max <= 100:
            raise ValueError("Invalid predicted depth percentiles")
        if self.colormap_name != "turbo":
            raise ValueError("Only the turbo colormap is currently supported")

    def raw_range(self) -> DepthRange:
        return DepthRange(self.min_depth_m, self.max_depth_m)

    def predicted_range(self, depth_m: np.ndarray) -> DepthRange:
        minimum, maximum = depth_percentile_range(
            depth_m,
            percentile_min=self.pred_percentile_min,
            percentile_max=self.pred_percentile_max,
            fallback_min=self.min_depth_m,
            fallback_max=self.max_depth_m,
        )
        return DepthRange(minimum, maximum)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def depth_percentile_range(
    depth_m: np.ndarray,
    *,
    percentile_min: float = 1.0,
    percentile_max: float = 99.0,
    fallback_min: float = 0.1,
    fallback_max: float = 5.0,
) -> tuple[float, float]:
    """Return a stable display range from valid positive metric depth values."""

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim > 2:
        depth = depth.squeeze()
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return float(fallback_min), float(max(fallback_max, fallback_min + 1e-6))

    pmin = float(np.clip(percentile_min, 0.0, 100.0))
    pmax = float(np.clip(percentile_max, 0.0, 100.0))
    if pmax <= pmin:
        raise ValueError("percentile_max must be greater than percentile_min")
    values = depth[valid]
    minimum, maximum = (float(value) for value in np.percentile(values, [pmin, pmax]))
    if maximum <= minimum:
        maximum = minimum + 1e-6
    return minimum, maximum


def colorize_depth_fast(
    depth_m: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    valid_max_m: float | None = None,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """Colorize metric depth as BGR, with invalid values rendered black.

    This mirrors the vectorized OpenCV path in ``AS-Depth/realtime/pipeline_d435.py``.
    ``depth_m`` is converted to a temporary float32 array and is never modified.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim > 2:
        depth = depth.squeeze()
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got shape {depth.shape}")
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError("vmax must be greater than vmin")

    normalized = (depth - np.float32(vmin)) * np.float32(255.0 / (vmax - vmin))
    np.clip(normalized, 0.0, 255.0, out=normalized)
    with np.errstate(invalid="ignore"):
        gray = normalized.astype(np.uint8)
    colored = cv2.applyColorMap(gray, colormap)

    valid = np.isfinite(depth) & (depth > 0)
    if valid_max_m is not None:
        valid &= depth <= float(valid_max_m)
    colored[~valid] = 0
    return np.ascontiguousarray(colored)


def colorize_depth_rgb(
    depth_m: np.ndarray,
    *,
    depth_range: DepthRange,
    valid_max_m: float | None = None,
) -> np.ndarray:
    """Return a browser/PIL-friendly RGB visualization."""

    bgr = colorize_depth_fast(
        depth_m,
        vmin=depth_range.min_m,
        vmax=depth_range.max_m,
        valid_max_m=valid_max_m,
    )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
