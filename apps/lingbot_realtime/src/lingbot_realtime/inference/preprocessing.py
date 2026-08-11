"""Shared metric-depth preprocessing for inference adapters."""

from __future__ import annotations

import numpy as np


def sanitize_metric_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    """Return contiguous float32 metric depth with invalid samples set to zero.

    The camera frame remains untouched so raw visualization and persistence retain the
    original sensor values. This matches the D435 preprocessing contract used by the
    AS-Depth realtime pipeline.
    """

    if max_depth_m <= 0:
        raise ValueError("max_depth_m must be positive")
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected HxW depth, got shape {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0) & (depth <= float(max_depth_m))
    return np.ascontiguousarray(np.where(valid, depth, 0.0), dtype=np.float32)
