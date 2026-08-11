from __future__ import annotations

from typing import Any

import numpy as np

from lingbot_realtime.domain import CaptureRecord, RGBDFrame
from lingbot_realtime.visualization import DepthVisualizationConfig

PROTOCOL = "lingbot.realtime.snapshot.v1"


def _append_array(
    parts: list[bytes],
    layout: dict[str, Any],
    name: str,
    array: np.ndarray,
) -> None:
    contiguous = np.ascontiguousarray(array)
    current = sum(len(part) for part in parts)
    alignment = max(1, contiguous.dtype.itemsize)
    padding = (-current) % alignment
    if padding:
        parts.append(b"\x00" * padding)
        current += padding
    raw = contiguous.tobytes(order="C")
    parts.append(raw)
    layout[name] = {
        "offset": current,
        "bytes": len(raw),
        "shape": list(contiguous.shape),
        "dtype": contiguous.dtype.name,
    }


def _add_visualization_spec(
    layout: dict[str, Any],
    name: str,
    depth_range: dict[str, float],
    *,
    valid_max_m: float,
    mode: str,
    percentiles: tuple[float, float] | None = None,
) -> None:
    spec = layout[name]
    spec["unit"] = "meter" if spec["dtype"] == "float32" else "device_unit"
    spec["visualization"] = {
        **depth_range,
        "valid_max_m": float(valid_max_m),
        "colormap": "turbo",
        "range_mode": mode,
    }
    if percentiles is not None:
        spec["visualization"]["percentile"] = {
            "min": float(percentiles[0]),
            "max": float(percentiles[1]),
        }


def pack_preview(
    frame: RGBDFrame,
    revision: int,
    phase: str,
    *,
    depth_viz: DepthVisualizationConfig | None = None,
) -> tuple[dict[str, Any], bytes]:
    depth_viz = depth_viz or DepthVisualizationConfig()
    parts: list[bytes] = []
    layout: dict[str, Any] = {}
    _append_array(parts, layout, "color", frame.color_rgb.astype(np.uint8, copy=False))
    _append_array(parts, layout, "raw_depth", frame.raw_depth_u16.astype(np.uint16, copy=False))
    _add_visualization_spec(
        layout,
        "raw_depth",
        depth_viz.raw_range().to_dict(),
        valid_max_m=depth_viz.valid_max_depth_m,
        mode="fixed",
    )
    layout["raw_depth"]["scale_m"] = float(frame.depth_scale_m)
    payload = b"".join(parts)
    header = {
        "type": "preview_frame",
        "protocol": PROTOCOL,
        "revision": revision,
        "phase": phase,
        "frame_id": frame.frame_id,
        "timestamp": frame.timestamp,
        "intrinsics": frame.intrinsics.to_dict(),
        "depth_scale_m": frame.depth_scale_m,
        "payload_bytes": len(payload),
        **layout,
    }
    return header, payload


def pack_capture(
    record: CaptureRecord,
    revision: int,
    phase: str,
    *,
    depth_viz: DepthVisualizationConfig | None = None,
) -> tuple[dict[str, Any], bytes]:
    if record.result is None:
        raise ValueError("Capture result is not ready")
    depth_viz = depth_viz or DepthVisualizationConfig()
    parts: list[bytes] = []
    layout: dict[str, Any] = {}
    _append_array(parts, layout, "color", record.frame.color_rgb.astype(np.uint8, copy=False))
    _append_array(
        parts,
        layout,
        "raw_depth",
        record.frame.raw_depth_u16.astype(np.uint16, copy=False),
    )
    _append_array(
        parts,
        layout,
        "pred_depth",
        record.result.pred_depth_m.astype(np.float32, copy=False),
    )
    _add_visualization_spec(
        layout,
        "raw_depth",
        depth_viz.raw_range().to_dict(),
        valid_max_m=depth_viz.valid_max_depth_m,
        mode="fixed",
    )
    layout["raw_depth"]["scale_m"] = float(record.frame.depth_scale_m)
    _add_visualization_spec(
        layout,
        "pred_depth",
        depth_viz.predicted_range(record.result.pred_depth_m).to_dict(),
        valid_max_m=depth_viz.valid_max_depth_m,
        mode="percentile",
        percentiles=(depth_viz.pred_percentile_min, depth_viz.pred_percentile_max),
    )
    payload = b"".join(parts)
    header = {
        "type": "capture_result",
        "protocol": PROTOCOL,
        "revision": revision,
        "phase": phase,
        "capture_id": record.capture_id,
        "frame_id": record.frame.frame_id,
        "timestamp": record.frame.timestamp,
        "intrinsics": record.frame.intrinsics.to_dict(),
        "depth_scale_m": record.frame.depth_scale_m,
        "inference_elapsed_sec": record.result.elapsed_sec,
        "measurements": [item.to_dict() for item in record.measurements],
        "payload_bytes": len(payload),
        **layout,
    }
    return header, payload
