from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np


def _freeze_mapping(value: dict[str, Any] | MappingProxyType[str, Any] | None) -> MappingProxyType:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class FramePacket:
    """相机 worker 输出的不可变 BGR/uint16 毫米帧。"""

    frame_id: int
    timestamp_ns: int
    color_bgr: np.ndarray
    raw_depth_mm: np.ndarray
    metadata: MappingProxyType[str, Any] = field(default_factory=lambda: _freeze_mapping(None))

    def __post_init__(self) -> None:
        color = np.array(self.color_bgr, dtype=np.uint8, copy=True, order="C")
        raw = np.array(self.raw_depth_mm, dtype=np.uint16, copy=True, order="C")
        if self.frame_id < 0 or self.timestamp_ns < 0:
            raise ValueError("frame id and timestamp cannot be negative")
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError(f"color_bgr must be HxWx3, got {color.shape}")
        if raw.ndim != 2 or raw.shape != color.shape[:2]:
            raise ValueError(f"raw_depth_mm must align with color: {color.shape}, {raw.shape}")
        color.setflags(write=False)
        raw.setflags(write=False)
        object.__setattr__(self, "color_bgr", color)
        object.__setattr__(self, "raw_depth_mm", raw)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class PredictionPacket:
    frame: FramePacket
    depth_m: np.ndarray
    timings_ms: MappingProxyType[str, float] = field(default_factory=lambda: MappingProxyType({}))
    metadata: MappingProxyType[str, Any] = field(default_factory=lambda: _freeze_mapping(None))

    def __post_init__(self) -> None:
        depth = np.array(self.depth_m, dtype=np.float32, copy=True, order="C")
        if depth.ndim != 2 or depth.shape != self.frame.raw_depth_mm.shape:
            raise ValueError(f"prediction depth must align with frame, got {depth.shape}")
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        depth.setflags(write=False)
        object.__setattr__(self, "depth_m", depth)
        timings = {str(k): float(v) for k, v in self.timings_ms.items()}
        if any(v < 0 for v in timings.values()):
            raise ValueError("timings cannot be negative")
        object.__setattr__(self, "timings_ms", MappingProxyType(timings))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


class CameraSource(Protocol):
    def start(self) -> None: ...
    def read(self) -> FramePacket | None: ...
    def close(self) -> None: ...


class PredictionConsumer(Protocol):
    def publish(self, packet: PredictionPacket) -> None: ...
    def close(self) -> None: ...


class RealtimePreprocessor(Protocol):
    def prepare(self, frame: FramePacket) -> Any: ...


class RealtimePostprocessor(Protocol):
    def process(self, values: Any, frame: FramePacket) -> np.ndarray: ...
