from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Phase(str, Enum):
    STARTING = "starting"
    PREVIEW = "preview"
    INFERENCING = "inferencing"
    READY = "ready"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float

    def normalized_matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx / self.width, 0.0, self.ppx / self.width],
                [0.0, self.fy / self.height, self.ppy / self.height],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RGBDFrame:
    frame_id: int
    timestamp: float
    color_rgb: np.ndarray
    raw_depth_u16: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    depth_scale_m: float

    def __post_init__(self) -> None:
        color = np.array(self.color_rgb, dtype=np.uint8, copy=True, order="C")
        raw = np.array(self.raw_depth_u16, dtype=np.uint16, copy=True, order="C")
        depth = np.array(self.depth_m, dtype=np.float32, copy=True, order="C")
        if color.ndim != 3 or color.shape[2] != 3:
            raise ValueError(f"color_rgb must be HxWx3, got {color.shape}")
        if raw.shape != color.shape[:2] or depth.shape != raw.shape:
            raise ValueError(
                "RGB-D arrays must be aligned: "
                f"color={color.shape}, raw={raw.shape}, depth={depth.shape}"
            )
        if self.frame_id < 0 or self.timestamp < 0 or self.depth_scale_m <= 0:
            raise ValueError("frame id/timestamp must be non-negative and depth scale positive")
        color.setflags(write=False)
        raw.setflags(write=False)
        depth.setflags(write=False)
        object.__setattr__(self, "color_rgb", color)
        object.__setattr__(self, "raw_depth_u16", raw)
        object.__setattr__(self, "depth_m", depth)

    def frozen_copy(self) -> "RGBDFrame":
        return RGBDFrame(
            frame_id=self.frame_id,
            timestamp=self.timestamp,
            color_rgb=np.array(self.color_rgb, dtype=np.uint8, copy=True, order="C"),
            raw_depth_u16=np.array(self.raw_depth_u16, dtype=np.uint16, copy=True, order="C"),
            depth_m=np.array(self.depth_m, dtype=np.float32, copy=True, order="C"),
            intrinsics=self.intrinsics,
            depth_scale_m=self.depth_scale_m,
        )


@dataclass(frozen=True)
class InferenceResult:
    pred_depth_m: np.ndarray
    points: np.ndarray
    elapsed_sec: float

    def __post_init__(self) -> None:
        depth = np.array(self.pred_depth_m, dtype=np.float32, copy=True, order="C")
        points = np.array(self.points, dtype=np.float32, copy=True, order="C")
        if depth.ndim != 2 or points.shape != (*depth.shape, 3):
            raise ValueError(
                f"Inference arrays must be HxW and HxWx3, got {depth.shape}, {points.shape}"
            )
        depth[~np.isfinite(depth) | (depth < 0)] = 0.0
        depth.setflags(write=False)
        points.setflags(write=False)
        object.__setattr__(self, "pred_depth_m", depth)
        object.__setattr__(self, "points", points)

    def frozen_copy(self) -> "InferenceResult":
        return InferenceResult(
            pred_depth_m=self.pred_depth_m,
            points=self.points,
            elapsed_sec=self.elapsed_sec,
        )


@dataclass(frozen=True)
class PredictionPacket:
    frame: RGBDFrame
    result: InferenceResult | None
    sequence: int
    timings_ms: dict[str, float] = field(default_factory=dict)

    def frozen_copy(self) -> "PredictionPacket":
        return PredictionPacket(
            frame=self.frame.frozen_copy(),
            result=self.result.frozen_copy() if self.result is not None else None,
            sequence=self.sequence,
            timings_ms=dict(self.timings_ms),
        )


@dataclass(frozen=True)
class SampledPoint:
    requested_x: int
    requested_y: int
    sampled_x: int
    sampled_y: int
    radius: int
    depth_m: float
    xyz_m: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Measurement:
    measurement_id: int
    start: SampledPoint
    end: SampledPoint
    distance_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "measurement_id": self.measurement_id,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "distance_m": self.distance_m,
        }


@dataclass
class CaptureRecord:
    capture_id: str
    frame: RGBDFrame
    status: str = "inferencing"
    result: InferenceResult | None = None
    error: str | None = None
    measurements: list[Measurement] = field(default_factory=list)
    next_measurement_id: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "status": self.status,
            "error": self.error,
            "frame_id": self.frame.frame_id,
            "timestamp": self.frame.timestamp,
            "width": self.frame.intrinsics.width,
            "height": self.frame.intrinsics.height,
            "intrinsics": self.frame.intrinsics.to_dict(),
            "depth_scale_m": self.frame.depth_scale_m,
            "inference_elapsed_sec": (self.result.elapsed_sec if self.result is not None else None),
            "measurements": [item.to_dict() for item in self.measurements],
        }
