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
