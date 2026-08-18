from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .packets import FramePacket


@dataclass(frozen=True, slots=True)
class Resolution:
    height: int
    width: int

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 4, self.height, self.width)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (1, self.height, self.width)


@dataclass(frozen=True, slots=True)
class D435HostPreprocessor:
    """BGR/raw millimeter -> float input [RGB 0..1, metric depth meters]."""

    resolution: Resolution
    depth_scale: float = 1000.0
    max_depth_m: float = 10.0

    def __post_init__(self) -> None:
        if self.resolution.height <= 0 or self.resolution.width <= 0:
            raise ValueError("resolution must be positive")
        if self.depth_scale <= 0 or self.max_depth_m <= 0:
            raise ValueError("depth_scale and max_depth_m must be positive")

    def raw_depth_m(self, frame: FramePacket) -> np.ndarray:
        values = frame.raw_depth_mm.astype(np.float32) / np.float32(self.depth_scale)
        invalid = (frame.raw_depth_mm == 0) | ~np.isfinite(values) | (values >= self.max_depth_m)
        np.clip(values, 0.0, self.max_depth_m, out=values)
        values[invalid] = 0.0
        return np.ascontiguousarray(values, dtype=np.float32)

    def prepare(self, frame: FramePacket) -> np.ndarray:
        rgb = cv2.cvtColor(frame.color_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = cv2.resize(
            rgb,
            (self.resolution.width, self.resolution.height),
            interpolation=cv2.INTER_LINEAR,
        )
        raw = cv2.resize(
            self.raw_depth_m(frame),
            (self.resolution.width, self.resolution.height),
            interpolation=cv2.INTER_NEAREST,
        )
        return np.ascontiguousarray(
            np.concatenate((rgb.transpose(2, 0, 1), raw[None]), axis=0)[None],
            dtype=np.float32,
        )
