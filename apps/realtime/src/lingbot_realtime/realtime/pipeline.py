from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from .packets import (
    CameraSource,
    FramePacket,
    PredictionConsumer,
    PredictionPacket,
    RealtimePostprocessor,
    RealtimePreprocessor,
)


def _numpy(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("depth", next(iter(value.values())))
    if isinstance(value, (tuple, list)):
        value = value[0]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True, slots=True)
class MetricDepthPostprocessor:
    restore_camera_shape: bool = True

    def process(self, values: Any, frame: FramePacket) -> np.ndarray:
        depth = np.asarray(_numpy(values), dtype=np.float32)
        if depth.ndim == 4 and depth.shape[:2] == (1, 1):
            depth = depth[0, 0]
        elif depth.ndim == 3 and depth.shape[0] == 1:
            depth = depth[0]
        if depth.ndim != 2:
            raise ValueError(f"runtime output must reduce to HxW, got {depth.shape}")
        if self.restore_camera_shape and depth.shape != frame.raw_depth_mm.shape:
            depth = cv2.resize(
                depth,
                (frame.raw_depth_mm.shape[1], frame.raw_depth_mm.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        result = np.ascontiguousarray(depth, dtype=np.float32)
        result[~np.isfinite(result) | (result < 0)] = 0.0
        return result


class RealtimePipeline:
    def __init__(
        self,
        camera: CameraSource,
        runtime: Any,
        preprocessor: RealtimePreprocessor,
        *,
        postprocessor: RealtimePostprocessor | None = None,
        consumers: Iterable[PredictionConsumer] = (),
    ) -> None:
        self.camera = camera
        self.runtime = runtime
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor or MetricDepthPostprocessor()
        self.consumers = tuple(consumers)
        self._started = False

    def start(self) -> None:
        if not self._started:
            self.camera.start()
            self._started = True

    def process(self, frame: FramePacket) -> PredictionPacket:
        started = time.perf_counter()
        inputs = self.preprocessor.prepare(frame)
        preprocess_ms = (time.perf_counter() - started) * 1000.0
        output = self.runtime.infer(inputs)
        values = getattr(output, "values", output)
        timings = dict(getattr(output, "timings_ms", {}))
        post_started = time.perf_counter()
        depth = self.postprocessor.process(values, frame)
        timings.update(
            {
                "preprocess": preprocess_ms,
                "postprocess": (time.perf_counter() - post_started) * 1000.0,
            }
        )
        packet = PredictionPacket(frame, depth, MappingProxyType(timings))
        for consumer in self.consumers:
            consumer.publish(packet)
        return packet

    def run(self, *, max_frames: int | None = None) -> Iterator[PredictionPacket]:
        if max_frames is not None and max_frames < 0:
            raise ValueError("max_frames cannot be negative")
        self.start()
        count = 0
        while max_frames is None or count < max_frames:
            frame = self.camera.read()
            if frame is None:
                break
            yield self.process(frame)
            count += 1

    def close(self) -> None:
        for consumer in reversed(self.consumers):
            consumer.close()
        close = getattr(self.runtime, "close", None)
        if callable(close):
            close()
        self.camera.close()
        self._started = False

    def __enter__(self) -> "RealtimePipeline":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
