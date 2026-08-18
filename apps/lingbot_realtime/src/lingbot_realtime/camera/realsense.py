from __future__ import annotations

import time
from contextlib import suppress
from typing import Any

import cv2
import numpy as np

from lingbot_realtime.domain import CameraIntrinsics, RGBDFrame


class RealSenseFrameSource:
    """RealSense RGB-D source with depth aligned to the color stream."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._rs: Any = None
        self._pipeline: Any = None
        self._align: Any = None
        self._intrinsics: CameraIntrinsics | None = None
        self._depth_scale_m = 0.001
        self._frame_id = 0

    @property
    def name(self) -> str:
        return "realsense"

    def _import_rs(self) -> Any:
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is required for --source=realsense. "
                "Use --source=fixture for macOS/CI testing without hardware."
            ) from exc
        return rs

    def start(self) -> None:
        if self._pipeline is not None:
            return
        rs = self._import_rs()
        errors: list[str] = []
        profile = None
        pipeline = None
        selected = None
        candidates = [(self.width, self.height, self.fps)]
        for fallback_fps in (30, 15, 6):
            candidate = (self.width, self.height, fallback_fps)
            if candidate not in candidates:
                candidates.append(candidate)

        for width, height, fps in candidates:
            candidate_pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            try:
                profile = candidate_pipeline.start(config)
                pipeline = candidate_pipeline
                selected = (width, height, fps)
                break
            except RuntimeError as exc:
                with suppress(Exception):
                    candidate_pipeline.stop()
                errors.append(f"{width}x{height}@{fps}: {exc}")

        if pipeline is None or profile is None or selected is None:
            raise RuntimeError("Unable to start RealSense. Tried: " + "; ".join(errors))

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale_m = float(depth_sensor.get_depth_scale())
        self._intrinsics = CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            ppx=float(intr.ppx),
            ppy=float(intr.ppy),
        )
        self.width, self.height, self.fps = selected
        self._rs = rs
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)

    def read(self, timeout_sec: float = 5.0) -> RGBDFrame:
        if self._pipeline is None or self._align is None or self._intrinsics is None:
            raise RuntimeError("RealSense source is not running")
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=max(1, int(timeout_sec * 1000)))
            aligned = self._align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RuntimeError("Aligned RealSense frame is incomplete")
            color_bgr = np.asanyarray(color_frame.get_data()).astype(np.uint8, copy=True)
            color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
            raw_depth_u16 = np.asanyarray(depth_frame.get_data()).astype(np.uint16, copy=True)
        except RuntimeError as exc:
            raise RuntimeError(f"RealSense frame grab failed: {exc}") from exc

        self._frame_id += 1
        depth_m = raw_depth_u16.astype(np.float32) * self._depth_scale_m
        return RGBDFrame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            color_rgb=np.ascontiguousarray(color_rgb),
            raw_depth_u16=np.ascontiguousarray(raw_depth_u16),
            depth_m=np.ascontiguousarray(depth_m),
            intrinsics=self._intrinsics,
            depth_scale_m=self._depth_scale_m,
        )

    def stop(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._align = None
        self._intrinsics = None
        if pipeline is not None:
            pipeline.stop()
