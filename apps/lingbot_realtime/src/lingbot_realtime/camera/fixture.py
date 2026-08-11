from __future__ import annotations

import time

import numpy as np

from lingbot_realtime.domain import CameraIntrinsics, RGBDFrame


class FixtureFrameSource:
    """Deterministic animated RGB-D source for macOS and CI testing."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._running = False
        self._frame_id = 0
        self._next_frame_at = 0.0
        self._intrinsics = CameraIntrinsics(
            width=self.width,
            height=self.height,
            fx=self.width * 0.72,
            fy=self.width * 0.72,
            ppx=(self.width - 1) / 2.0,
            ppy=(self.height - 1) / 2.0,
        )
        yy, xx = np.mgrid[0 : self.height, 0 : self.width]
        self._xx = xx.astype(np.float32)
        self._yy = yy.astype(np.float32)

    @property
    def name(self) -> str:
        return "fixture"

    def start(self) -> None:
        self._running = True
        self._next_frame_at = time.monotonic()

    def read(self, timeout_sec: float = 5.0) -> RGBDFrame:
        if not self._running:
            raise RuntimeError("Fixture source is not running")
        now = time.monotonic()
        delay = self._next_frame_at - now
        if delay > 0:
            if delay > timeout_sec:
                raise TimeoutError("Fixture frame timeout")
            time.sleep(delay)
        self._next_frame_at = max(self._next_frame_at + 1.0 / self.fps, time.monotonic())
        self._frame_id += 1

        phase = self._frame_id * 0.035
        nx = self._xx / max(1.0, self.width - 1)
        ny = self._yy / max(1.0, self.height - 1)
        wave = 0.12 * np.sin(nx * 8.0 + phase) * np.cos(ny * 6.0 - phase)
        depth_m = 0.75 + 1.35 * nx + 0.35 * ny + wave

        cx = self.width * (0.5 + 0.18 * np.sin(phase * 0.7))
        cy = self.height * (0.5 + 0.12 * np.cos(phase * 0.5))
        radius = min(self.width, self.height) * 0.13
        object_mask = (self._xx - cx) ** 2 + (self._yy - cy) ** 2 < radius**2
        depth_m = np.where(object_mask, depth_m - 0.32, depth_m).astype(np.float32)

        red = np.clip(nx * 255.0, 0, 255)
        green = np.clip(ny * 255.0, 0, 255)
        blue = np.clip((1.0 - nx) * 170.0 + object_mask * 85.0, 0, 255)
        color_rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)

        raw_depth_u16 = np.rint(depth_m * 1000.0).astype(np.uint16)
        hole_mask = ((self._xx.astype(np.int32) + self._yy.astype(np.int32)) % 97) == 0
        raw_depth_u16[hole_mask] = 0
        raw_depth_m = raw_depth_u16.astype(np.float32) * 0.001

        return RGBDFrame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            color_rgb=np.ascontiguousarray(color_rgb),
            raw_depth_u16=np.ascontiguousarray(raw_depth_u16),
            depth_m=np.ascontiguousarray(raw_depth_m),
            intrinsics=self._intrinsics,
            depth_scale_m=0.001,
        )

    def stop(self) -> None:
        self._running = False
