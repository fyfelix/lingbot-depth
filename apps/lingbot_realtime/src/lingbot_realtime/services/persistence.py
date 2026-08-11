from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from lingbot_realtime.domain import CaptureRecord


def _colorize_depth(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0) & (depth_m <= max_depth_m)
    scaled = np.clip(depth_m / max_depth_m, 0.0, 1.0)
    image = cv2.applyColorMap((scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


class PersistenceService:
    def __init__(self, enabled: bool, output_root: Path, max_depth_m: float) -> None:
        self.enabled = bool(enabled)
        self.output_root = Path(output_root)
        self.max_depth_m = float(max_depth_m)

    def capture_dir(self, capture_id: str) -> Path:
        return self.output_root / capture_id

    def save_capture(self, record: CaptureRecord, engine_name: str, device_name: str) -> None:
        if not self.enabled or record.result is None:
            return
        root = self.capture_dir(record.capture_id)
        root.mkdir(parents=True, exist_ok=True)
        frame = record.frame
        result = record.result
        cv2.imwrite(str(root / "rgb.png"), cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR))
        np.save(root / "raw_depth.npy", frame.depth_m.astype(np.float32), allow_pickle=False)
        np.save(root / "pred_depth.npy", result.pred_depth_m.astype(np.float32), allow_pickle=False)
        cv2.imwrite(
            str(root / "raw_depth_vis.png"),
            _colorize_depth(frame.depth_m, self.max_depth_m),
        )
        cv2.imwrite(
            str(root / "pred_depth_vis.png"),
            _colorize_depth(result.pred_depth_m, self.max_depth_m),
        )
        valid = (
            np.isfinite(result.points).all(axis=-1)
            & np.isfinite(result.pred_depth_m)
            & (result.pred_depth_m > 0)
        )
        if valid.any():
            cloud = trimesh.PointCloud(result.points[valid], frame.color_rgb[valid])
            cloud.export(root / "point_cloud.ply")
        self._write_json(root / "intrinsics.json", frame.intrinsics.to_dict())
        self._write_json(
            root / "meta.json",
            {
                **record.to_dict(),
                "engine": engine_name,
                "device": device_name,
                "depth_unit": "meter",
            },
        )
        self.save_measurements(record)

    def save_measurements(self, record: CaptureRecord) -> None:
        if not self.enabled:
            return
        root = self.capture_dir(record.capture_id)
        if not root.is_dir():
            return
        self._write_json(
            root / "measurements.json",
            {
                "capture_id": record.capture_id,
                "measurements": [m.to_dict() for m in record.measurements],
            },
        )

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)
