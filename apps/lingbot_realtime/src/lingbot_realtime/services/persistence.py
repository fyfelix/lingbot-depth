from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import trimesh

from lingbot_realtime.domain import CaptureRecord
from lingbot_realtime.visualization import DepthVisualizationConfig, colorize_depth_fast


class PersistenceService:
    def __init__(
        self,
        enabled: bool,
        output_root: Path,
        max_depth_m: float,
        depth_viz: DepthVisualizationConfig | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.output_root = Path(output_root)
        self.max_depth_m = float(max_depth_m)
        self.depth_viz = depth_viz or DepthVisualizationConfig(valid_max_depth_m=self.max_depth_m)

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
        raw_range = self.depth_viz.raw_range()
        pred_range = self.depth_viz.predicted_range(result.pred_depth_m)
        cv2.imwrite(
            str(root / "raw_depth_vis.png"),
            colorize_depth_fast(
                frame.depth_m,
                vmin=raw_range.min_m,
                vmax=raw_range.max_m,
                valid_max_m=self.depth_viz.valid_max_depth_m,
            ),
        )
        cv2.imwrite(
            str(root / "pred_depth_vis.png"),
            colorize_depth_fast(
                result.pred_depth_m,
                vmin=pred_range.min_m,
                vmax=pred_range.max_m,
                valid_max_m=self.depth_viz.valid_max_depth_m,
            ),
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
                "depth_visualization": {
                    "config": self.depth_viz.to_dict(),
                    "raw": raw_range.to_dict(),
                    "predicted": pred_range.to_dict(),
                },
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
