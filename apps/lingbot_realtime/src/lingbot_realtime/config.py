from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lingbot_realtime.visualization import DepthVisualizationConfig


@dataclass(frozen=True)
class AppConfig:
    source: str = "realsense"
    inference_engine: str = "mdm"
    model_path: str | None = None
    device: str = "auto"
    width: int = 640
    height: int = 480
    fps: int = 30
    max_depth_m: float = 6.0
    resolution_level: int = 9
    apply_mask: bool = True
    bind: str = "127.0.0.1"
    port: int = 8000
    preview_fps: float = 15.0
    ack_timeout_sec: float = 10.0
    save_results: bool = False
    output_root: Path = Path("apps/lingbot_realtime/runs")
    vis_min_m: float = 0.1
    vis_max_m: float = 5.0
    pred_vis_percentile_min: float = 1.0
    pred_vis_percentile_max: float = 99.0

    def depth_viz_config(self) -> DepthVisualizationConfig:
        return DepthVisualizationConfig(
            min_depth_m=self.vis_min_m,
            max_depth_m=self.vis_max_m,
            valid_max_depth_m=self.max_depth_m,
            pred_percentile_min=self.pred_vis_percentile_min,
            pred_percentile_max=self.pred_vis_percentile_max,
        )

    def validate(self) -> None:
        if self.source not in {"fixture", "realsense"}:
            raise ValueError(f"Unsupported source: {self.source}")
        if self.inference_engine not in {"mock", "mdm"}:
            raise ValueError(f"Unsupported inference engine: {self.inference_engine}")
        if self.inference_engine == "mdm" and not self.model_path:
            raise ValueError("--model-path is required when --inference-engine=mdm")
        if self.width <= 0 or self.height <= 0 or self.fps <= 0:
            raise ValueError("Camera width, height and fps must be positive")
        if self.max_depth_m <= 0:
            raise ValueError("--max-depth must be positive")
        if self.vis_min_m < 0 or self.vis_max_m <= self.vis_min_m:
            raise ValueError("--vis-max must be greater than --vis-min")
        if not 0 <= self.pred_vis_percentile_min < self.pred_vis_percentile_max <= 100:
            raise ValueError("Invalid predicted depth visualization percentiles")
        if not 0 <= self.resolution_level <= 9:
            raise ValueError("--resolution-level must be between 0 and 9")
        if self.preview_fps <= 0:
            raise ValueError("--preview-fps must be positive")
        if self.ack_timeout_sec <= 0:
            raise ValueError("--ack-timeout must be positive")
