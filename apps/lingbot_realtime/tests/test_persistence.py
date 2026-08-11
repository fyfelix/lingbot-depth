from __future__ import annotations

import json

import cv2

from lingbot_realtime.camera.fixture import FixtureFrameSource
from lingbot_realtime.domain import CaptureRecord
from lingbot_realtime.inference.mock import MockInferenceEngine
from lingbot_realtime.services.persistence import PersistenceService
from lingbot_realtime.visualization import DepthVisualizationConfig


def test_persistence_uses_shared_depth_visualization_policy(tmp_path) -> None:
    source = FixtureFrameSource(width=24, height=18, fps=120)
    source.start()
    frame = source.read()
    source.stop()
    record = CaptureRecord("capture_test", frame, status="ready")
    record.result = MockInferenceEngine().infer(frame)
    visualization = DepthVisualizationConfig(
        min_depth_m=0.2,
        max_depth_m=4.0,
        valid_max_depth_m=6.0,
        pred_percentile_min=5,
        pred_percentile_max=95,
    )
    persistence = PersistenceService(
        True,
        tmp_path,
        max_depth_m=6.0,
        depth_viz=visualization,
    )

    persistence.save_capture(record, "mock", "cpu")

    root = tmp_path / "capture_test"
    metadata = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    assert metadata["depth_visualization"]["raw"] == {"min_m": 0.2, "max_m": 4.0}
    assert (
        metadata["depth_visualization"]["predicted"]["max_m"]
        > metadata["depth_visualization"]["predicted"]["min_m"]
    )
    raw_visualization = cv2.imread(str(root / "raw_depth_vis.png"), cv2.IMREAD_COLOR)
    predicted_visualization = cv2.imread(str(root / "pred_depth_vis.png"), cv2.IMREAD_COLOR)
    assert raw_visualization.shape == (18, 24, 3)
    assert predicted_visualization.shape == (18, 24, 3)
