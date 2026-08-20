from __future__ import annotations

import time

import numpy as np
from fastapi.testclient import TestClient

from lingbot_realtime.camera.fixture import FixtureFrameSource
from lingbot_realtime.config import AppConfig
from lingbot_realtime.inference.mock import MockInferenceEngine
from lingbot_realtime.realtime import D435HostPreprocessor, FramePacket, WebPublisher
from lingbot_realtime.realtime.preprocess import Resolution
from lingbot_realtime.services.persistence import PersistenceService
from lingbot_realtime.services.runtime import RuntimeController
from lingbot_realtime.web.app import create_app


def _wait(predicate, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def _runtime(tmp_path=None, *, auto_connect=True) -> RuntimeController:
    config = AppConfig(
        source="fixture",
        inference_engine="mock",
        width=32,
        height=24,
        fps=120,
        preview_fps=60,
        auto_connect=auto_connect,
        record_root=tmp_path or AppConfig.record_root,
    )
    return RuntimeController(
        config,
        FixtureFrameSource(32, 24, 120),
        MockInferenceEngine(),
        PersistenceService(False, config.output_root, config.max_depth_m),
    )


def test_frame_packet_and_domain_arrays_are_immutable() -> None:
    color = np.zeros((4, 6, 3), dtype=np.uint8)
    raw = np.ones((4, 6), dtype=np.uint16)
    packet = FramePacket(1, 2, color, raw)
    color[:] = 255
    raw[:] = 9
    assert not packet.color_bgr.flags.writeable
    assert not packet.raw_depth_mm.flags.writeable
    assert packet.color_bgr.max() == 0
    assert packet.raw_depth_mm.max() == 1


def test_host_preprocess_uses_rgb_zero_one_and_metric_depth() -> None:
    color = np.zeros((2, 3, 3), dtype=np.uint8)
    color[..., 2] = 255
    raw = np.full((2, 3), 2000, dtype=np.uint16)
    raw[0, 0] = 0
    inputs = D435HostPreprocessor(Resolution(480, 640), max_depth_m=6.0).prepare(
        FramePacket(0, 0, color, raw)
    )
    assert inputs.shape == (1, 4, 480, 640)
    assert inputs.dtype == np.float32
    assert inputs[0, 0].max() == 1.0
    assert inputs[0, 1:3].max() == 0.0
    assert inputs[0, 3].max() == 2.0
    assert inputs[0, 3].min() == 0.0


def test_controller_streams_continuously_and_snapshot_freezes_latest_prediction() -> None:
    runtime = _runtime()
    runtime.start()
    try:
        _wait(lambda: runtime.status()["prediction_frame_id"] is not None)
        before = runtime.status()["frame_id"]
        capture = runtime.capture()
        _wait(lambda: runtime.status()["phase"] == "ready")
        frozen = runtime.get_capture(capture["capture_id"])
        _wait(lambda: runtime.status()["frame_id"] > before)
        assert frozen["frame_id"] == capture["frame_id"]
        assert runtime.status()["frame_id"] > frozen["frame_id"]
    finally:
        runtime.shutdown()


def test_connect_disconnect_inference_record_and_realtime_ack(tmp_path) -> None:
    runtime = _runtime(tmp_path, auto_connect=False)
    with TestClient(create_app(runtime)) as client:
        assert client.get("/status").json()["camera_status"] == "idle"
        assert client.post("/camera/connect").status_code == 200
        _wait(lambda: client.get("/status").json()["prediction_frame_id"] is not None)
        assert client.post("/record/toggle").status_code == 200
        _wait(lambda: client.get("/status").json()["saved"] > 0)
        assert client.post("/record/toggle").status_code == 200
        assert client.post("/inference?enabled=false").status_code == 200
        assert client.post("/inference?enabled=true").status_code == 200
        with client.websocket_connect("/ws/realtime?cloud_point_budget=40") as websocket:
            ready = websocket.receive_json()
            assert ready["flow_control"] == "frame_ack"
            header = websocket.receive_json()
            payload = websocket.receive_bytes()
            assert header["payload_bytes"] == len(payload)
            assert np.prod(header["pred_cloud_depth"]["shape"]) <= 40
            websocket.send_json({"type": "frame_ack", "frame_id": header["frame_id"]})
        assert client.post("/camera/disconnect").status_code == 200


def test_point_budget_increases_stride() -> None:
    stride = WebPublisher.effective_stride((480, 640), requested=1, point_budget=1000)
    assert stride > 1
    assert ((479 + stride) // stride) * ((639 + stride) // stride) <= 1000


def test_camera_retries_pause_after_repeated_failures() -> None:
    class FailingSource:
        name = "fixture"

        def start(self) -> None:
            raise RuntimeError("synthetic camera failure")

        def read(self, timeout_sec: float = 5.0):
            raise AssertionError("read should not be called")

        def stop(self) -> None:
            return None

    config = AppConfig(source="fixture", auto_connect=True, inference_enabled=False)
    runtime = RuntimeController(
        config,
        FailingSource(),
        None,
        PersistenceService(False, config.output_root, config.max_depth_m),
    )
    runtime.start()
    try:
        _wait(lambda: runtime.status()["camera_retry_paused"], timeout=5.0)
        status = runtime.status()
        assert status["camera_status"] == "error"
        assert "Automatic retries paused" in status["camera_error"]
        assert status["frame_drops"] == 3

        runtime.connect_camera()
        assert runtime.status()["camera_retry_paused"] is False
    finally:
        runtime.shutdown()
