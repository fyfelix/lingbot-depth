import time

from lingbot_realtime.camera.fixture import FixtureFrameSource
from lingbot_realtime.config import AppConfig
from lingbot_realtime.inference.mock import MockInferenceEngine
from lingbot_realtime.services.persistence import PersistenceService
from lingbot_realtime.services.runtime import RuntimeConflict, RuntimeController


def _wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition did not become true")


def _runtime():
    config = AppConfig(
        source="fixture",
        inference_engine="mock",
        width=32,
        height=24,
        fps=120,
        preview_fps=30,
        save_results=False,
    )
    runtime = RuntimeController(
        config,
        FixtureFrameSource(32, 24, 120),
        MockInferenceEngine(),
        PersistenceService(False, config.output_root, config.max_depth_m),
    )
    runtime.start()
    return runtime


def test_preview_does_not_infer_and_capture_infers_once():
    runtime = _runtime()
    try:
        _wait_until(lambda: runtime.status()["frame_id"] is not None)
        assert runtime.status()["phase"] == "preview"
        capture = runtime.capture()
        assert capture["status"] == "inferencing"
        _wait_until(lambda: runtime.status()["phase"] == "ready")
        status = runtime.status()
        assert status["capture"]["status"] == "ready"
        assert runtime.get_capture(capture["capture_id"])["status"] == "ready"
    finally:
        runtime.shutdown()


def test_capture_is_rejected_until_model_and_frame_are_ready():
    runtime = _runtime()
    try:
        _wait_until(lambda: runtime.status()["frame_id"] is not None)
        # The mock loader is fast, so this assertion checks the normal ready path.
        _wait_until(lambda: runtime.status()["model_status"] == "ready")
        runtime.capture()
        try:
            runtime.capture()
        except RuntimeConflict:
            pass
        else:
            raise AssertionError("second capture should be rejected")
    finally:
        runtime.shutdown()
