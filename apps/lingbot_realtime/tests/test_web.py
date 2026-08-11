import time

from fastapi.testclient import TestClient

from lingbot_realtime.camera.fixture import FixtureFrameSource
from lingbot_realtime.config import AppConfig
from lingbot_realtime.inference.mock import MockInferenceEngine
from lingbot_realtime.services.persistence import PersistenceService
from lingbot_realtime.services.runtime import RuntimeController
from lingbot_realtime.web.app import create_app


def _make_app():
    config = AppConfig(
        source="fixture",
        inference_engine="mock",
        width=24,
        height=18,
        fps=120,
        preview_fps=30,
        save_results=False,
    )
    runtime = RuntimeController(
        config,
        FixtureFrameSource(24, 18, 120),
        MockInferenceEngine(),
        PersistenceService(False, config.output_root, config.max_depth_m),
    )
    return create_app(runtime)


def _wait_status(client, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get("/status").json()
        if predicate(status):
            return status
        time.sleep(0.02)
    raise AssertionError("status predicate did not become true")


def test_web_capture_and_measurement_routes():
    with TestClient(_make_app()) as client:
        _wait_status(client, lambda s: s["frame_id"] is not None and s["model_status"] == "ready")
        response = client.post("/api/capture")
        assert response.status_code == 202
        capture_id = response.json()["capture_id"]
        status = _wait_status(client, lambda s: s["phase"] == "ready")
        assert status["capture"]["capture_id"] == capture_id
        measured = client.post(
            f"/api/captures/{capture_id}/measurements",
            json={"start": [3, 3], "end": [15, 3]},
        )
        assert measured.status_code == 201
        assert measured.json()["distance_m"] > 0
        assert client.delete(f"/api/captures/{capture_id}/measurements").status_code == 200
        assert client.post("/api/recapture").status_code == 200
