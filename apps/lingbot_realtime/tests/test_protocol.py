import numpy as np

from lingbot_realtime.camera.fixture import FixtureFrameSource
from lingbot_realtime.domain import CaptureRecord, InferenceResult
from lingbot_realtime.web.protocol import pack_capture, pack_preview


def _frame():
    source = FixtureFrameSource(width=8, height=6, fps=120)
    source.start()
    return source.read()


def test_preview_payload_has_typed_arrays_and_offsets() -> None:
    frame = _frame()
    header, payload = pack_preview(frame, revision=7, phase="preview")
    assert header["protocol"].startswith("lingbot.realtime")
    assert header["payload_bytes"] == len(payload)
    color = header["color"]
    raw = header["raw_depth"]
    assert color["dtype"] == "uint8"
    assert raw["dtype"] == "uint16"
    assert raw["unit"] == "device_unit"
    assert raw["scale_m"] == frame.depth_scale_m
    assert raw["visualization"] == {
        "min_m": 0.1,
        "max_m": 5.0,
        "valid_max_m": 6.0,
        "colormap": "turbo",
        "range_mode": "fixed",
    }
    assert (
        np.frombuffer(
            payload,
            dtype=np.uint8,
            count=color["bytes"],
            offset=color["offset"],
        ).size
        == 8 * 6 * 3
    )


def test_capture_payload_includes_prediction() -> None:
    frame = _frame()
    record = CaptureRecord("capture_test", frame)
    record.status = "ready"
    record.result = InferenceResult(
        pred_depth_m=np.ones((6, 8), dtype=np.float32),
        points=np.ones((6, 8, 3), dtype=np.float32),
        elapsed_sec=0.01,
    )
    header, payload = pack_capture(record, revision=8, phase="ready")
    assert header["type"] == "capture_result"
    assert header["pred_depth"]["dtype"] == "float32"
    assert header["pred_depth"]["unit"] == "meter"
    pred_viz = header["pred_depth"]["visualization"]
    assert pred_viz["range_mode"] == "percentile"
    assert pred_viz["percentile"] == {"min": 1.0, "max": 99.0}
    assert pred_viz["min_m"] == 1.0
    assert pred_viz["max_m"] > pred_viz["min_m"]
    assert len(payload) == header["payload_bytes"]
