from __future__ import annotations

import json

import numpy as np
import pytest

from lingbot_realtime.camera.fixture import FixtureFrameSource
from lingbot_realtime.inference.tensorrt import (
    TensorRTInferenceEngine,
    validate_deployment_manifest,
)


class _Runner:
    input_shape = (1, 4, 480, 640)

    def __init__(self) -> None:
        self.closed = False
        self.last_input = None

    def infer(self, inputs):
        self.last_input = np.asarray(inputs)
        return np.ones((1, 480, 640), dtype=np.float16)

    def close(self) -> None:
        self.closed = True


def test_tensorrt_fake_runner_uses_fp16_fixed_contract(tmp_path) -> None:
    runner = _Runner()
    engine = TensorRTInferenceEngine(tmp_path / "missing.engine", runner=runner)
    engine.load()
    source = FixtureFrameSource(32, 24, 120)
    source.start()
    result = engine.infer(source.read())
    source.stop()
    assert runner.last_input.shape == (1, 4, 480, 640)
    assert runner.last_input.dtype == np.float16
    assert float(runner.last_input[:, :3].min()) >= 0.0
    assert float(runner.last_input[:, :3].max()) <= 1.0
    assert float(runner.last_input[:, 3].max()) <= engine.max_depth_m
    assert result.pred_depth_m.shape == (24, 32)
    assert result.pred_depth_m.dtype == np.float32
    engine.close()
    assert runner.closed


def test_manifest_rejects_tensorrt_major_and_checksum_mismatch(tmp_path) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"engine")
    manifest = tmp_path / "deployment.json"
    manifest.write_text(
        json.dumps(
            {
                "precision": "fp16",
                "tensorrt_major": 11,
                "tensor_contract": {
                    "input": {"name": "rgbd_input", "shape": [1, 4, 480, 640]},
                    "output": {"name": "depth", "shape": [1, 480, 640]},
                },
                "artifacts": {"engine": {"sha256": "0" * 64}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="checksum"):
        validate_deployment_manifest(manifest, engine, runtime_tensorrt_version="11.0")
    payload = json.loads(manifest.read_text())
    import hashlib

    payload["artifacts"]["engine"]["sha256"] = hashlib.sha256(b"engine").hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="major mismatch"):
        validate_deployment_manifest(manifest, engine, runtime_tensorrt_version="10.7")
