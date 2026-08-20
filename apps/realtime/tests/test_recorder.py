from __future__ import annotations

import json
from types import MappingProxyType

import cv2
import numpy as np

from lingbot_realtime.realtime import FramePacket, PredictionPacket, Recorder, make_session_paths


def test_continuous_recorder_roundtrip(tmp_path) -> None:
    paths = make_session_paths(tmp_path, session_id="test")
    recorder = Recorder(paths, fps=30)
    for index in range(3):
        color = np.full((8, 10, 3), index * 20, dtype=np.uint8)
        raw = np.full((8, 10), 1000 + index, dtype=np.uint16)
        frame = FramePacket(
            index,
            100 + index,
            color,
            raw,
            MappingProxyType({"intrinsics": {"fx": 5.0}}),
        )
        recorder.publish(PredictionPacket(frame, raw.astype(np.float32) / 1000.0))
    recorder.close()

    raw_depth = np.load(paths.raw_depth_path)
    pred_depth = np.load(paths.pred_depth_path)
    lines = paths.frames_path.read_text(encoding="utf-8").splitlines()
    metadata = json.loads(paths.meta_path.read_text(encoding="utf-8"))
    video = cv2.VideoCapture(str(paths.rgb_path))
    assert raw_depth.shape == (3, 8, 10) and raw_depth.dtype == np.uint16
    assert pred_depth.shape == (3, 8, 10) and pred_depth.dtype == np.float32
    assert len(lines) == 3
    assert metadata["contracts"]["raw_depth"].startswith("uint16 millimeter")
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
    video.release()
