from __future__ import annotations

import sys
import types

import pytest

from lingbot_realtime.camera.realsense import RealSenseFrameSource


def test_realsense_start_fails_fast_when_no_device_is_enumerated(monkeypatch) -> None:
    class EmptyContext:
        def query_devices(self):
            return []

    fake_rs = types.SimpleNamespace(context=EmptyContext)
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)

    with pytest.raises(RuntimeError, match="No RealSense device detected"):
        RealSenseFrameSource().start()


def test_realsense_device_enumeration_errors_are_actionable() -> None:
    class BrokenContext:
        def query_devices(self):
            raise RuntimeError("permission denied")

    fake_rs = types.SimpleNamespace(context=BrokenContext)

    with pytest.raises(
        RuntimeError, match="Unable to enumerate RealSense devices: permission denied"
    ):
        RealSenseFrameSource._ensure_device(fake_rs)
