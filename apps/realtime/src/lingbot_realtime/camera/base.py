from __future__ import annotations

from typing import Protocol

from lingbot_realtime.domain import RGBDFrame


class FrameSource(Protocol):
    @property
    def name(self) -> str: ...

    def start(self) -> None: ...

    def read(self, timeout_sec: float = 5.0) -> RGBDFrame: ...

    def stop(self) -> None: ...
