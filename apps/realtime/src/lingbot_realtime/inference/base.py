from __future__ import annotations

from typing import Protocol

from lingbot_realtime.domain import InferenceResult, RGBDFrame


class InferenceEngine(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def device_name(self) -> str: ...

    def load(self) -> None: ...

    def infer(self, frame: RGBDFrame) -> InferenceResult: ...

    def close(self) -> None: ...
