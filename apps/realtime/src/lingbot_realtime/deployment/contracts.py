from __future__ import annotations

import re
from dataclasses import dataclass

INPUT_NAME = "rgbd_input"
OUTPUT_NAME = "depth"
DEFAULT_OPSET = 18
FP32_STABILITY_POLICY = "transformer_layernorm_softmax_add_layerscale_v1"


@dataclass(frozen=True, slots=True)
class Resolution:
    height: int = 480
    width: int = 640

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("resolution must be positive")

    @classmethod
    def parse(cls, value: str) -> "Resolution":
        match = re.fullmatch(r"\s*(\d+)\s*[xX*×]\s*(\d+)\s*", value)
        if match is None:
            raise ValueError(f"invalid resolution {value!r}; expected HxW")
        result = cls(int(match.group(1)), int(match.group(2)))
        if result != cls():
            raise ValueError("first deployment release is fixed to 480x640")
        return result

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 4, self.height, self.width)

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (1, self.height, self.width)

    @property
    def label(self) -> str:
        return f"{self.height}x{self.width}"
