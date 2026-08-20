from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .packets import PredictionPacket


@dataclass(frozen=True, slots=True)
class WebFrame:
    header: MappingProxyType[str, Any]
    payload: bytes


class WebPublisher:
    """Build the canonical realtime WebGL packet used by the local web service."""

    PROTOCOL = "lingbot.realtime.webgl.v2"
    FLOW_CONTROL = "frame_ack"

    def __init__(
        self,
        sink: Callable[[WebFrame], Any] | None = None,
        *,
        max_depth_m: float = 6.0,
    ) -> None:
        if max_depth_m <= 0:
            raise ValueError("max_depth_m must be positive")
        self.sink = sink
        self.max_depth_m = float(max_depth_m)

    @staticmethod
    def effective_stride(shape: tuple[int, int], requested: int, point_budget: int) -> int:
        stride = max(1, int(requested))
        while (
            point_budget > 0
            and ((shape[0] + stride - 1) // stride) * ((shape[1] + stride - 1) // stride)
            > point_budget
        ):
            stride += 1
        return stride

    def build_frame(
        self,
        packet: PredictionPacket,
        *,
        include_color: bool = True,
        include_raw: bool = True,
        include_pred: bool = True,
        include_raw_cloud: bool = False,
        include_pred_cloud: bool = False,
        cloud_stride: int = 2,
        cloud_point_budget: int = 180_000,
        header_fields: dict[str, Any] | None = None,
    ) -> WebFrame:
        color = np.ascontiguousarray(packet.frame.color_bgr, dtype=np.uint8)
        raw = np.ascontiguousarray(packet.frame.raw_depth_mm, dtype=np.uint16)
        metric = np.nan_to_num(
            packet.depth_m,
            copy=True,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        prediction = np.rint(metric * np.float32(1000.0))
        invalid = (
            (metric <= 0.0) | (metric > self.max_depth_m) | (prediction > np.iinfo(np.uint16).max)
        )
        prediction[invalid] = 0
        pred = np.ascontiguousarray(prediction, dtype=np.uint16)
        values: list[tuple[str, np.ndarray, dict[str, Any]]] = []
        streams = {
            "color": bool(include_color),
            "raw_depth": bool(include_raw),
            "pred_depth": bool(include_pred),
            "raw_cloud_depth": bool(include_raw_cloud),
            "pred_cloud_depth": bool(include_pred_cloud),
        }
        if include_color:
            values.append(
                (
                    "color",
                    color,
                    {"shape": list(color.shape), "dtype": "uint8", "layout": "BGR"},
                )
            )
        if include_raw:
            values.append(
                (
                    "raw_depth",
                    raw,
                    {"shape": list(raw.shape), "dtype": "uint16", "unit": "millimeter"},
                )
            )
        if include_pred:
            values.append(
                (
                    "pred_depth",
                    pred,
                    {
                        "shape": list(pred.shape),
                        "dtype": "uint16",
                        "unit": "millimeter",
                        "source_dtype": "float32_meter",
                        "max_depth_m": self.max_depth_m,
                    },
                )
            )
        if include_raw_cloud:
            stride = self.effective_stride(raw.shape, cloud_stride, cloud_point_budget)
            cloud = np.ascontiguousarray(raw[::stride, ::stride])
            values.append(
                (
                    "raw_cloud_depth",
                    cloud,
                    {
                        "shape": list(cloud.shape),
                        "dtype": "uint16",
                        "unit": "millimeter",
                        "source": "raw_depth",
                        "stride": stride,
                        "requested_stride": max(1, int(cloud_stride)),
                        "point_budget": max(0, int(cloud_point_budget)),
                    },
                )
            )
        if include_pred_cloud:
            stride = self.effective_stride(pred.shape, cloud_stride, cloud_point_budget)
            cloud = np.ascontiguousarray(pred[::stride, ::stride])
            values.append(
                (
                    "pred_cloud_depth",
                    cloud,
                    {
                        "shape": list(cloud.shape),
                        "dtype": "uint16",
                        "unit": "millimeter",
                        "source": "pred_depth",
                        "source_dtype": "float32_meter",
                        "stride": stride,
                        "requested_stride": max(1, int(cloud_stride)),
                        "point_budget": max(0, int(cloud_point_budget)),
                    },
                )
            )
        payload = bytearray()
        layout: dict[str, Any] = {}
        for name, value, spec in values:
            alignment = max(1, value.dtype.itemsize)
            payload.extend(b"\0" * ((-len(payload)) % alignment))
            offset = len(payload)
            raw_bytes = value.tobytes(order="C")
            payload.extend(raw_bytes)
            layout[name] = {**spec, "offset": offset, "bytes": len(raw_bytes)}
        header = {
            "type": "frame",
            "protocol": self.PROTOCOL,
            "flow_control": self.FLOW_CONTROL,
            "frame_id": packet.frame.frame_id,
            "timestamp_ns": packet.frame.timestamp_ns,
            "streams": streams,
            "cloud_stride": max(1, int(cloud_stride)),
            "cloud_point_budget": max(0, int(cloud_point_budget)),
            **dict(header_fields or {}),
            **layout,
            "payload_bytes": len(payload),
        }
        return WebFrame(MappingProxyType(header), bytes(payload))

    def publish(self, packet: PredictionPacket) -> None:
        if self.sink is None:
            raise RuntimeError("WebPublisher has no sink")
        self.sink(self.build_frame(packet))

    def close(self) -> None:
        return None
