from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np

from lingbot_realtime.domain import PredictionPacket
from lingbot_realtime.realtime import (
    FramePacket,
    WebPublisher,
)
from lingbot_realtime.realtime import (
    PredictionPacket as WirePredictionPacket,
)


def to_wire_packet(packet: PredictionPacket) -> WirePredictionPacket:
    frame = FramePacket(
        frame_id=packet.frame.frame_id,
        timestamp_ns=int(packet.frame.timestamp * 1_000_000_000),
        color_bgr=np.ascontiguousarray(packet.frame.color_rgb[..., ::-1]),
        raw_depth_mm=np.ascontiguousarray(
            np.rint(packet.frame.depth_m * 1000.0).clip(0, 65535),
            dtype=np.uint16,
        ),
        metadata=MappingProxyType({"intrinsics": packet.frame.intrinsics.to_dict()}),
    )
    depth = (
        packet.result.pred_depth_m
        if packet.result is not None
        else np.zeros_like(packet.frame.depth_m)
    )
    return WirePredictionPacket(
        frame,
        depth,
        MappingProxyType(dict(packet.timings_ms)),
        MappingProxyType({"source": "continuous"}),
    )


def build_web_frame(
    packet: PredictionPacket,
    *,
    max_depth_m: float,
    include_color: bool = True,
    include_raw: bool = True,
    include_pred: bool = True,
    include_raw_cloud: bool = False,
    include_pred_cloud: bool = True,
    cloud_stride: int,
    cloud_point_budget: int,
    fields: dict[str, Any],
) -> tuple[dict[str, Any], bytes]:
    prediction_available = packet.result is not None
    frame = WebPublisher(max_depth_m=max_depth_m).build_frame(
        to_wire_packet(packet),
        include_color=include_color,
        include_raw=include_raw,
        include_pred=include_pred and prediction_available,
        include_raw_cloud=include_raw_cloud,
        include_pred_cloud=include_pred_cloud and prediction_available,
        cloud_stride=cloud_stride,
        cloud_point_budget=cloud_point_budget,
        header_fields={
            **fields,
            "intrinsics": packet.frame.intrinsics.to_dict(),
            "depth_scale_m": packet.frame.depth_scale_m,
            "color_layout": "BGR",
            "depth_unit": "millimeter",
        },
    )
    return dict(frame.header), frame.payload


POINT_DTYPE = np.dtype([("xyz", "<f4", (3,)), ("rgba", "u1", (4,))])


def build_pointcloud(
    packet: PredictionPacket, *, stride: int, max_depth_m: float, point_budget: int
) -> bytes:
    frame = packet.frame
    intr = frame.intrinsics
    depth = frame.depth_m[::stride, ::stride]
    valid = np.isfinite(depth) & (depth > 0) & (depth <= max_depth_m)
    yy, xx = np.mgrid[0 : frame.depth_m.shape[0] : stride, 0 : frame.depth_m.shape[1] : stride]
    z = depth[valid]
    x = ((xx[valid] + 0.5 - intr.ppx) / intr.fx) * z
    y = -((yy[valid] + 0.5 - intr.ppy) / intr.fy) * z
    rgb = frame.color_rgb[::stride, ::stride][valid]
    if point_budget > 0 and len(z) > point_budget:
        keep = np.linspace(0, len(z) - 1, point_budget, dtype=np.int64)
        x, y, z, rgb = x[keep], y[keep], z[keep], rgb[keep]
    output = np.empty(len(z), dtype=POINT_DTYPE)
    output["xyz"][:, 0] = x
    output["xyz"][:, 1] = y
    output["xyz"][:, 2] = -z
    output["rgba"][:, :3] = rgb
    output["rgba"][:, 3] = 255
    return output.tobytes()
