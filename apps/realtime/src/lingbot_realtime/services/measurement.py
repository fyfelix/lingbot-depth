from __future__ import annotations

import math

import numpy as np

from lingbot_realtime.domain import CameraIntrinsics, Measurement, SampledPoint


class InvalidMeasurement(ValueError):
    pass


def _sample_point(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    x: int,
    y: int,
    max_depth_m: float,
    search_radius: int,
) -> SampledPoint:
    height, width = depth_m.shape
    if x < 0 or y < 0 or x >= width or y >= height:
        raise InvalidMeasurement(f"Point ({x}, {y}) is outside {width}x{height}")

    candidates: list[tuple[int, int, int]] = []
    for radius in range(max(0, search_radius) + 1):
        for yy in range(max(0, y - radius), min(height, y + radius + 1)):
            for xx in range(max(0, x - radius), min(width, x + radius + 1)):
                if max(abs(xx - x), abs(yy - y)) != radius:
                    continue
                candidates.append((radius, yy, xx))
        valid_candidates = []
        for item_radius, yy, xx in candidates:
            value = float(depth_m[yy, xx])
            if math.isfinite(value) and 0 < value <= max_depth_m:
                distance_sq = (xx - x) ** 2 + (yy - y) ** 2
                valid_candidates.append((distance_sq, yy, xx, value, item_radius))
        if valid_candidates:
            _, sampled_y, sampled_x, z, used_radius = min(valid_candidates)
            break
        candidates.clear()
    else:
        raise InvalidMeasurement(f"No valid predicted depth near ({x}, {y})")

    sx = width / max(1, intrinsics.width)
    sy = height / max(1, intrinsics.height)
    fx = intrinsics.fx * sx
    fy = intrinsics.fy * sy
    ppx = intrinsics.ppx * sx
    ppy = intrinsics.ppy * sy
    px = sampled_x + 0.5
    py = sampled_y + 0.5
    xyz = (
        ((px - ppx) / fx) * z,
        ((py - ppy) / fy) * z,
        z,
    )
    return SampledPoint(
        requested_x=x,
        requested_y=y,
        sampled_x=sampled_x,
        sampled_y=sampled_y,
        radius=used_radius,
        depth_m=z,
        xyz_m=tuple(float(value) for value in xyz),
    )


def measure_distance(
    measurement_id: int,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    start_xy: tuple[int, int],
    end_xy: tuple[int, int],
    max_depth_m: float,
    search_radius: int = 2,
) -> Measurement:
    if depth_m.ndim != 2:
        raise InvalidMeasurement(f"Expected HxW depth, got {depth_m.shape}")
    start = _sample_point(depth_m, intrinsics, start_xy[0], start_xy[1], max_depth_m, search_radius)
    end = _sample_point(depth_m, intrinsics, end_xy[0], end_xy[1], max_depth_m, search_radius)
    distance = math.dist(start.xyz_m, end.xyz_m)
    return Measurement(
        measurement_id=measurement_id,
        start=start,
        end=end,
        distance_m=float(distance),
    )
