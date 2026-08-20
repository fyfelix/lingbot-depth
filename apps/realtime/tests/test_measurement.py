import math

import numpy as np
import pytest

from lingbot_realtime.domain import CameraIntrinsics
from lingbot_realtime.services.measurement import InvalidMeasurement, measure_distance


def test_measurement_projects_two_points_in_metric_camera_space() -> None:
    intrinsics = CameraIntrinsics(width=10, height=4, fx=100.0, fy=100.0, ppx=4.5, ppy=1.5)
    depth = np.ones((4, 10), dtype=np.float32)
    result = measure_distance(1, depth, intrinsics, (4, 1), (5, 1), 6.0)
    assert result.measurement_id == 1
    assert result.start.depth_m == pytest.approx(1.0)
    assert result.end.depth_m == pytest.approx(1.0)
    assert result.distance_m == pytest.approx(0.01)


def test_measurement_falls_back_to_nearest_valid_depth() -> None:
    intrinsics = CameraIntrinsics(width=3, height=3, fx=100.0, fy=100.0, ppx=1.0, ppy=1.0)
    depth = np.zeros((3, 3), dtype=np.float32)
    depth[1, 2] = 2.0
    result = measure_distance(1, depth, intrinsics, (1, 1), (2, 1), 6.0, search_radius=2)
    assert result.start.sampled_x == 2
    assert result.start.sampled_y == 1
    assert result.start.radius == 1
    assert math.isfinite(result.distance_m)


def test_measurement_rejects_invalid_points() -> None:
    intrinsics = CameraIntrinsics(width=3, height=3, fx=100.0, fy=100.0, ppx=1.0, ppy=1.0)
    with pytest.raises(InvalidMeasurement):
        measure_distance(1, np.zeros((3, 3), dtype=np.float32), intrinsics, (0, 0), (1, 1), 6.0)
