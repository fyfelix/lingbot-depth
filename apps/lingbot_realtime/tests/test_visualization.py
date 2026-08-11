from __future__ import annotations

import numpy as np
import pytest

from lingbot_realtime.visualization import (
    DepthVisualizationConfig,
    colorize_depth_fast,
    depth_percentile_range,
)


def test_percentile_range_ignores_invalid_values_and_outliers() -> None:
    depth = np.array([[0.0, np.nan, np.inf, 1.0, 2.0, 100.0]], dtype=np.float32)

    minimum, maximum = depth_percentile_range(
        depth,
        percentile_min=0,
        percentile_max=50,
        fallback_min=0.1,
        fallback_max=5.0,
    )

    assert minimum == pytest.approx(1.0)
    assert maximum == pytest.approx(2.0)


def test_percentile_range_uses_configured_fallback_without_valid_depth() -> None:
    minimum, maximum = depth_percentile_range(
        np.array([[0.0, np.nan]], dtype=np.float32),
        fallback_min=0.2,
        fallback_max=4.0,
    )

    assert (minimum, maximum) == pytest.approx((0.2, 4.0))


def test_fast_colorizer_preserves_input_and_renders_invalid_pixels_black() -> None:
    depth = np.array([[0.0, 0.5, 2.0, 7.0, np.nan]], dtype=np.float32)
    original = depth.copy()

    colored = colorize_depth_fast(depth, vmin=0.1, vmax=5.0, valid_max_m=6.0)

    assert colored.shape == (1, 5, 3)
    assert colored.dtype == np.uint8
    np.testing.assert_array_equal(colored[0, 0], np.zeros(3, dtype=np.uint8))
    np.testing.assert_array_equal(colored[0, 3], np.zeros(3, dtype=np.uint8))
    np.testing.assert_array_equal(colored[0, 4], np.zeros(3, dtype=np.uint8))
    assert colored[0, 1].any()
    np.testing.assert_array_equal(depth, original)


def test_prediction_range_uses_percentiles_while_raw_range_is_fixed() -> None:
    config = DepthVisualizationConfig(
        min_depth_m=0.1,
        max_depth_m=5.0,
        pred_percentile_min=25,
        pred_percentile_max=75,
    )
    depth = np.array([[1.0, 2.0, 3.0, 100.0]], dtype=np.float32)

    assert config.raw_range().to_dict() == {"min_m": 0.1, "max_m": 5.0}
    predicted = config.predicted_range(depth)
    assert predicted.min_m == pytest.approx(1.75)
    assert predicted.max_m == pytest.approx(27.25)
