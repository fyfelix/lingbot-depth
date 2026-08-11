from __future__ import annotations

import numpy as np

from lingbot_realtime.inference.preprocessing import sanitize_metric_depth


def test_sanitize_metric_depth_preserves_valid_metric_values_only() -> None:
    source = np.array(
        [[np.nan, np.inf, -1.0, 0.0, 0.4, 5.9, 6.1]],
        dtype=np.float32,
    )
    original = source.copy()

    sanitized = sanitize_metric_depth(source, max_depth_m=6.0)

    np.testing.assert_array_equal(
        sanitized,
        np.array([[0.0, 0.0, 0.0, 0.0, 0.4, 5.9, 0.0]], dtype=np.float32),
    )
    assert sanitized.flags.c_contiguous
    np.testing.assert_array_equal(source, original)
