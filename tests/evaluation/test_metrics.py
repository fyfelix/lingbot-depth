import math

import numpy as np

from evaluation.core.metrics import compute_depth_metrics


def test_perfect_prediction_has_zero_error_and_full_accuracy():
    target = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    metrics = compute_depth_metrics(target.copy(), target)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["abs_rel"] == 0.0
    assert metrics["delta_1_05"] == 1.0
    assert metrics["delta_1_10"] == 1.0
    assert metrics["delta_1_25"] == 1.0


def test_invalid_values_are_excluded():
    prediction = np.array([[1.0, np.nan], [0.0, 2.0]], dtype=np.float32)
    target = np.array([[1.0, 3.0], [4.0, 1.0]], dtype=np.float32)
    metrics = compute_depth_metrics(prediction, target)

    assert metrics["mae"] == 0.5
    assert math.isclose(metrics["rmse"], math.sqrt(0.5))
    assert metrics["abs_rel"] == 0.5
    assert metrics["delta_1_25"] == 0.5


def test_empty_valid_mask_returns_nan_metrics():
    metrics = compute_depth_metrics(
        np.array([[np.nan]], dtype=np.float32),
        np.array([[1.0]], dtype=np.float32),
    )
    assert all(math.isnan(value) for value in metrics.values())
