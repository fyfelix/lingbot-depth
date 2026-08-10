from pathlib import Path

import cv2
import numpy as np

from evaluation.core.io import read_gt_depth, read_raw_depth


def test_png_depth_scaling_and_validity(tmp_path: Path):
    path = tmp_path / "depth.png"
    value = np.array([[0, 1000], [2000, 7000]], dtype=np.uint16)
    assert cv2.imwrite(str(path), value)

    raw = read_raw_depth(path, depth_scale=1000.0, min_depth=0.1, max_depth=5.0)
    gt = read_gt_depth(path, depth_scale=1000.0, min_depth=0.1, max_depth=5.0)

    np.testing.assert_array_equal(raw, np.array([[0.0, 1.0], [2.0, 0.0]], dtype=np.float32))
    assert np.isnan(gt[0, 0])
    assert np.isnan(gt[1, 1])
    assert gt[0, 1] == 1.0


def test_exr_depth_is_already_in_meters(tmp_path: Path):
    path = tmp_path / "depth.exr"
    value = np.full((3, 4), 1.25, dtype=np.float32)
    assert cv2.imwrite(str(path), value)

    loaded = read_raw_depth(path, depth_scale=1.0, min_depth=0.1, max_depth=10.0)
    np.testing.assert_allclose(loaded, value)
