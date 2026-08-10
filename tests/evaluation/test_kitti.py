import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from evaluation.core.inference import InferenceInputDataset
from evaluation.core.io import normalize_asdepth_prediction
from evaluation.core.output import RunLayout, save_prediction
from evaluation.core.pipeline import run_pipeline
from evaluation.core.types import RunConfig
from evaluation.datasets.kitti import (
    KITTI_BENCHMARK_NAME,
    KITTI_DEPTH_SCALE,
    KITTI_MIN_DEPTH,
    load_kitti,
)
from evaluation.evaluators.depth import run_depth_evaluation
from evaluation.prepare_kitti_jsonl import build_manifest


class FakeModel:
    def infer(self, _image, depth_in, **_kwargs):
        return {"depth": depth_in.clone()}


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def write_rgb(path: Path, shape=(4, 5)):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.full((*shape, 3), 127, dtype=np.uint8))


def write_depth(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.asarray(values, dtype=np.uint16))


def kitti_row(name: str):
    return {
        "rgb": f"image/{name}.png",
        "lidar": f"velodyne_raw/{name.replace('_sync_image_', '_sync_velodyne_raw_')}.png",
        "depth": (
            "groundtruth_depth/"
            f"{name.replace('_sync_image_', '_sync_groundtruth_depth_')}.png"
        ),
        "intrinsics": f"intrinsics/{name}.txt",
        "name": name,
        # The adapter intentionally ignores this historical GT upper limit.
        "depth-range": [KITTI_MIN_DEPTH, 80.0],
    }


def test_kitti_adapter_uses_devkit_gt_range_and_separate_raw_limit(tmp_path):
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    manifest = tmp_path / "val_selection_cropped.jsonl"
    write_jsonl(manifest, [kitti_row(name)])

    collection = load_kitti(manifest)
    sample = collection.samples[0]

    assert collection.name == "kitti"
    assert collection.metadata["benchmark"] == KITTI_BENCHMARK_NAME
    assert sample.sample_id == name
    assert sample.subset == "default"
    assert sample.depth_scale == KITTI_DEPTH_SCALE
    assert sample.min_depth == KITTI_MIN_DEPTH
    assert sample.max_depth == float("inf")
    assert sample.raw_max_depth == 80.0
    assert sample.metadata["scene"] == "2011_09_26_drive_0005"
    assert sample.metadata["frame"] == "0000000128_image_02"
    assert sample.metadata["intrinsics_path"] == str(
        (tmp_path / f"intrinsics/{name}.txt").resolve()
    )


@pytest.mark.parametrize("raw_key", ["lidar", "velodyne_raw", "raw_depth"])
def test_kitti_adapter_accepts_all_raw_depth_field_names(tmp_path, raw_key):
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    row = kitti_row(name)
    raw_value = row.pop("lidar")
    row[raw_key] = raw_value
    manifest = tmp_path / f"{raw_key}.jsonl"
    write_jsonl(manifest, [row])

    sample = load_kitti(manifest).samples[0]

    assert sample.raw_depth_path == (tmp_path / raw_value).resolve()


def test_kitti_raw_is_clipped_at_80m_but_gt_is_not(tmp_path):
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    manifest = tmp_path / "val.jsonl"
    row = kitti_row(name)
    write_jsonl(manifest, [row])
    write_rgb(tmp_path / row["rgb"], shape=(1, 3))
    encoded = np.array([[0, 256, 81 * 256]], dtype=np.uint16)
    write_depth(tmp_path / row["lidar"], encoded)
    write_depth(tmp_path / row["depth"], encoded)

    loaded = InferenceInputDataset(load_kitti(manifest).samples, load_gt=True)[0]

    np.testing.assert_array_equal(loaded.raw_depth, np.array([[0.0, 1.0, 0.0]]))
    assert np.isnan(loaded.gt_depth[0, 0])
    np.testing.assert_allclose(loaded.gt_depth[0, 1:], np.array([1.0, 81.0]))


def test_prepare_manifest_pairs_standard_val_selection_layout(tmp_path):
    selection = tmp_path / "depth_selection/val_selection_cropped"
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    row = kitti_row(name)
    for relative in (row["rgb"], row["lidar"], row["depth"], row["intrinsics"]):
        path = selection / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    output = tmp_path / "manifests/kitti.jsonl"

    rows, selection_root = build_manifest(tmp_path, output, allow_partial=True)

    assert selection_root == selection.resolve()
    assert len(rows) == 1
    assert "depth-range" not in rows[0]
    assert load_kitti(output).samples[0].sample_id == name


def test_kitti_pipeline_writes_standard_and_specialized_visualizations(
    tmp_path, monkeypatch
):
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    manifest = tmp_path / "val.jsonl"
    row = kitti_row(name)
    write_jsonl(manifest, [row])
    write_rgb(tmp_path / row["rgb"])
    depth = np.full((4, 5), 256, dtype=np.uint16)
    write_depth(tmp_path / row["lidar"], depth)
    write_depth(tmp_path / row["depth"], depth)
    intrinsics_path = tmp_path / row["intrinsics"]
    intrinsics_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(intrinsics_path, np.array([[10, 0, 2], [0, 10, 2], [0, 0, 1]]))

    collection = load_kitti(manifest)
    sample = collection.samples[0]
    run_dir = tmp_path / "outputs/run"
    config = RunConfig(
        dataset="kitti",
        stage="all",
        run_dir=run_dir,
        model_path="fake-model",
        device="cpu",
        save_visualizations=True,
        visualization_max_depth=80.0,
        disable_pointcloud_knn_filter=True,
    )
    monkeypatch.setattr("evaluation.core.inference.load_model", lambda *_args: FakeModel())

    layout = run_pipeline(collection, config)

    assert layout.prediction_path(sample).is_file()
    assert layout.visualization_path(sample).is_file()
    assert layout.kitti_prediction_visualization_path(sample).is_file()
    assert layout.kitti_pointcloud_visualization_path(sample).is_file()
    with layout.metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata["results"]["evaluation"]["summary"]["overall"]["rmse"] == 0.0


def test_kitti_visualizations_can_be_disabled(tmp_path, monkeypatch):
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    manifest = tmp_path / "val.jsonl"
    row = kitti_row(name)
    write_jsonl(manifest, [row])
    write_rgb(tmp_path / row["rgb"])
    depth = np.full((4, 5), 256, dtype=np.uint16)
    write_depth(tmp_path / row["lidar"], depth)
    write_depth(tmp_path / row["depth"], depth)
    collection = load_kitti(manifest)
    sample = collection.samples[0]
    config = RunConfig(
        dataset="kitti",
        stage="all",
        run_dir=tmp_path / "outputs/run",
        model_path="fake-model",
        device="cpu",
        save_visualizations=False,
        visualization_max_depth=80.0,
    )
    monkeypatch.setattr("evaluation.core.inference.load_model", lambda *_args: FakeModel())

    layout = run_pipeline(collection, config)

    assert not layout.kitti_prediction_visualization_path(sample).exists()
    assert not layout.visualization_path(sample).exists()
    assert not layout.kitti_pointcloud_visualization_path(sample).exists()


def test_kitti_evaluation_includes_gt_beyond_raw_limit(tmp_path):
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    manifest = tmp_path / "val.jsonl"
    row = kitti_row(name)
    write_jsonl(manifest, [row])
    target = np.array([[0, 256, 81 * 256]], dtype=np.uint16)
    write_depth(tmp_path / row["depth"], target)

    collection = load_kitti(manifest)
    sample = collection.samples[0]
    layout = RunLayout(tmp_path / "run")
    save_prediction(layout.prediction_path(sample), np.array([[1.0, 1.0, 81.0]]))

    result = run_depth_evaluation(collection, layout)

    with (layout.metrics_dir / "per_sample.csv").open("r", encoding="utf-8") as file:
        assert ",2," in file.read()
    assert result["summary"]["overall"]["mae"] == 0.0


def test_kitti_evaluation_counts_finite_nonpositive_predictions(tmp_path):
    name = "2011_09_26_drive_0005_sync_image_0000000128_image_02"
    manifest = tmp_path / "val.jsonl"
    row = kitti_row(name)
    write_jsonl(manifest, [row])
    target = np.full((1, 4), 256, dtype=np.uint16)
    write_depth(tmp_path / row["depth"], target)

    collection = load_kitti(manifest)
    sample = collection.samples[0]
    layout = RunLayout(tmp_path / "run")
    save_prediction(
        layout.prediction_path(sample),
        np.array([[0.0, -1.0, np.nan, np.inf]], dtype=np.float32),
    )

    result = run_depth_evaluation(collection, layout)

    with (layout.metrics_dir / "per_sample.csv").open("r", encoding="utf-8") as file:
        rows = file.read().splitlines()
    assert rows[1].split(",")[2] == "2"
    assert math.isclose(result["summary"]["overall"]["mae"], 1.5)


def test_kitti_prediction_normalization_preserves_nonpositive_and_nonfinite_values():
    prediction = np.array([[0.0, -1.0, np.nan, np.inf]], dtype=np.float64)

    normalized = normalize_asdepth_prediction(prediction, prediction.shape)

    np.testing.assert_array_equal(normalized[:, :2], np.array([[0.0, -1.0]], dtype=np.float32))
    assert np.isnan(normalized[0, 2])
    assert np.isposinf(normalized[0, 3])
