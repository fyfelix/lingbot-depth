import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.core.output import RunLayout
from evaluation.core.pipeline import run_pipeline
from evaluation.core.types import EvaluationSample, RunConfig
from evaluation.datasets.base import DatasetCollection


class FakeModel:
    def infer(self, _image, depth_in, **_kwargs):
        return {"depth": depth_in.clone()}


def write_rgb(path: Path, shape=(4, 5)):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((*shape, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def write_depth(path: Path, shape=(4, 5)):
    path.parent.mkdir(parents=True, exist_ok=True)
    depth = np.full(shape, 1000, dtype=np.uint16)
    assert cv2.imwrite(str(path), depth)


def test_pipeline_smoke_with_mock_model(tmp_path, monkeypatch):
    rgb_path = tmp_path / "data/rgb.png"
    raw_path = tmp_path / "data/raw.png"
    gt_path = tmp_path / "data/gt.png"
    write_rgb(rgb_path)
    write_depth(raw_path)
    write_depth(gt_path)

    sample = EvaluationSample(
        sample_id="scene/frame",
        subset="d435",
        rgb_path=rgb_path,
        raw_depth_path=raw_path,
        gt_depth_path=gt_path,
        depth_scale=1000.0,
        min_depth=0.1,
        max_depth=5.0,
    )
    collection = DatasetCollection(name="hammer", samples=[sample])
    run_dir = tmp_path / "outputs/run"
    config = RunConfig(
        dataset="hammer",
        stage="all",
        run_dir=run_dir,
        model_path="fake-model",
        device="cpu",
        save_visualizations=True,
    )
    monkeypatch.setattr("evaluation.core.inference.load_model", lambda *_args: FakeModel())

    layout = run_pipeline(collection, config)

    assert layout == RunLayout(run_dir.resolve())
    assert layout.prediction_path(sample).is_file()
    assert layout.visualization_path(sample).is_file()
    with layout.metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)
    assert metadata["status"] == "completed"
    assert metadata["results"]["evaluation"]["summary"]["overall"]["rmse"] == 0.0
