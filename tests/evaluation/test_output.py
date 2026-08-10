import json
from pathlib import Path

import numpy as np

from evaluation.core.output import RunLayout, sample_relative_path, write_json
from evaluation.core.types import EvaluationSample


def make_sample():
    return EvaluationSample(
        sample_id="scene/sequence/frame_001",
        subset="catknown",
        rgb_path=Path("rgb.png"),
        raw_depth_path=Path("raw.exr"),
        gt_depth_path=Path("gt.exr"),
        depth_scale=1.0,
        min_depth=0.1,
        max_depth=10.0,
    )


def test_artifacts_preserve_relative_sample_hierarchy(tmp_path):
    sample = make_sample()
    layout = RunLayout(tmp_path)

    assert sample_relative_path(sample, ".npy") == Path("catknown/scene/sequence/frame_001.npy")
    assert layout.visualization_path(sample) == tmp_path / Path(
        "visualizations/catknown/scene/sequence/frame_001_vis.jpg"
    )


def test_json_writes_atomically_and_serializes_paths(tmp_path):
    path = tmp_path / "run.json"
    write_json(path, {"path": tmp_path, "value": np.float32(1.5)})
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    assert value["path"] == str(tmp_path)
    assert value["value"] == 1.5
