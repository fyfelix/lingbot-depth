from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from evaluation.core.output import RunLayout, save_prediction
from evaluation.core.types import EvaluationSample
from evaluation.datasets.base import DatasetCollection
from evaluation.evaluators.ibims_official import run_ibims_official_evaluation


def test_ibims_official_export_workspace_and_summary(tmp_path):
    sample = EvaluationSample(
        sample_id="sample_01",
        subset="easy",
        rgb_path=Path("unused_rgb.png"),
        raw_depth_path=Path("unused_raw.png"),
        gt_depth_path=None,
        depth_scale=1.0,
        min_depth=0.01,
        max_depth=50.0,
        expected_shape=(480, 640),
    )
    collection = DatasetCollection(name="ibims", samples=[sample])
    layout = RunLayout(tmp_path / "run")
    prediction = np.ones((480, 640), dtype=np.float32)
    prediction[0, 0] = np.nan
    save_prediction(layout.prediction_path(sample), prediction)

    ibims_root = tmp_path / "ibims1"
    gt_dir = ibims_root / "ibims1_core_mat"
    script_dir = ibims_root / "evaluation_scripts"
    gt_dir.mkdir(parents=True)
    script_dir.mkdir(parents=True)
    savemat(gt_dir / "sample_01.mat", {"depth": np.ones((1, 1), dtype=np.float32)})
    (script_dir / "evaluate_ibims.py").write_text(
        "print('Results:')\nprint('rel = 0.1')\nprint('rms = 0.2')\n",
        encoding="utf-8",
    )

    result = run_ibims_official_evaluation(collection, layout, ibims_root)

    mat_path = layout.official_prediction_dir("easy") / "sample_01_results.mat"
    assert mat_path.is_file()
    assert np.isnan(loadmat(mat_path)["pred_depths"][0, 0])
    assert result["summary"]["easy"]["rel"] == 0.1
    assert (layout.metrics_dir / "summary.csv").is_file()
    assert (layout.official_log_path("easy")).is_file()
