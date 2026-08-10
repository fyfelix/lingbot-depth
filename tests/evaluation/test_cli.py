from pathlib import Path

import pytest

from evaluation.cli import build_parser, resolve_run_dir


def test_cli_exposes_dataset_subcommands():
    parser = build_parser()
    args = parser.parse_args(
        [
            "hammer",
            "--manifest",
            "data/HAMMER/test.jsonl",
            "--model-path",
            "model.pt",
            "--camera",
            "tof",
        ]
    )
    assert args.dataset == "hammer"
    assert args.camera == "tof"
    assert args.save_visualizations is True

    kitti = parser.parse_args(
        [
            "kitti",
            "--manifest",
            "data/kitti.jsonl",
            "--model-path",
            "model.pt",
        ]
    )
    assert kitti.dataset == "kitti"
    assert kitti.raw_max_depth == 80.0
    assert kitti.visualization_max_depth == 80.0
    assert kitti.pointcloud_knn_k == 16


def test_evaluate_only_requires_explicit_run_dir():
    parser = build_parser()
    args = parser.parse_args(
        [
            "clearpose",
            "--manifest",
            "data/clearpose/test.jsonl",
            "--stage",
            "evaluate",
        ]
    )
    with pytest.raises(SystemExit):
        resolve_run_dir(args, parser)


def test_explicit_run_dir_is_preserved():
    parser = build_parser()
    args = parser.parse_args(
        [
            "ibims",
            "--stage",
            "evaluate",
            "--run-dir",
            "outputs/evaluation/ibims/existing",
        ]
    )
    assert resolve_run_dir(args, parser) == Path("outputs/evaluation/ibims/existing")
