import json
from pathlib import Path

from evaluation.datasets.clearpose import load_clearpose
from evaluation.datasets.dreds import load_dreds
from evaluation.datasets.hammer import load_hammer
from evaluation.datasets.ibims import IBIMS_DEPTH_SCALE, load_ibims, manifest_for_level


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def test_hammer_adapter_uses_explicit_camera_and_relative_sample_id(tmp_path):
    manifest = tmp_path / "hammer" / "test.jsonl"
    row = {
        "rgb": "scene/rgb/frame.png",
        "depth": "scene/gt/frame.png",
        "d435_depth": "scene/d435/frame.png",
        "l515_depth": "scene/l515/frame.png",
        "tof_depth": "scene/tof/frame.png",
        "depth-range": [0.1, 6.0],
    }
    write_jsonl(manifest, [row])

    collection = load_hammer(manifest, "l515")
    sample = collection.samples[0]

    assert collection.name == "hammer"
    assert sample.subset == "l515"
    assert sample.sample_id == "scene/rgb/frame"
    assert sample.raw_depth_path == (manifest.parent / row["l515_depth"]).resolve()
    assert sample.depth_scale == 1000.0


def test_sequence_adapters_pair_frames_by_suffix(tmp_path):
    clear_manifest = tmp_path / "clearpose" / "test.jsonl"
    clear_sequence = clear_manifest.parent / "sequence_a"
    for name in ("000_color.png", "000_raw.png", "000_gt.png"):
        touch(clear_sequence / name)
    write_jsonl(
        clear_manifest,
        [
            {
                "rgb": "sequence_a",
                "rgb-suffix": "_color.png",
                "raw_depth-suffix": "_raw.png",
                "depth-suffix": "_gt.png",
                "depth-range": [0.1, 5.0],
            }
        ],
    )

    clearpose = load_clearpose(clear_manifest)
    assert clearpose.samples[0].sample_id == "sequence_a/000_color"
    assert clearpose.samples[0].subset == "default"

    dreds_manifest = tmp_path / "dreds" / "known.jsonl"
    dreds_sequence = dreds_manifest.parent / "sequence_b"
    for name in ("001_color.png", "001_raw.exr", "001_gt.exr"):
        touch(dreds_sequence / name)
    write_jsonl(
        dreds_manifest,
        [
            {
                "rgb": "sequence_b",
                "rgb-suffix": "_color.png",
                "raw_depth-suffix": "_raw.exr",
                "depth-suffix": "_gt.exr",
                "depth-range": [0.1, 10.0],
            }
        ],
    )

    dreds = load_dreds({"catknown": dreds_manifest}, ["catknown"])
    assert dreds.samples[0].subset == "catknown"
    assert dreds.samples[0].depth_scale == 1.0
    assert dreds.samples[0].allow_evaluation_resize is True


def test_max_samples_is_applied_per_subset(tmp_path):
    manifests = {}
    for variant in ("catknown", "catnovel"):
        manifest = tmp_path / variant / "test.jsonl"
        sequence = manifest.parent / "sequence"
        for index in range(3):
            for suffix in ("_color.png", "_raw.exr", "_gt.exr"):
                touch(sequence / f"{index:03d}{suffix}")
        write_jsonl(
            manifest,
            [
                {
                    "rgb": "sequence",
                    "rgb-suffix": "_color.png",
                    "raw_depth-suffix": "_raw.exr",
                    "depth-suffix": "_gt.exr",
                    "depth-range": [0.1, 10.0],
                }
            ],
        )
        manifests[variant] = manifest

    collection = load_dreds(manifests, ["catknown", "catnovel"], max_samples=1)
    assert len(collection.samples) == 2
    assert collection.subsets == ["catknown", "catnovel"]


def test_ibims_adapter_reads_level_manifests(tmp_path):
    root = tmp_path / "ibims1"
    manifest = manifest_for_level(root, "easy")
    write_jsonl(
        manifest,
        [
            {
                "dataset": "ibims",
                "difficulty": "easy",
                "sample_id": "living_room_01",
                "rgb": "../rgb/living_room_01.png",
                "raw_depth": "../raw/living_room_01.png",
                "depth-range": [0.01, 50.0],
            }
        ],
    )

    collection = load_ibims(root, ["easy"])
    sample = collection.samples[0]
    assert sample.sample_id == "living_room_01"
    assert sample.subset == "easy"
    assert sample.depth_scale == IBIMS_DEPTH_SCALE
    assert sample.expected_shape == (480, 640)
