from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from lingbot_realtime.deployment import (
    FP32_STABILITY_POLICY,
    FixedDeploymentWrapper,
    Resolution,
    build_manifest,
    convert_onnx_fp16,
    inspect_onnx,
    validate_manifest,
)
from lingbot_realtime.deployment.tensorrt import build_tensorrt_command


def test_strongly_typed_build_command_has_no_legacy_fp16_flag(tmp_path) -> None:
    command = build_tensorrt_command(
        tmp_path / "model.fp16.onnx",
        tmp_path / "model.engine",
        timing_cache_path=tmp_path / "timing.cache",
    )
    assert "--stronglyTyped" in command
    assert "--fp16" not in command


def test_deployment_manifest_records_fixed_fp16_contract(tmp_path) -> None:
    artifact = tmp_path / "model.fp16.onnx"
    artifact.write_bytes(b"onnx")
    manifest = build_manifest(
        tmp_path,
        model_source="local/model.pt",
        checkpoint_sha256="1" * 64,
        artifacts={"onnx_fp16": artifact},
    )
    validate_manifest(manifest)
    assert manifest["num_tokens"] == 1200
    assert manifest["dynamic_depth_token_mask"] is False
    assert manifest["static_depth_attention_mask"] is True
    assert manifest["fp32_stability_policy"] == FP32_STABILITY_POLICY
    assert manifest["tensor_contract"]["input"]["semantic"].startswith("rgb_0_1")
    assert json.dumps(manifest)


def test_resolution_is_fixed_for_first_release() -> None:
    assert Resolution.parse("480x640").output_shape == (1, 480, 640)


def test_export_wrapper_disables_dynamic_token_deletion_and_preserves_mask() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs = None

        def forward(self, image, *, num_tokens, depth, **kwargs):
            self.kwargs = {"num_tokens": num_tokens, **kwargs}
            prediction = torch.ones((1, image.shape[-2], image.shape[-1]))
            mask = torch.ones_like(prediction)
            mask[:, 0, 0] = 0.0
            return {"depth_reg": prediction, "mask": mask}

    model = Model()
    output = FixedDeploymentWrapper(model)(torch.ones((1, 4, 2, 3)))
    assert model.kwargs == {
        "num_tokens": 1200,
        "enable_depth_mask": False,
        "static_depth_attention_mask": True,
    }
    assert output.shape == (1, 2, 3)
    assert output[0, 0, 0] == 0
    assert output[0, 0, 1] == 1


def test_static_attention_mask_matches_deleted_token_outputs() -> None:
    from mdm.model.dinov2_rgbd.layers.block import Block

    torch.manual_seed(7)
    block = Block(dim=8, num_heads=2, init_values=0.1).eval()
    tokens = torch.randn(1, 7, 8)
    valid = torch.tensor([True, True, True, False, True, False, True])
    retained = torch.where(valid)[0]

    dense = block(tokens, attn_bias=valid[None, None, None, :])
    deleted = block(tokens[:, retained])

    torch.testing.assert_close(dense[:, retained], deleted, rtol=1e-5, atol=1e-6)


def test_fp16_conversion_freezes_symbolic_where_style_output_shape(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    graph = helper.make_graph(
        [
            helper.make_node(
                "ConstantOfShape",
                ["shape"],
                ["depth"],
                value=helper.make_tensor("value", onnx.TensorProto.FLOAT, [1], [1.0]),
            )
        ],
        "fixed-output",
        [helper.make_tensor_value_info("rgbd_input", onnx.TensorProto.FLOAT, [1, 4, 480, 640])],
        [helper.make_tensor_value_info("depth", onnx.TensorProto.FLOAT, [1, "H", "W"])],
        initializer=[helper.make_tensor("shape", onnx.TensorProto.INT64, [3], [1, 480, 640])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, source)
    convert_onnx_fp16(source, target)
    inspection = inspect_onnx(target, dtype="FLOAT16")
    assert inspection["outputs"][0]["shape"] == [1, 480, 640]


def test_fp16_conversion_keeps_transformer_stability_chain_in_fp32(tmp_path) -> None:
    onnx = pytest.importorskip("onnx")
    helper = onnx.helper
    source = tmp_path / "source.onnx"
    target = tmp_path / "target.onnx"
    add_name = "/model/encoder/blocks.0/attn/qkv/Add"
    norm_name = "/model/encoder/blocks.0/norm1/LayerNormalization"
    residual_name = "/model/encoder/blocks.0/Add"
    graph = helper.make_graph(
        [
            helper.make_node("Gather", ["rgbd_input", "channel"], ["plane"], axis=1),
            helper.make_node("Add", ["plane", "offset"], ["added"], name=add_name),
            helper.make_node(
                "LayerNormalization",
                ["added", "scale", "bias"],
                ["normalized"],
                name=norm_name,
                axis=-1,
            ),
            helper.make_node(
                "Add", ["added", "normalized"], ["depth"], name=residual_name
            ),
        ],
        "mixed-precision-chain",
        [helper.make_tensor_value_info("rgbd_input", onnx.TensorProto.FLOAT, [1, 4, 480, 640])],
        [helper.make_tensor_value_info("depth", onnx.TensorProto.FLOAT, [1, 480, 640])],
        initializer=[
            helper.make_tensor("channel", onnx.TensorProto.INT64, [], [0]),
            helper.make_tensor("offset", onnx.TensorProto.FLOAT, [], [0.25]),
            helper.make_tensor(
                "scale",
                onnx.TensorProto.FLOAT,
                [640],
                np.ones(640, dtype=np.float32).tobytes(),
                raw=True,
            ),
            helper.make_tensor(
                "bias",
                onnx.TensorProto.FLOAT,
                [640],
                np.zeros(640, dtype=np.float32).tobytes(),
                raw=True,
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx.save(model, source)

    convert_onnx_fp16(source, target)

    converted = onnx.load(target, load_external_data=False)
    onnx.checker.check_model(str(target), full_check=False)
    outputs = [output for node in converted.graph.node for output in node.output if output]
    assert len(outputs) == len(set(outputs))
    by_name = {node.name: node for node in converted.graph.node}
    fp32_add_output = by_name[add_name].output[0]
    assert fp32_add_output.endswith("_cast_to_fp16")
    assert by_name[norm_name].input[0] == fp32_add_output
    assert fp32_add_output in by_name[residual_name].input
    assert converted.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert converted.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16

    external = target.with_name(target.name + ".data")
    first_size = external.stat().st_size
    convert_onnx_fp16(source, target, overwrite=True)
    assert external.stat().st_size == first_size
