from __future__ import annotations

import tempfile
from heapq import heapify, heappop, heappush
from pathlib import Path
from typing import Any

import torch

from .contracts import DEFAULT_OPSET, INPUT_NAME, OUTPUT_NAME, Resolution


class FixedDeploymentWrapper(torch.nn.Module):
    """Fixed 1200-token graph with depth-token deletion disabled and mask retained."""

    def __init__(self, model: torch.nn.Module, *, num_tokens: int = 1200) -> None:
        super().__init__()
        self.model = model
        self.num_tokens = int(num_tokens)

    def forward(self, rgbd_input: torch.Tensor) -> torch.Tensor:
        image = rgbd_input[:, :3]
        depth = rgbd_input[:, 3]
        output = self.model(
            image,
            num_tokens=self.num_tokens,
            depth=depth,
            enable_depth_mask=False,
            static_depth_attention_mask=True,
        )
        prediction = output["depth_reg"]
        mask = output.get("mask")
        if mask is not None:
            prediction = torch.where(mask > 0.5, prediction, torch.zeros_like(prediction))
        return prediction


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export requires the deploy extra") from exc
    return onnx


def _guard(path: Path, overwrite: bool) -> None:
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(path)
    path.unlink()


def export_onnx_fp32(
    model: torch.nn.Module,
    output_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    resolution: Resolution = Resolution(),
    num_tokens: int = 1200,
    opset: int = DEFAULT_OPSET,
    overwrite: bool = False,
) -> Path:
    onnx = _require_onnx()
    target = Path(output_path).expanduser().resolve()
    external = target.with_name(target.name + ".data")
    _guard(target, overwrite)
    _guard(external, overwrite)
    target.parent.mkdir(parents=True, exist_ok=True)
    active_device = torch.device(device)
    model = model.to(active_device).float().eval()
    encoder = getattr(model, "encoder", None)
    if encoder is not None and hasattr(encoder, "onnx_compatible_mode"):
        encoder.onnx_compatible_mode = True
    wrapper = FixedDeploymentWrapper(model, num_tokens=num_tokens).eval()
    dummy = torch.zeros(resolution.input_shape, dtype=torch.float32, device=active_device)
    dummy[:, :3] = 0.5
    dummy[:, 3] = 1.0
    with tempfile.TemporaryDirectory(prefix="lingbot-onnx-") as temporary:
        temporary_path = Path(temporary) / "model.onnx"
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(temporary_path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )
        exported = onnx.load(str(temporary_path), load_external_data=True)
    for value in exported.graph.input:
        if value.name == INPUT_NAME:
            _freeze_value_shape(value, resolution.input_shape)
    for value in exported.graph.output:
        if value.name == OUTPUT_NAME:
            _freeze_value_shape(value, resolution.output_shape)
    from onnx.external_data_helper import convert_model_to_external_data

    convert_model_to_external_data(
        exported,
        all_tensors_to_one_file=True,
        location=external.name,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save_model(exported, str(target))
    inspect_onnx(target, resolution=resolution, dtype="FLOAT")
    return target


def _freeze_value_shape(value: Any, shape: tuple[int, ...]) -> None:
    tensor_shape = value.type.tensor_type.shape
    while len(tensor_shape.dim) < len(shape):
        tensor_shape.dim.add()
    while len(tensor_shape.dim) > len(shape):
        del tensor_shape.dim[-1]
    for dimension, size in zip(tensor_shape.dim, shape, strict=True):
        dimension.ClearField("dim_param")
        dimension.dim_value = int(size)


def _fp32_stability_nodes(model: Any) -> set[str]:
    """Keep numerically sensitive transformer state in FP32 while GEMMs stay FP16."""

    selected: set[str] = set()
    for node in model.graph.node:
        in_encoder_block = node.name.startswith("/model/encoder/blocks.")
        layer_scale = in_encoder_block and (
            "/ls1/" in node.name or "/ls2/" in node.name
        )
        if (
            node.op_type in {"LayerNormalization", "Softmax"}
            or (in_encoder_block and node.op_type == "Add")
            or (layer_scale and node.op_type == "Mul")
        ):
            selected.add(node.name)
    return selected


def _connect_blocked_chains(model: Any, blocked: set[str]) -> None:
    """Avoid a lossy FP32 -> FP16 -> FP32 round trip between blocked nodes."""

    fp32_outputs: dict[str, str] = {}
    for node in model.graph.node:
        if node.name not in blocked:
            continue
        for output in node.output:
            if output.endswith("_cast_to_fp16"):
                fp32_outputs[output.removesuffix("_cast_to_fp16")] = output

    for node in model.graph.node:
        if node.name not in blocked:
            continue
        for index, input_name in enumerate(node.input):
            if not input_name.endswith("_cast_to_fp32"):
                continue
            original = input_name.removesuffix("_cast_to_fp32")
            if original in fp32_outputs:
                node.input[index] = fp32_outputs[original]


def _deduplicate_node_outputs(graph: Any) -> None:
    """Collapse identical casts emitted for a tensor with multiple blocked consumers."""

    producers: dict[str, Any] = {}
    retained = []
    for node in graph.node:
        collisions = [producers[name] for name in node.output if name and name in producers]
        if collisions:
            serialized = node.SerializeToString()
            if any(existing.SerializeToString() != serialized for existing in collisions):
                names = [name for name in node.output if name and name in producers]
                raise ValueError(f"FP16 conversion produced conflicting tensor outputs: {names}")
            continue
        retained.append(node)
        for name in node.output:
            if name:
                producers[name] = node
    del graph.node[:]
    graph.node.extend(retained)


def _topological_sort_graph(graph: Any) -> None:
    """Restore ONNX node order after the converter appends boundary casts."""

    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("g"):
                _topological_sort_graph(attribute.g)
            for child in attribute.graphs:
                _topological_sort_graph(child)

    nodes = list(graph.node)
    producer = {
        output: index for index, node in enumerate(nodes) for output in node.output if output
    }
    dependencies: list[set[int]] = []
    consumers: list[list[int]] = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        parents = {producer[name] for name in node.input if name in producer}
        parents.discard(index)
        dependencies.append(parents)
        for parent in parents:
            consumers[parent].append(index)

    ready = [index for index, parents in enumerate(dependencies) if not parents]
    heapify(ready)
    ordered: list[int] = []
    while ready:
        index = heappop(ready)
        ordered.append(index)
        for child in consumers[index]:
            dependencies[child].discard(index)
            if not dependencies[child]:
                heappush(ready, child)
    if len(ordered) != len(nodes):
        raise ValueError("FP16 conversion produced a cyclic ONNX graph")
    del graph.node[:]
    graph.node.extend(nodes[index] for index in ordered)


def convert_onnx_fp16(
    source_path: str | Path,
    output_path: str | Path,
    *,
    resolution: Resolution = Resolution(),
    overwrite: bool = False,
) -> Path:
    try:
        import onnx
        from onnx.external_data_helper import convert_model_to_external_data
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as exc:
        raise RuntimeError("FP16 conversion requires onnx and onnxruntime") from exc
    source = Path(source_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    external = target.with_name(target.name + ".data")
    if not source.is_file():
        raise FileNotFoundError(source)
    _guard(target, overwrite)
    _guard(external, overwrite)
    model_metadata = onnx.load(str(source), load_external_data=False)
    blocked = _fp32_stability_nodes(model_metadata)
    converted = convert_float_to_float16(
        str(source),
        keep_io_types=False,
        disable_shape_infer=False,
        node_block_list=sorted(blocked),
        force_fp16_initializers=False,
    )
    _connect_blocked_chains(converted, blocked)
    _deduplicate_node_outputs(converted.graph)
    _topological_sort_graph(converted.graph)
    for value in converted.graph.input:
        if value.name == INPUT_NAME:
            _freeze_value_shape(value, resolution.input_shape)
    for value in converted.graph.output:
        if value.name == OUTPUT_NAME:
            _freeze_value_shape(value, resolution.output_shape)
    convert_model_to_external_data(
        converted,
        all_tensors_to_one_file=True,
        location=external.name,
        size_threshold=1024,
        convert_attribute=True,
    )
    onnx.save_model(converted, str(target))
    inspect_onnx(target, resolution=resolution, dtype="FLOAT16")
    return target


def _shape(value: Any) -> list[int | str]:
    output: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        output.append(
            int(dim.dim_value) if dim.HasField("dim_value") else str(dim.dim_param or "?")
        )
    return output


def inspect_onnx(
    path: str | Path,
    *,
    resolution: Resolution = Resolution(),
    dtype: str = "FLOAT",
) -> dict[str, Any]:
    onnx = _require_onnx()
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    try:
        onnx.checker.check_model(str(target), full_check=False)
    except Exception as exc:
        raise ValueError(f"invalid ONNX graph {target}: {exc}") from exc
    model = onnx.load(str(target), load_external_data=False)
    initializer_names = {item.name for item in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializer_names]
    outputs = list(model.graph.output)
    records_in = [
        {
            "name": v.name,
            "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
            "shape": _shape(v),
        }
        for v in inputs
    ]
    records_out = [
        {
            "name": v.name,
            "dtype": onnx.TensorProto.DataType.Name(v.type.tensor_type.elem_type),
            "shape": _shape(v),
        }
        for v in outputs
    ]
    expected_in = [{"name": INPUT_NAME, "dtype": dtype, "shape": list(resolution.input_shape)}]
    expected_out = [{"name": OUTPUT_NAME, "dtype": dtype, "shape": list(resolution.output_shape)}]
    if records_in != expected_in or records_out != expected_out:
        raise ValueError(f"ONNX tensor contract mismatch: {records_in}, {records_out}")
    return {"inputs": records_in, "outputs": records_out, "nodes": len(model.graph.node)}
