from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import torch

from lingbot_realtime.deployment.contracts import DEFAULT_OPSET, Resolution
from lingbot_realtime.deployment.manifest import build_manifest, sha256_file, write_manifest
from lingbot_realtime.deployment.onnx import convert_onnx_fp16, export_onnx_fp32, inspect_onnx
from lingbot_realtime.deployment.tensorrt import (
    TensorRTBuildConfig,
    benchmark_engine,
    build_tensorrt,
    probe_trtexec,
)


def _device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=value, repo_type="model", filename="model.pt")).resolve()


def _paths(root: str | Path) -> dict[str, Path]:
    base = Path(root).expanduser().resolve()
    return {
        "root": base,
        "fp32": base / "model.fp32.onnx",
        "fp16": base / "model.fp16.onnx",
        "engine": base / "model.engine",
        "timing": base / "timing.cache",
        "log": base / "build.log",
        "manifest": base / "deployment.json",
    }


def _export(args: argparse.Namespace) -> dict[str, Any]:
    import onnx
    import onnxruntime
    from mdm.model.v2 import MDMModel

    paths = _paths(args.output)
    checkpoint = _resolve_checkpoint(args.model)
    model = MDMModel.from_pretrained(checkpoint)
    export_onnx_fp32(
        model,
        paths["fp32"],
        device=_device(args.device),
        resolution=Resolution.parse(args.resolution),
        num_tokens=args.num_tokens,
        opset=args.opset,
        overwrite=args.overwrite,
    )
    convert_onnx_fp16(
        paths["fp32"],
        paths["fp16"],
        resolution=Resolution.parse(args.resolution),
        overwrite=args.overwrite,
    )
    manifest = build_manifest(
        paths["root"],
        model_source=args.model,
        checkpoint_sha256=sha256_file(checkpoint),
        artifacts={
            "onnx_fp32": paths["fp32"],
            "onnx_fp32_data": Path(str(paths["fp32"]) + ".data"),
            "onnx_fp16": paths["fp16"],
            "onnx_fp16_data": Path(str(paths["fp16"]) + ".data"),
        },
        versions={
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": onnxruntime.__version__,
            "onnx_opset": args.opset,
        },
        resolution=Resolution.parse(args.resolution),
        num_tokens=args.num_tokens,
    )
    write_manifest(paths["manifest"], manifest, overwrite=args.overwrite)
    return {
        "fp32_onnx": str(paths["fp32"]),
        "fp16_onnx": str(paths["fp16"]),
        "manifest": str(paths["manifest"]),
    }


def _gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    return {
        "available": True,
        "name": torch.cuda.get_device_name(index),
        "capability": list(capability),
    }


def _build(args: argparse.Namespace, *, benchmark: bool = False) -> dict[str, Any]:
    paths = _paths(args.output)
    source = Path(args.onnx).expanduser().resolve() if args.onnx else paths["fp16"]
    inspect_onnx(source, resolution=Resolution.parse(args.resolution), dtype="FLOAT16")
    config = TensorRTBuildConfig(
        trtexec_path=args.trtexec,
        workspace_mib=args.workspace_mib,
        builder_optimization_level=args.builder_optimization_level,
        profiling_verbosity=args.profiling_verbosity,
    )
    built = build_tensorrt(
        source,
        paths["engine"],
        timing_cache_path=paths["timing"],
        log_path=paths["log"],
        config=config,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return {"command": built}
    trtexec = probe_trtexec(args.trtexec)
    smoke = (
        benchmark_engine(
            paths["engine"], trtexec_path=args.trtexec, duration_sec=args.benchmark_duration
        )
        if benchmark
        else None
    )
    previous: dict[str, Any] = {}
    if paths["manifest"].is_file():
        previous = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    model = previous.get("model", {})
    artifacts = {
        "onnx_fp16": source,
        "engine": paths["engine"],
        "timing_cache": paths["timing"],
        "build_log": paths["log"],
    }
    for name in ("fp32", "fp16"):
        path = paths[name]
        if path.is_file():
            artifacts[f"onnx_{name}"] = path
            external = Path(str(path) + ".data")
            if external.is_file():
                artifacts[f"onnx_{name}_data"] = external
    versions = dict(previous.get("versions", {}))
    versions.update(
        {"torch": torch.__version__, "tensorrt": trtexec, "onnx_opset": DEFAULT_OPSET}
    )
    manifest = build_manifest(
        paths["root"],
        model_source=str(model.get("source", args.model or "unknown")),
        checkpoint_sha256=str(model.get("checkpoint_sha256", args.checkpoint_sha256 or "unknown")),
        artifacts=artifacts,
        versions=versions,
        gpu=_gpu_info(),
        benchmark=smoke,
        resolution=Resolution.parse(args.resolution),
        num_tokens=args.num_tokens,
    )
    manifest["tensorrt_major"] = trtexec["major"]
    write_manifest(paths["manifest"], manifest, overwrite=True)
    return {"engine": str(paths["engine"]), "manifest": str(paths["manifest"]), "benchmark": smoke}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lingbot-realtime-deploy",
        description="Export fixed FP16 ONNX and build a strongly-typed TensorRT engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common_export(target: argparse.ArgumentParser) -> None:
        target.add_argument(
            "--model", required=True, help="Hugging Face model id or local model.pt"
        )
        target.add_argument("--output", required=True)
        target.add_argument("--resolution", default="480x640")
        target.add_argument("--num-tokens", type=int, default=1200)
        target.add_argument("--opset", type=int, default=DEFAULT_OPSET)
        target.add_argument("--device", default="auto")
        target.add_argument("--overwrite", action="store_true")

    def common_build(target: argparse.ArgumentParser) -> None:
        target.add_argument("--output", required=True)
        target.add_argument("--onnx")
        target.add_argument("--model")
        target.add_argument("--checkpoint-sha256")
        target.add_argument("--resolution", default="480x640")
        target.add_argument("--num-tokens", type=int, default=1200)
        target.add_argument("--trtexec", default="trtexec")
        target.add_argument("--workspace-mib", type=int, default=8192)
        target.add_argument("--builder-optimization-level", type=int, default=5)
        target.add_argument(
            "--profiling-verbosity",
            choices=("none", "layer_names_only", "detailed"),
            default="detailed",
        )
        target.add_argument("--benchmark-duration", type=int, default=3)
        target.add_argument("--dry-run", action="store_true")
        target.add_argument("--overwrite", action="store_true")

    common_export(sub.add_parser("export"))
    common_build(sub.add_parser("build"))
    all_parser = sub.add_parser("all")
    common_export(all_parser)
    all_parser.add_argument("--trtexec", default="trtexec")
    all_parser.add_argument("--workspace-mib", type=int, default=8192)
    all_parser.add_argument("--builder-optimization-level", type=int, default=5)
    all_parser.add_argument(
        "--profiling-verbosity",
        choices=("none", "layer_names_only", "detailed"),
        default="detailed",
    )
    all_parser.add_argument("--benchmark-duration", type=int, default=3)
    all_parser.add_argument("--dry-run", action="store_true")
    all_parser.set_defaults(onnx=None, checkpoint_sha256=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = _export(args)
        elif args.command == "build":
            result = _build(args)
        else:
            exported = _export(args)
            result = {**exported, **_build(args, benchmark=True)}
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
