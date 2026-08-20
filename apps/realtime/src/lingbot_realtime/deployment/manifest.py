from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from .contracts import FP32_STABILITY_POLICY, INPUT_NAME, OUTPUT_NAME, Resolution


def sha256_file(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(root.resolve())),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_manifest(
    root: str | Path,
    *,
    model_source: str,
    checkpoint_sha256: str,
    artifacts: dict[str, str | Path],
    versions: dict[str, Any] | None = None,
    gpu: dict[str, Any] | None = None,
    benchmark: dict[str, Any] | None = None,
    resolution: Resolution = Resolution(),
    num_tokens: int = 1200,
) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    return {
        "schema": "lingbot.realtime.deployment.v1",
        "model": {"source": model_source, "checkpoint_sha256": checkpoint_sha256},
        "precision": "fp16",
        "fp32_stability_policy": FP32_STABILITY_POLICY,
        "resolution_level": 0,
        "num_tokens": int(num_tokens),
        "dynamic_depth_token_mask": False,
        "static_depth_attention_mask": True,
        "prediction_mask": True,
        "tensor_contract": {
            "input": {
                "name": INPUT_NAME,
                "dtype": "float16",
                "shape": list(resolution.input_shape),
                "layout": "NCHW",
                "semantic": "rgb_0_1_plus_metric_depth_meter_invalid_zero",
            },
            "output": {
                "name": OUTPUT_NAME,
                "dtype": "float16",
                "shape": list(resolution.output_shape),
                "semantic": "metric_depth_meter_invalid_zero",
            },
        },
        "versions": dict(versions or {}),
        "gpu": dict(gpu or {}),
        "benchmark": benchmark,
        "platform": {"python": platform.python_version(), "machine": platform.machine()},
        "artifacts": {
            name: _artifact(Path(path).expanduser().resolve(), base)
            for name, path in artifacts.items()
            if Path(path).expanduser().resolve().is_file()
        },
    }


def write_manifest(path: str | Path, value: dict[str, Any], *, overwrite: bool = False) -> Path:
    target = Path(path).expanduser().resolve()
    if target.exists() and not overwrite:
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)
    return target


def validate_manifest(value: dict[str, Any]) -> None:
    if value.get("schema") != "lingbot.realtime.deployment.v1":
        raise ValueError("unsupported deployment manifest schema")
    if value.get("precision") != "fp16":
        raise ValueError("deployment precision must be fp16")
    contract = value.get("tensor_contract", {})
    if contract.get("input", {}).get("shape") != [1, 4, 480, 640]:
        raise ValueError("deployment input shape must be [1,4,480,640]")
    if contract.get("output", {}).get("shape") != [1, 480, 640]:
        raise ValueError("deployment output shape must be [1,480,640]")
