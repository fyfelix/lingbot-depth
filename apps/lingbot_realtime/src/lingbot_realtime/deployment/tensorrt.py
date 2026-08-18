from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def tensorrt_major_from_version(value: str) -> int:
    normalized = value.strip()
    if "." in normalized and normalized.split(".", 1)[0].isdigit():
        return int(normalized.split(".", 1)[0])
    if normalized.isdigit():
        return int(normalized[:2]) if len(normalized) >= 6 else int(normalized[0])
    raise ValueError(f"unable to parse TensorRT version: {value!r}")


@dataclass(frozen=True, slots=True)
class TensorRTBuildConfig:
    trtexec_path: str = "trtexec"
    workspace_mib: int = 8192
    builder_optimization_level: int = 5
    profiling_verbosity: str = "detailed"

    def __post_init__(self) -> None:
        if self.workspace_mib <= 0:
            raise ValueError("workspace_mib must be positive")
        if not 0 <= self.builder_optimization_level <= 5:
            raise ValueError("builder optimization level must be in [0,5]")


def build_tensorrt_command(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    timing_cache_path: str | Path,
    config: TensorRTBuildConfig | None = None,
) -> list[str]:
    active = config or TensorRTBuildConfig()
    return [
        active.trtexec_path,
        f"--onnx={Path(onnx_path).expanduser().resolve()}",
        f"--saveEngine={Path(engine_path).expanduser().resolve()}",
        f"--timingCacheFile={Path(timing_cache_path).expanduser().resolve()}",
        "--stronglyTyped",
        f"--builderOptimizationLevel={active.builder_optimization_level}",
        f"--memPoolSize=workspace:{active.workspace_mib}",
        f"--profilingVerbosity={active.profiling_verbosity}",
        "--skipInference",
    ]


def probe_trtexec(path: str = "trtexec") -> dict[str, Any]:
    executable = shutil.which(path) if not Path(path).is_absolute() else path
    if not executable or not Path(executable).is_file():
        raise FileNotFoundError(f"trtexec executable not found: {path}")
    completed = subprocess.run(
        [str(executable), "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output = completed.stdout or ""
    versions = re.findall(r"(?:TensorRT[^v\n]*v|TensorRT\s+version:\s*)([0-9.]+)", output, re.I)
    if not versions:
        raise RuntimeError("unable to parse TensorRT version from trtexec output")
    version = versions[-1]
    return {
        "path": str(Path(executable).resolve()),
        "version": version,
        "major": tensorrt_major_from_version(version),
    }


def build_tensorrt(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    timing_cache_path: str | Path,
    log_path: str | Path,
    config: TensorRTBuildConfig | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Path | list[str]:
    source = Path(onnx_path).expanduser().resolve()
    engine = Path(engine_path).expanduser().resolve()
    cache = Path(timing_cache_path).expanduser().resolve()
    log = Path(log_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    for target in (engine, log):
        if target.exists() and not overwrite:
            raise FileExistsError(target)
    command = build_tensorrt_command(source, engine, timing_cache_path=cache, config=config)
    if dry_run:
        return command
    engine.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    temporary = log.with_name(f".{log.name}.tmp")
    temporary.write_text(completed.stdout or "", encoding="utf-8")
    temporary.replace(log)
    if completed.returncode != 0:
        raise RuntimeError(f"trtexec failed with exit code {completed.returncode}; see {log}")
    if not engine.is_file():
        raise RuntimeError(f"trtexec did not create engine: {engine}")
    return engine


def benchmark_engine(
    engine_path: str | Path,
    *,
    trtexec_path: str = "trtexec",
    duration_sec: int = 3,
) -> dict[str, Any]:
    command = [
        trtexec_path,
        f"--loadEngine={Path(engine_path).expanduser().resolve()}",
        "--warmUp=500",
        f"--duration={max(1, int(duration_sec))}",
        "--useSpinWait",
    ]
    completed = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
    )
    output = completed.stdout or ""
    throughput = re.search(r"Throughput:\s*([0-9.]+)\s*qps", output, re.I)
    latency = re.search(
        r"Latency:\s*min\s*=\s*[0-9.]+\s*ms,\s*max\s*=\s*[0-9.]+\s*ms,\s*mean\s*=\s*([0-9.]+)\s*ms",
        output,
        re.I,
    )
    if completed.returncode != 0:
        raise RuntimeError("TensorRT smoke benchmark failed")
    return {
        "throughput_fps": float(throughput.group(1)) if throughput else None,
        "mean_latency_ms": float(latency.group(1)) if latency else None,
        "duration_sec": max(1, int(duration_sec)),
    }
