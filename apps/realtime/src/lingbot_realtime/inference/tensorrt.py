"""Lazy TensorRT v3 runtime for the fixed FP16 RGB-D deployment graph."""

from __future__ import annotations

import ctypes
import hashlib
import json
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from lingbot_realtime.domain import InferenceResult, RGBDFrame
from lingbot_realtime.inference.mock import depth_to_points
from lingbot_realtime.inference.preprocessing import sanitize_metric_depth
from lingbot_realtime.realtime import D435HostPreprocessor, FramePacket
from lingbot_realtime.realtime.preprocess import Resolution


class TensorRTRunner(Protocol):
    @property
    def input_shape(self) -> tuple[int, ...]: ...
    def infer(self, inputs: Any) -> Any: ...
    def close(self) -> None: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _major(value: str) -> int:
    head = value.strip().split(".", 1)[0]
    if head.isdigit():
        return int(head)
    raise ValueError(f"invalid TensorRT version: {value!r}")


def validate_deployment_manifest(
    manifest_path: str | Path,
    engine_path: str | Path,
    *,
    runtime_tensorrt_version: str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    engine = Path(engine_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    precision = str(payload.get("precision", "")).lower()
    if precision not in {"fp16", "float16"}:
        raise RuntimeError(f"deployment manifest must declare FP16, got {precision!r}")
    tensor = payload.get("tensor_contract", payload.get("io", {}))
    input_spec = tensor.get("input", {}) if isinstance(tensor, dict) else {}
    output_spec = tensor.get("output", {}) if isinstance(tensor, dict) else {}
    if input_spec and (
        input_spec.get("name") != "rgbd_input"
        or list(input_spec.get("shape", [])) != [1, 4, 480, 640]
    ):
        raise RuntimeError(f"incompatible deployment input contract: {input_spec}")
    if output_spec and (
        output_spec.get("name") != "depth" or list(output_spec.get("shape", [])) != [1, 480, 640]
    ):
        raise RuntimeError(f"incompatible deployment output contract: {output_spec}")
    artifacts = payload.get("artifacts", {})
    engine_record = artifacts.get("engine", {}) if isinstance(artifacts, dict) else {}
    expected_hash = engine_record.get("sha256") if isinstance(engine_record, dict) else None
    if expected_hash and _sha256(engine) != expected_hash:
        raise RuntimeError("TensorRT engine checksum does not match deployment manifest")
    built_major = payload.get("tensorrt_major")
    if built_major is None:
        versions = payload.get("versions", {})
        trt = versions.get("tensorrt", {}) if isinstance(versions, dict) else {}
        built_major = trt.get("major") if isinstance(trt, dict) else None
    if runtime_tensorrt_version is not None and built_major is not None:
        if int(built_major) != _major(runtime_tensorrt_version):
            raise RuntimeError(
                f"TensorRT major mismatch: engine={built_major}, runtime={runtime_tensorrt_version}"
            )
    return payload


class _CudaRuntime:
    _LIBRARIES = ("libcudart.so", "libcudart.so.13", "libcudart.so.12", "libcudart.so.11.0")

    def __init__(self) -> None:
        last_error: OSError | None = None
        for name in self._LIBRARIES:
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise RuntimeError(f"cannot load CUDA runtime: {last_error}")
        void = ctypes.c_void_p
        integer = ctypes.c_int
        self.lib.cudaSetDevice.argtypes = [integer]
        self.lib.cudaSetDevice.restype = integer
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(void), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = integer
        self.lib.cudaFree.argtypes = [void]
        self.lib.cudaFree.restype = integer
        self.lib.cudaMemcpyAsync.argtypes = [void, void, ctypes.c_size_t, integer, void]
        self.lib.cudaMemcpyAsync.restype = integer
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(void)]
        self.lib.cudaStreamCreate.restype = integer
        self.lib.cudaStreamDestroy.argtypes = [void]
        self.lib.cudaStreamDestroy.restype = integer
        self.lib.cudaStreamSynchronize.argtypes = [void]
        self.lib.cudaStreamSynchronize.restype = integer
        self.lib.cudaGetErrorString.argtypes = [integer]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

    def _check(self, code: int, operation: str) -> None:
        if code:
            raw = self.lib.cudaGetErrorString(code)
            raise RuntimeError(f"{operation} failed: {raw.decode() if raw else code}")

    def set_device(self, index: int) -> None:
        self._check(self.lib.cudaSetDevice(index), "cudaSetDevice")

    def malloc(self, size: int) -> int:
        pointer = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(pointer), size), "cudaMalloc")
        if pointer.value is None:
            raise RuntimeError("cudaMalloc returned null")
        return int(pointer.value)

    def free(self, pointer: int | None) -> None:
        if pointer:
            self._check(self.lib.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def stream_create(self) -> int:
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        if stream.value is None:
            raise RuntimeError("cudaStreamCreate returned null")
        return int(stream.value)

    def stream_destroy(self, stream: int | None) -> None:
        if stream:
            self._check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy")

    def synchronize(self, stream: int) -> None:
        self._check(
            self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize"
        )

    def h2d(self, target: int, source: np.ndarray, stream: int) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(target),
                ctypes.c_void_p(int(source.ctypes.data)),
                source.nbytes,
                1,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync(H2D)",
        )

    def d2h(self, target: np.ndarray, source: int, stream: int) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(int(target.ctypes.data)),
                ctypes.c_void_p(source),
                target.nbytes,
                2,
                ctypes.c_void_p(stream),
            ),
            "cudaMemcpyAsync(D2H)",
        )


def _trt_dtype(trt: Any, value: Any) -> np.dtype:
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT32: np.int32,
        trt.DataType.INT8: np.int8,
        trt.DataType.BOOL: np.bool_,
    }
    if value not in mapping:
        raise RuntimeError(f"unsupported TensorRT dtype: {value}")
    return np.dtype(mapping[value])


class _TensorRTEngineRunner:
    def __init__(
        self,
        engine_path: Path,
        *,
        device: str = "cuda",
        require_fp16: bool = True,
    ) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT Python bindings are required") from exc
        if device == "cuda":
            device_index = 0
        elif device.startswith("cuda:"):
            device_index = int(device.split(":", 1)[1])
        else:
            raise ValueError(f"TensorRT requires CUDA, got {device!r}")
        self._cuda = _CudaRuntime()
        self._cuda.set_device(device_index)
        self._stream: int | None = self._cuda.stream_create()
        logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            self.close()
            raise RuntimeError(f"failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            self.close()
            raise RuntimeError("failed to create TensorRT execution context")
        names = tuple(self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors))
        inputs = tuple(
            n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
        )
        outputs = tuple(
            n for n in names if self._engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        )
        if inputs != ("rgbd_input",) or outputs != ("depth",):
            self.close()
            raise RuntimeError(f"engine IO must be rgbd_input -> depth, got {inputs} -> {outputs}")
        self._input_name, self._output_name = inputs[0], outputs[0]
        self._input_shape = tuple(int(x) for x in self._engine.get_tensor_shape(self._input_name))
        if self._input_shape != (1, 4, 480, 640):
            self.close()
            raise RuntimeError(f"engine input shape must be (1,4,480,640), got {self._input_shape}")
        self._input_dtype = _trt_dtype(trt, self._engine.get_tensor_dtype(self._input_name))
        self._output_dtype = _trt_dtype(trt, self._engine.get_tensor_dtype(self._output_name))
        if require_fp16 and (
            self._input_dtype != np.dtype(np.float16) or self._output_dtype != np.dtype(np.float16)
        ):
            self.close()
            raise RuntimeError("formal TensorRT engine must use FP16 input and output")
        self._input_pointer: int | None = None
        self._output_pointer: int | None = None
        self._input_capacity = self._output_capacity = 0

    @property
    def input_shape(self) -> tuple[int, ...]:
        return self._input_shape

    def _allocate(self, input_bytes: int, output_bytes: int) -> None:
        if input_bytes > self._input_capacity:
            self._cuda.free(self._input_pointer)
            self._input_pointer = self._cuda.malloc(input_bytes)
            self._input_capacity = input_bytes
        if output_bytes > self._output_capacity:
            self._cuda.free(self._output_pointer)
            self._output_pointer = self._cuda.malloc(output_bytes)
            self._output_capacity = output_bytes

    def infer(self, inputs: Any) -> np.ndarray:
        value = np.ascontiguousarray(inputs, dtype=self._input_dtype)
        if tuple(value.shape) != self._input_shape:
            raise ValueError(f"TensorRT input shape mismatch: {value.shape}")
        if not self._context.set_input_shape(self._input_name, tuple(value.shape)):
            raise RuntimeError(f"TensorRT rejected input shape {value.shape}")
        output_shape = tuple(int(x) for x in self._context.get_tensor_shape(self._output_name))
        if output_shape != (1, 480, 640):
            raise RuntimeError(f"TensorRT depth shape mismatch: {output_shape}")
        output = np.empty(output_shape, dtype=self._output_dtype)
        self._allocate(value.nbytes, output.nbytes)
        assert self._input_pointer and self._output_pointer and self._stream
        self._context.set_tensor_address(self._input_name, self._input_pointer)
        self._context.set_tensor_address(self._output_name, self._output_pointer)
        self._cuda.h2d(self._input_pointer, value, self._stream)
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        self._cuda.d2h(output, self._output_pointer, self._stream)
        self._cuda.synchronize(self._stream)
        return output

    def close(self) -> None:
        cuda = getattr(self, "_cuda", None)
        if cuda is None:
            return
        cuda.free(getattr(self, "_input_pointer", None))
        cuda.free(getattr(self, "_output_pointer", None))
        self._input_pointer = self._output_pointer = None
        cuda.stream_destroy(getattr(self, "_stream", None))
        self._stream = None

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


class TensorRTInferenceEngine:
    def __init__(
        self,
        engine_path: str | Path,
        *,
        manifest_path: str | Path | None = None,
        device: str = "cuda",
        max_depth_m: float = 6.0,
        runner: TensorRTRunner | None = None,
    ) -> None:
        self.engine_path = Path(engine_path).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve() if manifest_path else None
        self.requested_device = "cuda" if device == "auto" else device
        self.max_depth_m = float(max_depth_m)
        self._runner = runner
        # MDM performs ImageNet normalization inside its RGB-D encoder.  The
        # local TensorRT graph therefore consumes RGB in [0, 1], unlike the
        # AS-Depth graph that normalizes on the host.
        self._preprocessor = D435HostPreprocessor(
            Resolution(480, 640),
            depth_scale=1000.0,
            max_depth_m=self.max_depth_m,
        )

    @property
    def name(self) -> str:
        return "tensorrt"

    @property
    def device_name(self) -> str:
        return self.requested_device

    def load(self) -> None:
        if self._runner is not None:
            return
        if not self.engine_path.is_file():
            raise FileNotFoundError(self.engine_path)
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise RuntimeError("TensorRT Python bindings are required") from exc
        if self.manifest_path is not None:
            validate_deployment_manifest(
                self.manifest_path,
                self.engine_path,
                runtime_tensorrt_version=str(trt.__version__),
            )
        self._runner = _TensorRTEngineRunner(self.engine_path, device=self.requested_device)

    def _prepare(self, frame: RGBDFrame) -> np.ndarray:
        depth_m = sanitize_metric_depth(frame.depth_m, self.max_depth_m)
        raw_mm = np.rint(depth_m * np.float32(1000.0))
        raw_mm[(raw_mm < 0.0) | (raw_mm > np.iinfo(np.uint16).max)] = 0.0
        packet = FramePacket(
            frame_id=frame.frame_id,
            timestamp_ns=int(frame.timestamp * 1_000_000_000),
            color_bgr=np.ascontiguousarray(frame.color_rgb[..., ::-1]),
            raw_depth_mm=np.ascontiguousarray(raw_mm, dtype=np.uint16),
        )
        return np.ascontiguousarray(self._preprocessor.prepare(packet), dtype=np.float16)

    def infer(self, frame: RGBDFrame) -> InferenceResult:
        if self._runner is None:
            raise RuntimeError("TensorRT engine is not loaded")
        started = time.perf_counter()
        output = np.asarray(self._runner.infer(self._prepare(frame)), dtype=np.float32).squeeze()
        if output.shape != (480, 640):
            raise RuntimeError(f"TensorRT output must be 480x640, got {output.shape}")
        if output.shape != frame.depth_m.shape:
            output = cv2.resize(
                output,
                (frame.depth_m.shape[1], frame.depth_m.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        valid = np.isfinite(output) & (output > 0) & (output <= self.max_depth_m)
        pred = np.where(valid, output, 0.0).astype(np.float32)
        return InferenceResult(pred, depth_to_points(pred, frame), time.perf_counter() - started)

    def close(self) -> None:
        if self._runner is not None:
            self._runner.close()
        self._runner = None
