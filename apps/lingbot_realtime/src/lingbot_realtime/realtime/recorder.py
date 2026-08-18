from __future__ import annotations

import json
import struct
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, TextIO

import cv2
import numpy as np

from .packets import PredictionPacket

_NPY_MAGIC_PREFIX = b"\x93NUMPY\x01\x00"
_NPY_ALIGN = 64
_NPY_PLACEHOLDER_COUNT = 10**15


def _format_header(shape: tuple[int, ...], descr: str, target_len: int | None = None) -> str:
    value = f"{{'descr': {descr!r}, 'fortran_order': False, 'shape': {tuple(map(int, shape))!r}, }}"
    if target_len is None:
        return value
    if len(value) > target_len:
        raise ValueError("streaming npy header placeholder is too short")
    return value[:-1] + " " * (target_len - len(value)) + "}"


def _wrap_header(value: str) -> bytes:
    header = value.encode("latin-1")
    length = len(header) + 1
    padding = _NPY_ALIGN - ((10 + length) % _NPY_ALIGN)
    return _NPY_MAGIC_PREFIX + struct.pack("<H", length + padding) + header + b" " * padding + b"\n"


def _atomic_json(path: Path, value: dict[str, Any], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


class StreamingNpyWriter:
    """Stream equal-shaped frames into a valid ``(N, ...)`` NPY file."""

    def __init__(self, path: str | Path, frame_shape: tuple[int, ...], dtype: Any) -> None:
        self.path = Path(path).expanduser().resolve()
        self.frame_shape = tuple(int(item) for item in frame_shape)
        if not self.frame_shape or any(item <= 0 for item in self.frame_shape):
            raise ValueError("streaming npy frame shape must be positive")
        self.dtype = np.dtype(dtype)
        descriptor = np.lib.format.dtype_to_descr(self.dtype)
        if not isinstance(descriptor, str):
            raise ValueError("structured dtypes are unsupported")
        self._descriptor = descriptor
        placeholder = _format_header((_NPY_PLACEHOLDER_COUNT, *self.frame_shape), descriptor)
        self._target_len = len(placeholder)
        header = _wrap_header(_format_header((0, *self.frame_shape), descriptor, self._target_len))
        self._header_length = len(header)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream: BinaryIO = self.path.open("wb")
        self._stream.write(header)
        self._count = 0
        self._closed = False

    @property
    def frames(self) -> int:
        return self._count

    def write(self, value: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("StreamingNpyWriter is closed")
        array = np.asarray(value)
        if tuple(array.shape) != self.frame_shape:
            raise ValueError(f"shape mismatch: expected={self.frame_shape}, actual={array.shape}")
        self._stream.write(np.ascontiguousarray(array, dtype=self.dtype).tobytes())
        self._count += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        header = _wrap_header(
            _format_header((self._count, *self.frame_shape), self._descriptor, self._target_len)
        )
        if len(header) != self._header_length:
            raise RuntimeError("streaming npy header length drift")
        self._stream.seek(0)
        self._stream.write(header)
        self._stream.flush()
        self._stream.close()

    def __enter__(self) -> "StreamingNpyWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass(frozen=True, slots=True)
class SessionPaths:
    root: Path
    rgb_path: Path
    raw_depth_path: Path
    pred_depth_path: Path
    frames_path: Path
    meta_path: Path


def make_session_paths(
    output_root: str | Path,
    *,
    session_id: str | None = None,
    overwrite: bool = False,
) -> SessionPaths:
    identifier = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    root = Path(output_root).expanduser().resolve() / f"session_{identifier}"
    if root.exists() and not overwrite:
        raise FileExistsError(f"realtime session already exists: {root}")
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(root)
    root.mkdir(parents=True, exist_ok=True)
    return SessionPaths(
        root=root,
        rgb_path=root / "rgb.mp4",
        raw_depth_path=root / "raw_depth.npy",
        pred_depth_path=root / "pred_depth.npy",
        frames_path=root / "frames.jsonl",
        meta_path=root / "meta.json",
    )


class Recorder:
    """Continuous recording contract: RGB MP4, raw uint16 mm and pred float32 m."""

    def __init__(
        self,
        paths: SessionPaths,
        *,
        fps: int,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> None:
        if fps <= 0:
            raise ValueError("recorder fps must be positive")
        self.paths = paths
        self.fps = int(fps)
        self._metadata = dict(metadata or {})
        self._overwrite = bool(overwrite)
        self._lock = threading.Lock()
        self._video: cv2.VideoWriter | None = None
        self._raw: StreamingNpyWriter | None = None
        self._pred: StreamingNpyWriter | None = None
        self._frames: TextIO | None = None
        self._count = 0
        self._closed = False

    @property
    def frames_written(self) -> int:
        return self._count

    def _open(self, packet: PredictionPacket) -> None:
        outputs = (
            self.paths.rgb_path,
            self.paths.raw_depth_path,
            self.paths.pred_depth_path,
            self.paths.frames_path,
            self.paths.meta_path,
        )
        existing = [path for path in outputs if path.exists()]
        if existing and not self._overwrite:
            raise FileExistsError(f"realtime outputs already exist: {existing}")
        height, width = packet.frame.color_bgr.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video = cv2.VideoWriter(
            str(self.paths.rgb_path), int(fourcc), float(self.fps), (width, height)
        )
        if not video.isOpened():
            raise RuntimeError(f"cannot open mp4 writer: {self.paths.rgb_path}")
        self._video = video
        self._raw = StreamingNpyWriter(
            self.paths.raw_depth_path, tuple(packet.frame.raw_depth_mm.shape), np.uint16
        )
        self._pred = StreamingNpyWriter(
            self.paths.pred_depth_path, tuple(packet.depth_m.shape), np.float32
        )
        self._frames = self.paths.frames_path.open("w", encoding="utf-8")
        _atomic_json(
            self.paths.meta_path,
            {
                "schema": "lingbot.realtime.recording.v1",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "fps": self.fps,
                "files": {
                    "rgb": "rgb.mp4",
                    "raw_depth": "raw_depth.npy",
                    "pred_depth": "pred_depth.npy",
                    "frames": "frames.jsonl",
                },
                "contracts": {
                    "rgb": "BGR mp4v",
                    "raw_depth": "uint16 millimeter (N,H,W)",
                    "pred_depth": "float32 meter (N,H,W), invalid=0.0",
                },
                "metadata": self._metadata,
            },
            overwrite=self._overwrite,
        )

    def publish(self, packet: PredictionPacket) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Recorder is closed")
            if self._video is None:
                self._open(packet)
            assert self._video is not None and self._raw is not None
            assert self._pred is not None and self._frames is not None
            self._video.write(packet.frame.color_bgr)
            self._raw.write(packet.frame.raw_depth_mm)
            self._pred.write(packet.depth_m)
            self._frames.write(
                json.dumps(
                    {
                        "record_index": self._count,
                        "frame_id": packet.frame.frame_id,
                        "timestamp_ns": packet.frame.timestamp_ns,
                        "timings_ms": dict(packet.timings_ms),
                        "frame_metadata": dict(packet.frame.metadata),
                        "prediction_metadata": dict(packet.metadata),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._frames.flush()
            self._count += 1

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._video is not None:
                self._video.release()
            if self._raw is not None:
                self._raw.close()
            if self._pred is not None:
                self._pred.close()
            if self._frames is not None:
                self._frames.flush()
                self._frames.close()


SessionRecorder = Recorder
