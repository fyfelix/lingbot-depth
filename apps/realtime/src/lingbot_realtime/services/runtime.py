from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import cv2
import numpy as np

from lingbot_realtime.camera.base import FrameSource
from lingbot_realtime.config import AppConfig
from lingbot_realtime.domain import CaptureRecord, Phase, PredictionPacket, RGBDFrame
from lingbot_realtime.inference.base import InferenceEngine
from lingbot_realtime.realtime import FramePacket as WireFramePacket
from lingbot_realtime.realtime import PredictionPacket as RecordingPacket
from lingbot_realtime.realtime import Recorder, make_session_paths
from lingbot_realtime.services.measurement import measure_distance
from lingbot_realtime.services.persistence import PersistenceService

logger = logging.getLogger(__name__)


class RuntimeConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeUpdate:
    revision: int
    status: dict[str, Any]
    frame: RGBDFrame | None
    capture: CaptureRecord | None


class _FpsTracker:
    def __init__(self) -> None:
        self.samples: list[float] = []
        self.total = 0.0
        self.count = 0

    def tick(self, seconds: float) -> tuple[float, float, float]:
        seconds = max(seconds, 1e-9)
        self.samples.append(seconds)
        self.total += seconds
        self.count += 1
        while len(self.samples) > 1 and sum(self.samples) > 1.0:
            self.samples.pop(0)
        return 1.0 / seconds, self.count / self.total, len(self.samples) / sum(self.samples)

    def reset(self) -> None:
        self.samples.clear()
        self.total = 0.0
        self.count = 0


class RuntimeController:
    """Single-owner camera worker with latest-frame replacement and continuous inference."""

    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        engine: InferenceEngine | None,
        persistence: PersistenceService,
    ) -> None:
        self.config = config
        self.source = source
        self.engine = engine
        self.persistence = persistence
        self._condition = threading.Condition()
        self._phase = Phase.STARTING
        self._revision = 0
        self._inference_revision = 0
        self._publication_id = 0
        self._latest_frame: RGBDFrame | None = None
        self._latest_prediction: PredictionPacket | None = None
        self._active_capture: CaptureRecord | None = None
        self._camera_status = "idle"
        self._camera_error: str | None = None
        self._camera_failures = 0
        self._camera_retry_paused = False
        self._camera_requested = bool(config.auto_connect)
        self._model_status = "disabled" if engine is None else "idle"
        self._model_error: str | None = None
        self._inference_enabled = bool(config.inference_enabled and engine is not None)
        self._record_on = False
        self._recorder: Recorder | None = None
        self._record_session_dir: str | None = None
        self._saved = 0
        self._frame_count = 0
        self._frame_drops = 0
        self._inference_fps = self._inference_avg_fps = self._inference_window_fps = 0.0
        self._e2e_fps = self._e2e_avg_fps = self._e2e_window_fps = 0.0
        self._stop_requested = False
        self._quit_requested = False
        self._started = False
        self._started_at = time.time()
        self._camera_started_at = 0.0
        self._camera_thread: threading.Thread | None = None
        self._model_thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snapshot")

    @property
    def quit_requested(self) -> bool:
        with self._condition:
            return self._quit_requested

    def _touch_locked(self) -> None:
        self._revision += 1
        self._condition.notify_all()

    def start(self) -> None:
        with self._condition:
            if self._started:
                return
            self._started = True
            self._phase = Phase.PREVIEW
            if self.engine is not None:
                self._model_status = "loading"
            self._touch_locked()
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True, name="rgbd-worker"
        )
        self._camera_thread.start()
        if self.engine is not None:
            self._model_thread = threading.Thread(
                target=self._load_model, daemon=True, name="model-loader"
            )
            self._model_thread.start()

    def _load_model(self) -> None:
        assert self.engine is not None
        try:
            self.engine.load()
        except Exception as exc:  # noqa: BLE001
            with self._condition:
                self._model_status = "error"
                self._model_error = str(exc) or exc.__class__.__name__
                if self._inference_enabled:
                    self._inference_enabled = False
                    self._inference_revision += 1
                self._touch_locked()
            return
        with self._condition:
            self._model_status = "ready"
            self._model_error = None
            self._touch_locked()

    def connect_camera(self) -> dict[str, Any]:
        with self._condition:
            if self._stop_requested:
                raise RuntimeConflict("Service is stopping")
            self._camera_requested = True
            self._camera_error = None
            self._camera_failures = 0
            self._camera_retry_paused = False
            self._touch_locked()
            return self._status_locked()

    def disconnect_camera(self) -> dict[str, Any]:
        recorder: Recorder | None = None
        with self._condition:
            self._camera_requested = False
            self._camera_failures = 0
            self._camera_retry_paused = False
            self._record_on = False
            recorder, self._recorder = self._recorder, None
            self._camera_status = (
                "disconnecting" if self._camera_status in {"connecting", "running"} else "idle"
            )
            self._touch_locked()
        if recorder is not None:
            recorder.close()
        return self.status()

    def toggle_inference(self, enabled: bool) -> dict[str, Any]:
        with self._condition:
            if self._record_on:
                raise RuntimeConflict("Cannot switch inference while recording is active")
            if enabled and (self.engine is None or self._model_status != "ready"):
                raise RuntimeConflict("No ready model runtime is loaded")
            next_enabled = bool(enabled)
            if self._inference_enabled != next_enabled:
                self._inference_enabled = next_enabled
                self._inference_revision += 1
                self._latest_prediction = None
                self._publication_id += 1
                self._inference_fps = 0.0
                self._inference_avg_fps = 0.0
                self._inference_window_fps = 0.0
                self._touch_locked()
            return self._status_locked()

    def _new_recorder(self) -> Recorder:
        identifier = self.config.record_session_id
        if identifier is None:
            identifier = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        paths = make_session_paths(
            self.config.record_root,
            session_id=identifier,
            overwrite=self.config.record_overwrite,
        )
        self._record_session_dir = str(paths.root)
        return Recorder(
            paths,
            fps=self.config.fps,
            overwrite=self.config.record_overwrite,
            metadata={
                "engine": self.engine.name if self.engine is not None else None,
                "device": self.engine.device_name if self.engine is not None else None,
                "max_depth_m": self.config.max_depth_m,
                "resolution_level": self.config.resolution_level,
                "num_tokens": self.config.num_tokens,
            },
        )

    def toggle_recording(self) -> bool:
        recorder_to_close: Recorder | None = None
        with self._condition:
            if not self.config.record_enabled:
                raise RuntimeConflict("Recording disabled at startup")
            if self._camera_status != "running":
                raise RuntimeConflict("Camera is not running")
            if not self._inference_enabled or self._model_status != "ready":
                raise RuntimeConflict("Inference must be enabled before recording")
            if self._record_on:
                self._record_on = False
                recorder_to_close, self._recorder = self._recorder, None
            else:
                self._recorder = self._new_recorder()
                self._record_on = True
            enabled = self._record_on
            self._touch_locked()
        if recorder_to_close is not None:
            recorder_to_close.close()
        return enabled

    @staticmethod
    def _recording_packet(packet: PredictionPacket) -> RecordingPacket:
        frame = packet.frame
        raw_mm = np.rint(frame.depth_m * 1000.0)
        raw_mm[~np.isfinite(raw_mm) | (raw_mm < 0) | (raw_mm > 65535)] = 0
        wire_frame = WireFramePacket(
            frame_id=frame.frame_id,
            timestamp_ns=int(frame.timestamp * 1_000_000_000),
            color_bgr=cv2.cvtColor(frame.color_rgb, cv2.COLOR_RGB2BGR),
            raw_depth_mm=raw_mm.astype(np.uint16),
            metadata=MappingProxyType(
                {
                    "intrinsics": frame.intrinsics.to_dict(),
                    "source_depth_scale_m": frame.depth_scale_m,
                }
            ),
        )
        assert packet.result is not None
        return RecordingPacket(
            wire_frame,
            packet.result.pred_depth_m,
            MappingProxyType(dict(packet.timings_ms)),
            MappingProxyType({"engine": "continuous"}),
        )

    def _camera_loop(self) -> None:
        source_active = False
        inference_tracker = _FpsTracker()
        e2e_tracker = _FpsTracker()
        retry_delay_sec = 0.5
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop_requested or self._camera_requested,
                    timeout=0.5,
                )
                if self._stop_requested:
                    break
                should_run = self._camera_requested
            if not should_run:
                continue
            try:
                if not source_active:
                    with self._condition:
                        self._camera_status = "connecting"
                        self._camera_error = None
                        self._touch_locked()
                    self.source.start()
                    source_active = True
                    inference_tracker.reset()
                    e2e_tracker.reset()
                    with self._condition:
                        self._camera_status = "running"
                        self._camera_started_at = time.time()
                        self._touch_locked()
                started = time.perf_counter()
                frame = self.source.read(timeout_sec=self.config.camera_read_timeout_sec)
                with self._condition:
                    inference_enabled = self._inference_enabled and self._model_status == "ready"
                result = None
                timings: dict[str, float] = {}
                if inference_enabled and self.engine is not None:
                    try:
                        result = self.engine.infer(frame)
                    except Exception as exc:  # noqa: BLE001
                        with self._condition:
                            self._model_status = "error"
                            self._model_error = str(exc) or exc.__class__.__name__
                            if self._inference_enabled:
                                self._inference_enabled = False
                                self._inference_revision += 1
                            self._record_on = False
                            failed_recorder, self._recorder = self._recorder, None
                            self._touch_locked()
                        if failed_recorder is not None:
                            failed_recorder.close()
                        inf = (0.0, 0.0, 0.0)
                    else:
                        timings["inference"] = result.elapsed_sec * 1000.0
                        inf = inference_tracker.tick(result.elapsed_sec)
                else:
                    inf = (0.0, 0.0, 0.0)
                e2e = e2e_tracker.tick(time.perf_counter() - started)
                packet = PredictionPacket(frame, result, self._publication_id + 1, timings)
                with self._condition:
                    if not self._camera_requested or self._stop_requested:
                        continue
                    self._latest_frame = frame
                    self._latest_prediction = packet
                    self._publication_id += 1
                    self._frame_count += 1
                    self._camera_error = None
                    self._camera_failures = 0
                    self._camera_retry_paused = False
                    self._inference_fps, self._inference_avg_fps, self._inference_window_fps = inf
                    self._e2e_fps, self._e2e_avg_fps, self._e2e_window_fps = e2e
                    retry_delay_sec = 0.5
                    recorder = self._recorder if self._record_on and result is not None else None
                    self._touch_locked()
                if recorder is not None:
                    close_recorder = False
                    try:
                        recorder.publish(self._recording_packet(packet))
                    except RuntimeError as exc:
                        with self._condition:
                            if self._record_on:
                                self._camera_error = f"Recording stopped: {exc}"
                                self._record_on = False
                                self._recorder = None
                                self._touch_locked()
                        close_recorder = True
                    else:
                        with self._condition:
                            self._saved = recorder.frames_written
                            if (
                                self.config.max_record_frames
                                and self._saved >= self.config.max_record_frames
                            ):
                                self._record_on = False
                                self._recorder = None
                                close_recorder = True
                            self._touch_locked()
                    if close_recorder:
                        recorder.close()
            except Exception as exc:  # noqa: BLE001
                if source_active:
                    with suppress(Exception):
                        self.source.stop()
                    source_active = False
                with self._condition:
                    self._frame_drops += 1
                    self._camera_status = "error"
                    error = str(exc) or exc.__class__.__name__
                    self._camera_error = error
                    self._camera_failures += 1
                    if self._camera_failures >= 3:
                        self._camera_requested = False
                        self._camera_retry_paused = True
                        self._camera_error = (
                            f"{error} Automatic retries paused after "
                            f"{self._camera_failures} consecutive failures; "
                            "press Connect to retry."
                        )
                    self._touch_locked()
                # Hardware/USB failures can persist for several seconds. Back off
                # retries without making disconnect or shutdown wait for a sleep.
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._stop_requested or not self._camera_requested,
                        timeout=retry_delay_sec,
                    )
                # Keep persistent USB failures from repeatedly restarting the
                # librealsense pipeline and destabilizing the host controller.
                retry_delay_sec = min(retry_delay_sec * 2.0, 30.0)
            with self._condition:
                disconnect = not self._camera_requested or self._stop_requested
            if disconnect and source_active:
                with suppress(Exception):
                    self.source.stop()
                source_active = False
                with self._condition:
                    self._camera_status = "stopped" if self._stop_requested else "idle"
                    self._camera_started_at = 0.0
                    self._touch_locked()
        if source_active:
            with suppress(Exception):
                self.source.stop()
        with self._condition:
            self._camera_status = "stopped"
            self._camera_started_at = 0.0
            self._touch_locked()

    def _status_locked(self) -> dict[str, Any]:
        now = time.time()
        capture = self._active_capture
        visible_frame = self._latest_frame
        if self._inference_enabled and self._model_status == "ready":
            visible_frame = (
                self._latest_prediction.frame
                if self._latest_prediction and self._latest_prediction.result is not None
                else None
            )
        return {
            "phase": self._phase.value,
            "revision": self._revision,
            "source": self.source.name,
            "camera_status": self._camera_status,
            "camera_error": self._camera_error,
            "camera_retry_paused": self._camera_retry_paused,
            "model_status": self._model_status,
            "model_error": self._model_error,
            "engine": self.engine.name if self.engine is not None else "sensor-only",
            "device": self.engine.device_name if self.engine is not None else "none",
            "frame_id": visible_frame.frame_id if visible_frame else None,
            "raw_frame_id": self._latest_frame.frame_id if self._latest_frame else None,
            "frame": self._frame_count,
            "publication_id": self._publication_id,
            "prediction_frame_id": (
                self._latest_prediction.frame.frame_id
                if self._latest_prediction and self._latest_prediction.result is not None
                else None
            ),
            "intrinsics": visible_frame.intrinsics.to_dict() if visible_frame else None,
            "capture": capture.to_dict() if capture is not None else None,
            "inference_enabled": self._inference_enabled,
            "inference_revision": self._inference_revision,
            "inference_fps": round(self._inference_fps, 2),
            "inference_avg_fps": round(self._inference_avg_fps, 2),
            "inference_window_fps": round(self._inference_window_fps, 2),
            "e2e_fps": round(self._e2e_fps, 2),
            "e2e_avg_fps": round(self._e2e_avg_fps, 2),
            "e2e_window_fps": round(self._e2e_window_fps, 2),
            "frame_drops": self._frame_drops,
            "record_allowed": self.config.record_enabled,
            "record_on": self._record_on,
            "saved": self._saved,
            "session_dir": self._record_session_dir,
            "last_error": self._camera_error or self._model_error or "",
            "is_disp": False,
            "uptime_sec": round(now - self._started_at, 1),
            "camera_uptime_sec": round(
                now - self._camera_started_at if self._camera_started_at else 0.0,
                1,
            ),
            "quit_requested": self._quit_requested,
            "save_results": self.persistence.enabled,
            "stream_url": "/ws/realtime",
            "render_backend": "webgl",
        }

    def status(self) -> dict[str, Any]:
        with self._condition:
            return self._status_locked()

    def wait_for_update(self, after_revision: int, timeout: float = 1.0) -> RuntimeUpdate:
        with self._condition:
            self._condition.wait_for(
                lambda: self._revision > after_revision or self._stop_requested,
                timeout=timeout,
            )
            return RuntimeUpdate(
                self._revision,
                self._status_locked(),
                self._latest_frame,
                self._active_capture,
            )

    def wait_for_packet(
        self, after_publication: int, timeout: float = 1.0
    ) -> tuple[int, PredictionPacket, dict[str, Any]] | None:
        with self._condition:
            self._condition.wait_for(
                lambda: self._publication_id > after_publication or self._stop_requested,
                timeout=timeout,
            )
            packet = self._latest_prediction
            if packet is None or self._publication_id <= after_publication:
                return None
            fields = {
                "frame_id": self._publication_id,
                "frame": self._frame_count,
                "timestamp": packet.frame.timestamp,
                "inference_dt_sec": packet.result.elapsed_sec if packet.result else 0.0,
                "inference_fps": round(self._inference_fps, 2),
                "inference_avg_fps": round(self._inference_avg_fps, 2),
                "inference_window_fps": round(self._inference_window_fps, 2),
                "e2e_fps": round(self._e2e_fps, 2),
                "e2e_avg_fps": round(self._e2e_avg_fps, 2),
                "e2e_window_fps": round(self._e2e_window_fps, 2),
                "inference_enabled": self._inference_enabled,
                "inference_revision": self._inference_revision,
                "pred_depth_source": (
                    self.engine.name
                    if self.engine is not None and packet.result is not None
                    else None
                ),
                "record_on": self._record_on,
                "saved": self._saved,
                "intrinsics": packet.frame.intrinsics.to_dict(),
            }
            return self._publication_id, packet, fields

    def stream_should_end(self) -> bool:
        with self._condition:
            return self._quit_requested or self._camera_status not in {"connecting", "running"}

    def latest_frame_packet(self) -> PredictionPacket | None:
        with self._condition:
            if self._latest_prediction is not None:
                return self._latest_prediction.frozen_copy()
            if self._latest_frame is None:
                return None
            return PredictionPacket(self._latest_frame.frozen_copy(), None, self._publication_id)

    def capture(self) -> dict[str, Any]:
        with self._condition:
            if self._active_capture is not None:
                raise RuntimeConflict("A snapshot is already active; recapture first")
            packet = self._latest_prediction
            if packet is None or packet.result is None:
                raise RuntimeConflict("No complete continuous prediction is available")
            capture_id = time.strftime("capture_%Y%m%d-%H%M%S_") + uuid.uuid4().hex[:8]
            record = CaptureRecord(capture_id, packet.frame.frozen_copy())
            self._active_capture = record
            self._phase = Phase.INFERENCING
            result = packet.result.frozen_copy()
            self._touch_locked()
            response = record.to_dict()
        self._executor.submit(self._finish_frozen_capture, record, result)
        return response

    def _finish_frozen_capture(self, record: CaptureRecord, result: Any) -> None:
        with self._condition:
            if self._active_capture is not record or self._stop_requested:
                return
            record.result = result
            record.status = "ready"
            record.error = None
            self._phase = Phase.READY
            self._touch_locked()
        assert self.engine is not None
        self.persistence.save_capture(record, self.engine.name, self.engine.device_name)

    def retry_inference(self, capture_id: str) -> dict[str, Any]:
        with self._condition:
            record = self._active_capture
            if record is None or record.capture_id != capture_id:
                raise KeyError(capture_id)
            if self.engine is None or self._model_status != "ready":
                raise RuntimeConflict("Model is not ready")
            if record.status != "error":
                raise RuntimeConflict("Only failed inference can be retried")
            record.status = "inferencing"
            record.error = None
            self._phase = Phase.INFERENCING
            self._touch_locked()
        self._executor.submit(self._run_snapshot_inference, record)
        return record.to_dict()

    def _run_snapshot_inference(self, record: CaptureRecord) -> None:
        assert self.engine is not None
        try:
            result = self.engine.infer(record.frame)
        except Exception as exc:  # noqa: BLE001
            with self._condition:
                if self._active_capture is record:
                    record.status = "error"
                    record.error = str(exc) or exc.__class__.__name__
                    self._phase = Phase.ERROR
                    self._touch_locked()
            return
        self._finish_frozen_capture(record, result)

    def recapture(self) -> dict[str, Any]:
        with self._condition:
            if self._phase == Phase.INFERENCING:
                raise RuntimeConflict("Snapshot capture is still being frozen")
            self._active_capture = None
            self._phase = Phase.PREVIEW
            self._touch_locked()
            return self._status_locked()

    def get_capture(self, capture_id: str) -> dict[str, Any]:
        with self._condition:
            if self._active_capture is None or self._active_capture.capture_id != capture_id:
                raise KeyError(capture_id)
            return self._active_capture.to_dict()

    def add_measurement(
        self, capture_id: str, start_xy: tuple[int, int], end_xy: tuple[int, int]
    ) -> dict[str, Any]:
        with self._condition:
            record = self._active_capture
            if record is None or record.capture_id != capture_id:
                raise KeyError(capture_id)
            if record.result is None or record.status != "ready":
                raise RuntimeConflict("Capture result is not ready")
            measurement = measure_distance(
                record.next_measurement_id,
                record.result.pred_depth_m,
                record.frame.intrinsics,
                start_xy,
                end_xy,
                self.config.max_depth_m,
            )
            record.next_measurement_id += 1
            record.measurements.append(measurement)
            self._touch_locked()
        self.persistence.save_measurements(record)
        return measurement.to_dict()

    def delete_measurement(self, capture_id: str, measurement_id: int) -> None:
        with self._condition:
            record = self._active_capture
            if record is None or record.capture_id != capture_id:
                raise KeyError(capture_id)
            before = len(record.measurements)
            record.measurements = [
                m for m in record.measurements if m.measurement_id != measurement_id
            ]
            if len(record.measurements) == before:
                raise KeyError(measurement_id)
            self._touch_locked()
        self.persistence.save_measurements(record)

    def clear_measurements(self, capture_id: str) -> None:
        with self._condition:
            record = self._active_capture
            if record is None or record.capture_id != capture_id:
                raise KeyError(capture_id)
            record.measurements.clear()
            self._touch_locked()
        self.persistence.save_measurements(record)

    def request_quit(self) -> None:
        with self._condition:
            self._quit_requested = True
        self.shutdown()

    def shutdown(self) -> None:
        recorder: Recorder | None = None
        with self._condition:
            if self._stop_requested:
                return
            self._stop_requested = True
            self._camera_requested = False
            self._record_on = False
            recorder, self._recorder = self._recorder, None
            self._phase = Phase.STOPPED
            self._touch_locked()
        if recorder is not None:
            recorder.close()
        if (
            self._camera_thread is not None
            and self._camera_thread is not threading.current_thread()
        ):
            self._camera_thread.join(timeout=3.0)
        if self._model_thread is not None and self._model_thread is not threading.current_thread():
            self._model_thread.join(timeout=3.0)
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.engine is not None:
            self.engine.close()
