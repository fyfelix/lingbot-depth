from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from lingbot_realtime.camera.base import FrameSource
from lingbot_realtime.config import AppConfig
from lingbot_realtime.domain import CaptureRecord, Phase, RGBDFrame
from lingbot_realtime.inference.base import InferenceEngine
from lingbot_realtime.services.measurement import measure_distance
from lingbot_realtime.services.persistence import PersistenceService


class RuntimeConflict(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeUpdate:
    revision: int
    status: dict[str, Any]
    frame: RGBDFrame | None
    capture: CaptureRecord | None


class RuntimeController:
    def __init__(
        self,
        config: AppConfig,
        source: FrameSource,
        engine: InferenceEngine,
        persistence: PersistenceService,
    ) -> None:
        self.config = config
        self.source = source
        self.engine = engine
        self.persistence = persistence
        self._condition = threading.Condition()
        self._phase = Phase.STARTING
        self._revision = 0
        self._latest_frame: RGBDFrame | None = None
        self._active_capture: CaptureRecord | None = None
        self._camera_status = "idle"
        self._camera_error: str | None = None
        self._model_status = "idle"
        self._model_error: str | None = None
        self._stop_requested = False
        self._quit_requested = False
        self._started = False
        self._camera_thread: threading.Thread | None = None
        self._model_thread: threading.Thread | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")

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
            self._model_status = "loading"
            self._touch_locked()
        self._camera_thread = threading.Thread(
            target=self._camera_loop, daemon=True, name="frame-source"
        )
        self._model_thread = threading.Thread(
            target=self._load_model, daemon=True, name="model-loader"
        )
        self._camera_thread.start()
        self._model_thread.start()

    def _load_model(self) -> None:
        try:
            self.engine.load()
        except Exception as exc:  # noqa: BLE001
            with self._condition:
                self._model_status = "error"
                self._model_error = str(exc) or exc.__class__.__name__
                self._touch_locked()
            return
        with self._condition:
            self._model_status = "ready"
            self._model_error = None
            self._touch_locked()

    def _camera_loop(self) -> None:
        source_active = False
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop_requested or self._phase == Phase.PREVIEW,
                    timeout=0.5,
                )
                if self._stop_requested:
                    break
                should_preview = self._phase == Phase.PREVIEW
            if not should_preview:
                continue
            try:
                if not source_active:
                    with self._condition:
                        self._camera_status = "connecting"
                        self._camera_error = None
                        self._touch_locked()
                    self.source.start()
                    source_active = True
                    with self._condition:
                        self._camera_status = "running"
                        self._touch_locked()
                frame = self.source.read(timeout_sec=0.75)
                with self._condition:
                    if self._phase == Phase.PREVIEW and not self._stop_requested:
                        self._latest_frame = frame
                        self._camera_error = None
                        self._touch_locked()
            except Exception as exc:  # noqa: BLE001
                if source_active:
                    try:
                        self.source.stop()
                    except Exception:
                        pass
                    source_active = False
                with self._condition:
                    self._camera_status = "error"
                    self._camera_error = str(exc) or exc.__class__.__name__
                    self._touch_locked()
                time.sleep(1.0)

            with self._condition:
                pause_requested = self._phase != Phase.PREVIEW or self._stop_requested
            if pause_requested and source_active:
                try:
                    self.source.stop()
                finally:
                    source_active = False
                    with self._condition:
                        self._camera_status = "paused" if not self._stop_requested else "stopped"
                        self._touch_locked()

        if source_active:
            try:
                self.source.stop()
            except Exception:
                pass
        with self._condition:
            self._camera_status = "stopped"
            self._touch_locked()

    def status(self) -> dict[str, Any]:
        with self._condition:
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        capture = self._active_capture
        return {
            "phase": self._phase.value,
            "revision": self._revision,
            "source": self.source.name,
            "camera_status": self._camera_status,
            "camera_error": self._camera_error,
            "model_status": self._model_status,
            "model_error": self._model_error,
            "engine": self.engine.name,
            "device": self.engine.device_name,
            "frame_id": self._latest_frame.frame_id if self._latest_frame else None,
            "capture": capture.to_dict() if capture is not None else None,
            "quit_requested": self._quit_requested,
            "save_results": self.persistence.enabled,
        }

    def wait_for_update(self, after_revision: int, timeout: float = 1.0) -> RuntimeUpdate:
        with self._condition:
            self._condition.wait_for(
                lambda: self._revision > after_revision or self._stop_requested,
                timeout=timeout,
            )
            return RuntimeUpdate(
                revision=self._revision,
                status=self._status_locked(),
                frame=self._latest_frame,
                capture=self._active_capture,
            )

    def capture(self) -> dict[str, Any]:
        with self._condition:
            if self._phase != Phase.PREVIEW:
                raise RuntimeConflict(f"Cannot capture while phase={self._phase.value}")
            if self._model_status != "ready":
                raise RuntimeConflict(f"Model is not ready: {self._model_status}")
            if self._latest_frame is None:
                raise RuntimeConflict("No camera frame is available")
            capture_id = time.strftime("capture_%Y%m%d-%H%M%S_") + uuid.uuid4().hex[:8]
            record = CaptureRecord(capture_id=capture_id, frame=self._latest_frame.frozen_copy())
            self._active_capture = record
            self._phase = Phase.INFERENCING
            self._touch_locked()
        self._executor.submit(self._run_inference, record)
        return record.to_dict()

    def _run_inference(self, record: CaptureRecord) -> None:
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
        with self._condition:
            if self._active_capture is not record or self._stop_requested:
                return
            record.result = result
            record.status = "ready"
            record.error = None
            self._phase = Phase.READY
            self._touch_locked()
        self.persistence.save_capture(record, self.engine.name, self.engine.device_name)

    def retry_inference(self, capture_id: str) -> dict[str, Any]:
        with self._condition:
            record = self._active_capture
            if record is None or record.capture_id != capture_id:
                raise KeyError(capture_id)
            if record.status != "error":
                raise RuntimeConflict("Only failed inference can be retried")
            record.status = "inferencing"
            record.error = None
            self._phase = Phase.INFERENCING
            self._touch_locked()
        self._executor.submit(self._run_inference, record)
        return record.to_dict()

    def recapture(self) -> dict[str, Any]:
        with self._condition:
            if self._phase == Phase.INFERENCING:
                raise RuntimeConflict("Inference is still running")
            self._active_capture = None
            self._latest_frame = None
            self._phase = Phase.PREVIEW
            self._camera_error = None
            self._touch_locked()
            return self._status_locked()

    def get_capture(self, capture_id: str) -> dict[str, Any]:
        with self._condition:
            if self._active_capture is None or self._active_capture.capture_id != capture_id:
                raise KeyError(capture_id)
            return self._active_capture.to_dict()

    def add_measurement(
        self,
        capture_id: str,
        start_xy: tuple[int, int],
        end_xy: tuple[int, int],
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
                item for item in record.measurements if item.measurement_id != measurement_id
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
        with self._condition:
            if self._stop_requested:
                return
            self._stop_requested = True
            self._phase = Phase.STOPPED
            self._touch_locked()
        if (
            self._camera_thread is not None
            and self._camera_thread is not threading.current_thread()
        ):
            self._camera_thread.join(timeout=2.0)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.engine.close()
