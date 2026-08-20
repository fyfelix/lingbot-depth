from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from lingbot_realtime.realtime import WebPublisher
from lingbot_realtime.services.measurement import InvalidMeasurement
from lingbot_realtime.services.runtime import RuntimeConflict, RuntimeController
from lingbot_realtime.web.protocol import PROTOCOL, pack_capture, pack_preview
from lingbot_realtime.web.realtime_protocol import build_pointcloud, build_web_frame

REALTIME_PROTOCOL = WebPublisher.PROTOCOL
REALTIME_FLOW_CONTROL = WebPublisher.FLOW_CONTROL


class MeasurementRequest(BaseModel):
    start: tuple[int, int] = Field(description="Start pixel [x, y]")
    end: tuple[int, int] = Field(description="End pixel [x, y]")


def _json_error(exc: Exception, status: int) -> HTTPException:
    return HTTPException(status_code=status, detail=str(exc))


async def _send_json(websocket: WebSocket, value: dict[str, Any], timeout: float) -> None:
    await asyncio.wait_for(websocket.send_json(value), timeout=timeout)


async def _send_bytes(websocket: WebSocket, value: bytes, timeout: float) -> None:
    await asyncio.wait_for(websocket.send_bytes(value), timeout=timeout)


def _spectral_lut() -> list[int]:
    """Return the exact 256-entry Spectral LUT used by the reference WebGL UI."""

    anchors = np.asarray(
        [
            (158, 1, 66),
            (213, 62, 79),
            (244, 109, 67),
            (253, 174, 97),
            (254, 224, 139),
            (255, 255, 191),
            (230, 245, 152),
            (171, 221, 164),
            (102, 194, 165),
            (50, 136, 189),
            (94, 79, 162),
        ],
        dtype=np.float32,
    )
    positions = np.linspace(0.0, 1.0, anchors.shape[0])
    target = np.linspace(0.0, 1.0, 256)
    channels = [np.interp(target, positions, anchors[:, index]) for index in range(3)]
    return cast(
        list[int],
        np.stack(channels, axis=1).round().astype(np.uint8).reshape(-1).tolist(),
    )


def _ready_message(runtime: RuntimeController) -> dict[str, Any]:
    status = runtime.status()
    config = runtime.config
    return {
        "type": "ready",
        "protocol": REALTIME_PROTOCOL,
        "flow_control": REALTIME_FLOW_CONTROL,
        "max_in_flight_frames": 1,
        "frame_ack_timeout_sec": config.ack_timeout_sec,
        "color_layout": "BGR",
        "depth_unit": "millimeter",
        "depth_scale": 1000.0,
        "max_depth": config.max_depth_m,
        "vis_min": config.vis_min_m,
        "vis_max": config.vis_max_m,
        "pred_vis_percentile_min": config.pred_vis_percentile_min,
        "pred_vis_percentile_max": config.pred_vis_percentile_max,
        "inference_enabled": status["inference_enabled"],
        "inference_revision": status["inference_revision"],
        "pred_depth_source": (status["engine"] if status["inference_enabled"] else None),
        "cloud_stride": config.cloud_stride,
        "cloud_point_budget": config.cloud_point_budget,
        "colormap": "Spectral",
        "colormap_lut_rgb": _spectral_lut(),
    }


def create_app(runtime: RuntimeController) -> FastAPI:
    static_dir = Path(__file__).resolve().parent / "static"
    realtime_html = (static_dir / "realtime.html").read_text(encoding="utf-8")
    pointcloud_html = (static_dir / "pointcloud.html").read_text(encoding="utf-8")
    snapshot_html = (static_dir / "index.html").read_text(encoding="utf-8")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title="LingBot-Depth Realtime", version="0.3.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return realtime_html

    @app.get("/pointcloud", response_class=HTMLResponse)
    async def pointcloud() -> str:
        return pointcloud_html

    @app.get("/snapshot", response_class=HTMLResponse)
    async def snapshot() -> str:
        return snapshot_html

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(runtime.status())

    @app.post("/camera/connect")
    async def connect() -> JSONResponse:
        try:
            return JSONResponse(runtime.connect_camera())
        except RuntimeConflict as exc:
            raise _json_error(exc, 409) from exc

    @app.post("/camera/disconnect")
    async def disconnect() -> JSONResponse:
        return JSONResponse(runtime.disconnect_camera())

    @app.post("/inference")
    async def inference(enabled: bool = Query(...)) -> JSONResponse:
        try:
            return JSONResponse(runtime.toggle_inference(enabled))
        except RuntimeConflict as exc:
            raise _json_error(exc, 409) from exc

    @app.post("/record/toggle")
    async def record_toggle() -> JSONResponse:
        try:
            return JSONResponse({"record_on": runtime.toggle_recording(), **runtime.status()})
        except RuntimeConflict as exc:
            raise _json_error(exc, 409) from exc

    @app.post("/quit")
    @app.post("/api/quit")
    async def quit_service() -> JSONResponse:
        runtime.request_quit()
        return JSONResponse({"quit": True})

    @app.post("/api/capture")
    async def capture() -> JSONResponse:
        try:
            return JSONResponse(runtime.capture(), status_code=202)
        except RuntimeConflict as exc:
            raise _json_error(exc, 409) from exc

    @app.get("/api/captures/{capture_id}")
    async def get_capture(capture_id: str) -> JSONResponse:
        try:
            return JSONResponse(runtime.get_capture(capture_id))
        except KeyError as exc:
            raise _json_error(exc, 404) from exc

    @app.post("/api/captures/{capture_id}/retry")
    async def retry_capture(capture_id: str) -> JSONResponse:
        try:
            return JSONResponse(runtime.retry_inference(capture_id), status_code=202)
        except KeyError as exc:
            raise _json_error(exc, 404) from exc
        except RuntimeConflict as exc:
            raise _json_error(exc, 409) from exc

    @app.post("/api/captures/{capture_id}/measurements")
    async def add_measurement(capture_id: str, request: MeasurementRequest) -> JSONResponse:
        try:
            return JSONResponse(
                runtime.add_measurement(capture_id, request.start, request.end), status_code=201
            )
        except KeyError as exc:
            raise _json_error(exc, 404) from exc
        except RuntimeConflict as exc:
            raise _json_error(exc, 409) from exc
        except InvalidMeasurement as exc:
            raise _json_error(exc, 422) from exc

    @app.delete("/api/captures/{capture_id}/measurements/{measurement_id}")
    async def delete_measurement(capture_id: str, measurement_id: int) -> JSONResponse:
        try:
            runtime.delete_measurement(capture_id, measurement_id)
            return JSONResponse({"deleted": measurement_id})
        except KeyError as exc:
            raise _json_error(exc, 404) from exc

    @app.delete("/api/captures/{capture_id}/measurements")
    async def clear_measurements(capture_id: str) -> JSONResponse:
        try:
            runtime.clear_measurements(capture_id)
            return JSONResponse({"cleared": True})
        except KeyError as exc:
            raise _json_error(exc, 404) from exc

    @app.post("/api/recapture")
    async def recapture() -> JSONResponse:
        try:
            return JSONResponse(runtime.recapture())
        except RuntimeConflict as exc:
            raise _json_error(exc, 409) from exc

    @app.websocket("/ws/realtime")
    async def realtime_ws(
        websocket: WebSocket,
        ws_stream_fps: float = Query(runtime.config.preview_fps, ge=0.0, le=120.0),
        send_color: bool = Query(True),
        send_raw: bool = Query(True),
        send_pred: bool = Query(True),
        send_raw_cloud: bool = Query(False),
        send_pred_cloud: bool = Query(True),
        cloud_stride: int = Query(runtime.config.cloud_stride, ge=1, le=16),
        cloud_point_budget: int = Query(runtime.config.cloud_point_budget, ge=0, le=500_000),
    ) -> None:
        await websocket.accept()
        await _send_json(
            websocket,
            _ready_message(runtime),
            runtime.config.send_timeout_sec,
        )
        last_publication = -1
        next_frame_at = 0.0
        try:
            while True:
                if ws_stream_fps > 0 and asyncio.get_running_loop().time() < next_frame_at:
                    await asyncio.sleep(0.002)
                    continue
                item = await asyncio.to_thread(runtime.wait_for_packet, last_publication, 0.5)
                if item is None:
                    if runtime.stream_should_end():
                        return
                    continue
                publication, packet, fields = item
                header, payload = await asyncio.to_thread(
                    build_web_frame,
                    packet,
                    max_depth_m=runtime.config.max_depth_m,
                    include_color=send_color,
                    include_raw=send_raw,
                    include_pred=send_pred,
                    include_raw_cloud=send_raw_cloud,
                    include_pred_cloud=send_pred_cloud,
                    cloud_stride=cloud_stride,
                    cloud_point_budget=cloud_point_budget,
                    fields=fields,
                )
                await _send_json(websocket, header, runtime.config.send_timeout_sec)
                await _send_bytes(websocket, payload, runtime.config.send_timeout_sec)
                ack = await asyncio.wait_for(
                    websocket.receive_json(), runtime.config.ack_timeout_sec
                )
                if (
                    ack.get("type") not in {"frame_ack", "ack"}
                    or int(ack.get("frame_id", ack.get("revision", -1))) != publication
                ):
                    await websocket.close(code=1002, reason="invalid frame ACK")
                    return
                last_publication = publication
                if ws_stream_fps > 0:
                    next_frame_at = max(
                        next_frame_at + 1.0 / ws_stream_fps,
                        asyncio.get_running_loop().time(),
                    )
        except WebSocketDisconnect:
            return
        except asyncio.TimeoutError:
            await websocket.close(code=1013, reason="frame ACK timeout")

    @app.websocket("/ws/pointcloud")
    async def pointcloud_ws(
        websocket: WebSocket,
        stride: int = Query(runtime.config.cloud_stride, ge=1, le=16),
        max_depth: float = Query(runtime.config.max_depth_m, gt=0.0, le=20.0),
        point_budget: int = Query(runtime.config.cloud_point_budget, ge=0, le=500_000),
        stream_fps: float = Query(runtime.config.preview_fps, ge=0.0, le=60.0),
    ) -> None:
        await websocket.accept()
        status_payload = runtime.status()
        await _send_json(
            websocket,
            {
                "type": "ready",
                "stride": stride,
                "max_depth": max_depth,
                "point_budget": point_budget,
                "point_stride_bytes": 16,
                "coordinate": "x_right_y_up_z_forward_negative",
                **(status_payload.get("intrinsics") or {}),
            },
            runtime.config.send_timeout_sec,
        )
        last_publication = -1
        try:
            while True:
                item = await asyncio.to_thread(runtime.wait_for_packet, last_publication, 0.5)
                if item is None:
                    if runtime.quit_requested:
                        return
                    continue
                publication, packet, fields = item
                payload = await asyncio.to_thread(
                    build_pointcloud,
                    packet,
                    stride=stride,
                    max_depth_m=max_depth,
                    point_budget=point_budget,
                )
                await _send_json(
                    websocket,
                    {
                        "type": "frame",
                        "frame": publication,
                        "points": len(payload) // 16,
                        "bytes": len(payload),
                        "stride": stride,
                        "max_depth": max_depth,
                        "point_budget": point_budget,
                        "fps": fields["e2e_window_fps"],
                        "timestamp": fields["timestamp"],
                    },
                    runtime.config.send_timeout_sec,
                )
                await _send_bytes(websocket, payload, runtime.config.send_timeout_sec)
                last_publication = publication
                if stream_fps > 0:
                    await asyncio.sleep(1.0 / stream_fps)
        except WebSocketDisconnect:
            return

    @app.websocket("/ws/preview")
    async def preview(websocket: WebSocket) -> None:
        await websocket.accept()
        await _send_json(
            websocket,
            {"type": "hello", "protocol": PROTOCOL, "status": runtime.status()},
            runtime.config.send_timeout_sec,
        )
        last_revision = -1
        last_capture: str | None = None
        try:
            while True:
                update = await asyncio.to_thread(runtime.wait_for_update, last_revision, 1.0)
                if update.status.get("quit_requested") or update.status.get("phase") == "stopped":
                    return
                if update.revision == last_revision:
                    continue
                last_revision = update.revision
                packet: tuple[dict[str, Any], bytes] | None = None
                if (
                    update.status.get("phase") == "ready"
                    and update.capture
                    and update.capture.result is not None
                    and update.capture.capture_id != last_capture
                ):
                    packet = pack_capture(
                        update.capture,
                        update.revision,
                        "ready",
                        depth_viz=runtime.config.depth_viz_config(),
                    )
                    last_capture = update.capture.capture_id
                elif update.frame is not None:
                    packet = pack_preview(
                        update.frame,
                        update.revision,
                        update.status.get("phase", "preview"),
                        depth_viz=runtime.config.depth_viz_config(),
                    )
                if packet is None:
                    await _send_json(
                        websocket,
                        {
                            "type": "state",
                            "protocol": PROTOCOL,
                            "revision": update.revision,
                            "status": update.status,
                        },
                        runtime.config.send_timeout_sec,
                    )
                    continue
                header, payload = packet
                header["status"] = update.status
                await _send_json(websocket, header, runtime.config.send_timeout_sec)
                await _send_bytes(websocket, payload, runtime.config.send_timeout_sec)
                ack = await asyncio.wait_for(
                    websocket.receive_json(), runtime.config.ack_timeout_sec
                )
                if ack.get("type") != "ack" or int(ack.get("revision", -1)) != update.revision:
                    await websocket.close(code=1002, reason="invalid frame ACK")
                    return
        except WebSocketDisconnect:
            return
        except asyncio.TimeoutError:
            await websocket.close(code=1013, reason="frame ACK timeout")

    return app
