from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from lingbot_realtime.services.measurement import InvalidMeasurement
from lingbot_realtime.services.runtime import RuntimeConflict, RuntimeController
from lingbot_realtime.web.protocol import PROTOCOL, pack_capture, pack_preview


class MeasurementRequest(BaseModel):
    start: tuple[int, int] = Field(description="Start pixel [x, y]")
    end: tuple[int, int] = Field(description="End pixel [x, y]")


def create_app(runtime: RuntimeController) -> FastAPI:
    static_path = Path(__file__).resolve().parent / "static" / "index.html"
    index_html = static_path.read_text(encoding="utf-8")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        runtime.start()
        try:
            yield
        finally:
            runtime.shutdown()

    app = FastAPI(title="LingBot Realtime", version="0.1.0", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return index_html

    @app.get("/status")
    async def status() -> JSONResponse:
        return JSONResponse(runtime.status())

    @app.post("/api/capture")
    async def capture() -> JSONResponse:
        try:
            return JSONResponse(runtime.capture(), status_code=202)
        except RuntimeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/captures/{capture_id}")
    async def get_capture(capture_id: str) -> JSONResponse:
        try:
            return JSONResponse(runtime.get_capture(capture_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capture not found") from exc

    @app.post("/api/captures/{capture_id}/retry")
    async def retry_capture(capture_id: str) -> JSONResponse:
        try:
            return JSONResponse(runtime.retry_inference(capture_id), status_code=202)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capture not found") from exc
        except RuntimeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/captures/{capture_id}/measurements")
    async def add_measurement(capture_id: str, request: MeasurementRequest) -> JSONResponse:
        try:
            result = runtime.add_measurement(capture_id, request.start, request.end)
            return JSONResponse(result, status_code=201)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capture not found") from exc
        except RuntimeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvalidMeasurement as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.delete("/api/captures/{capture_id}/measurements/{measurement_id}")
    async def delete_measurement(capture_id: str, measurement_id: int) -> JSONResponse:
        try:
            runtime.delete_measurement(capture_id, measurement_id)
            return JSONResponse({"deleted": measurement_id})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Measurement not found") from exc

    @app.delete("/api/captures/{capture_id}/measurements")
    async def clear_measurements(capture_id: str) -> JSONResponse:
        try:
            runtime.clear_measurements(capture_id)
            return JSONResponse({"cleared": True})
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Capture not found") from exc

    @app.post("/api/recapture")
    async def recapture() -> JSONResponse:
        try:
            return JSONResponse(runtime.recapture())
        except RuntimeConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/quit")
    async def quit_service() -> JSONResponse:
        runtime.request_quit()
        return JSONResponse({"quit": True})

    @app.websocket("/ws/preview")
    async def preview(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_json(
            {"type": "hello", "protocol": PROTOCOL, "status": runtime.status()}
        )
        last_revision = -1
        last_payload_key: tuple[str, object] | None = None
        next_preview_at = 0.0
        try:
            while True:
                update = await asyncio.to_thread(runtime.wait_for_update, last_revision, 1.0)
                if update.status.get("quit_requested") or update.status.get("phase") == "stopped":
                    await websocket.close(code=1001, reason="service stopping")
                    return
                if update.revision == last_revision:
                    continue
                last_revision = update.revision
                phase = str(update.status["phase"])
                packet = None
                payload_key: tuple[str, object] | None = None
                now = asyncio.get_running_loop().time()
                if phase == "preview" and update.frame is not None and now >= next_preview_at:
                    payload_key = ("preview", update.frame.frame_id)
                    packet = pack_preview(update.frame, update.revision, phase)
                    next_preview_at = now + 1.0 / runtime.config.preview_fps
                elif (
                    phase == "ready"
                    and update.capture is not None
                    and update.capture.result is not None
                ):
                    payload_key = ("capture", update.capture.capture_id)
                    if payload_key != last_payload_key:
                        packet = pack_capture(update.capture, update.revision, phase)

                if packet is None:
                    await websocket.send_json(
                        {
                            "type": "state",
                            "protocol": PROTOCOL,
                            "revision": update.revision,
                            "status": update.status,
                        }
                    )
                    continue

                header, payload = packet
                await websocket.send_json(header)
                await websocket.send_bytes(payload)
                try:
                    ack = await asyncio.wait_for(
                        websocket.receive_json(), timeout=runtime.config.ack_timeout_sec
                    )
                except asyncio.TimeoutError:
                    await websocket.close(code=1013, reason="frame ACK timeout")
                    return
                if ack.get("type") != "ack" or int(ack.get("revision", -1)) != update.revision:
                    await websocket.close(code=1002, reason="invalid frame ACK")
                    return
                last_payload_key = payload_key
        except WebSocketDisconnect:
            return

    return app
