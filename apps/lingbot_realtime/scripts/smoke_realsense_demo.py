#!/usr/bin/env python3

"""Run a bounded privileged RealSense web smoke test on macOS.

The script starts the actual web entry point, waits for a camera frame, captures one
scene, waits for mock inference, creates one 2D measurement, and shuts the server down.
It is intentionally separate from the application runtime and is only a local test aid.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def _request(base_url: str, path: str, method: str = "GET", body: dict | None = None) -> dict:
    payload = None if body is None else json.dumps(body).encode("utf-8")
    request = Request(
        f"{base_url}{path}",
        data=payload,
        method=method,
        headers={"Content-Type": "application/json"} if payload else {},
    )
    with urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def _wait_for_frame(base_url: str, deadline: float) -> dict:
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        try:
            status = _request(base_url, "/status")
            if status.get("frame_id") is not None:
                return status
        except (HTTPError, URLError, TimeoutError) as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(last_error)


def _wait_for_capture(base_url: str, capture_id: str, deadline: float) -> dict:
    while time.monotonic() < deadline:
        capture = _request(base_url, f"/api/captures/{capture_id}")
        if capture.get("status") in {"ready", "error"}:
            return capture
        time.sleep(0.25)
    raise RuntimeError(f"capture {capture_id} did not finish")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()

    if os.geteuid() != 0:
        raise SystemExit(
            "Run this smoke test through macOS administrator authorization, for example:\n"
            '  osascript -e \'do shell script ".../smoke_realsense_demo.py" '
            "with administrator privileges'"
        )

    app_dir = Path(__file__).resolve().parents[1]
    repo_dir = app_dir.parents[1]
    base_url = f"http://127.0.0.1:{args.port}"
    command = [
        sys.executable,
        "-m",
        "lingbot_realtime",
        "--source",
        "realsense",
        "--inference-engine",
        "mock",
        "--bind",
        "127.0.0.1",
        "--port",
        str(args.port),
    ]
    server = subprocess.Popen(
        command,
        cwd=repo_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    try:
        status = _wait_for_frame(base_url, time.monotonic() + 30)
        capture = _request(base_url, "/api/capture", method="POST")
        capture_id = str(capture["capture_id"])
        result = _wait_for_capture(base_url, capture_id, time.monotonic() + 30)
        if result.get("status") != "ready":
            print(f"SMOKE_FAILED capture={capture_id} error={result.get('error')}")
            return 1

        measurement = _request(
            base_url,
            f"/api/captures/{capture_id}/measurements",
            method="POST",
            body={"start": [100, 100], "end": [200, 200]},
        )
        print(
            "SMOKE_OK"
            f" phase={status.get('phase')}"
            f" camera={status.get('camera_status')}"
            f" model={status.get('model_status')}"
            f" frame={status.get('frame_id')}"
            f" capture={capture_id}"
            f" measurement={measurement.get('measurement_id')}"
            f" distance_m={float(measurement['distance_m']):.6f}"
        )
        return 0
    finally:
        try:
            _request(base_url, "/api/quit", method="POST")
        except (HTTPError, URLError, TimeoutError):
            pass
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.terminate()
            server.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
