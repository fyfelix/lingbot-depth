# LingBot Realtime

Fork-local RealSense snapshot web application. It imports the upstream `mdm` model through its
public API and keeps all application code inside `apps/lingbot_realtime`.

## macOS development mode

No RealSense device or model download is required:

```bash
uv sync --project apps/lingbot_realtime --extra test
uv run --project apps/lingbot_realtime \
  python -m lingbot_realtime --source fixture --inference-engine mock
```

Open <http://127.0.0.1:8000>. The fixture source generates a deterministic animated RGB-D scene.
Capture, inference state transitions, depth visualization, point-cloud rendering, and 2D metric
measurement all remain available.

Run tests with:

```bash
(cd apps/lingbot_realtime && uv run pytest)
```

## MDM inference on a fixture

```bash
uv run --project apps/lingbot_realtime \
  python -m lingbot_realtime \
  --source fixture \
  --inference-engine mdm \
  --model-path robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  --device mps
```

## RealSense

`pyrealsense2` is imported only when `--source realsense` is selected. On Linux it can be
installed through the `realsense` extra. macOS has no compatible PyPI wheel for every Python
release, so this app includes a reproducible source-build installer that targets its own virtual
environment. It does not install Python files into Homebrew or modify upstream project files.

From the repository root:

```bash
brew install librealsense cmake ninja pkg-config
uv sync --project apps/lingbot_realtime --extra test
apps/lingbot_realtime/scripts/install_pyrealsense2_macos.sh
apps/lingbot_realtime/.venv/bin/python \
  apps/lingbot_realtime/scripts/check_realsense_macos.py
```

The installer builds the same librealsense version reported by Homebrew for the app's exact
Python ABI and CPU architecture. Re-run it after changing the app Python version or upgrading
Homebrew librealsense.

### macOS 12 and newer USB access

Librealsense's official macOS documentation requires elevated privileges on macOS 12 and newer.
The OS attaches its `UVCAssistant` driver to the camera's UVC interfaces, so an ordinary process
can discover the D435 but fails to claim it with `RS2_USB_STATUS_ACCESS` or `failed to set power
state`.

First validate SDK access with the exact app interpreter:

```bash
sudo -H apps/lingbot_realtime/.venv/bin/python \
  apps/lingbot_realtime/scripts/check_realsense_macos.py
```

For a bounded end-to-end check that starts the real web server, waits for a D435 frame, captures
the scene, runs mock inference, creates one 2D measurement, and shuts down cleanly:

```bash
sudo -H apps/lingbot_realtime/.venv/bin/python \
  apps/lingbot_realtime/scripts/smoke_realsense_demo.py
```

Then start the camera with mock inference to isolate capture from model setup:

```bash
apps/lingbot_realtime/scripts/run_realsense_macos.sh
```

Open <http://127.0.0.1:8000>. The launcher explicitly elevates only the app's Python process and
defaults to mock inference. To run MDM after camera validation:

```bash
apps/lingbot_realtime/scripts/run_realsense_macos.sh \
  --inference-engine mdm \
  --model-path robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  --device mps
```

Use a USB 3.x data cable and prefer a direct MacBook port. Hubs and docks add another failure
point for D4xx bandwidth and power negotiation. The default camera profile is `640x480@30`; depth
is aligned to the color stream, and the device-provided depth scale and color intrinsics are used
for inference and measurement.

## Optional result persistence

Pass `--save-results --output-root apps/lingbot_realtime/runs`. Each capture writes RGB,
raw/predicted metric depth arrays, visualizations, point cloud, intrinsics, metadata, and
measurements into its own directory.
