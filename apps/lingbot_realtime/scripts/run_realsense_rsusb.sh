#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$app_dir/../.." && pwd)"
conda_bin="${LINGBOT_CONDA_BIN:-/home/asdepth/miniconda3/bin/conda}"
conda_env="${LINGBOT_REALTIME_RSUSB_ENV:-asdepth}"
realsense_root="${LINGBOT_LIBREALSENSE_RSUSB_ROOT:-/home/asdepth/librealsense/build/Release}"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "error: this launcher is for Linux only" >&2
  exit 2
fi
if [[ ! -x "$conda_bin" ]]; then
  echo "error: conda executable not found: $conda_bin" >&2
  exit 2
fi
if [[ ! -f "$realsense_root/librealsense2.so.2.57.7" ]]; then
  echo "error: RSUSB librealsense build not found: $realsense_root" >&2
  exit 2
fi

cd "$repo_dir"
export LD_LIBRARY_PATH="$realsense_root${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$realsense_root:$app_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"

args=(
  python -m lingbot_realtime
  --source realsense
  --backend mock
  --fps "${LINGBOT_REALTIME_FPS:-6}"
  --camera-read-timeout "${LINGBOT_REALTIME_CAMERA_READ_TIMEOUT:-3}"
  --bind "${LINGBOT_REALTIME_BIND:-0.0.0.0}"
  --port "${LINGBOT_REALTIME_PORT:-8766}"
)

exec "$conda_bin" run --no-capture-output -n "$conda_env" "${args[@]}" "$@"
