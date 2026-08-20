#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(cd "$script_dir/.." && pwd)"
venv_python="${LINGBOT_REALTIME_PYTHON:-$app_dir/.venv/bin/python}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this installer is for macOS only" >&2
  exit 2
fi

for command_name in brew cmake ninja git; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "error: missing required command: $command_name" >&2
    exit 2
  fi
done

if [[ ! -x "$venv_python" ]]; then
  echo "error: application virtual environment not found: $venv_python" >&2
  echo "run: uv sync --project $app_dir --extra test" >&2
  exit 2
fi

if ! brew list --versions librealsense >/dev/null 2>&1; then
  echo "error: Homebrew librealsense is not installed" >&2
  echo "run: brew install librealsense" >&2
  exit 2
fi

librealsense_version="$(brew list --versions librealsense | awk 'NR == 1 { print $2 }')"
if [[ -z "$librealsense_version" ]]; then
  echo "error: could not determine the Homebrew librealsense version" >&2
  exit 2
fi

python_site="$($venv_python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
python_arch="$($venv_python -c 'import platform; print(platform.machine())')"
python_version="$($venv_python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
brew_prefix="$(brew --prefix)"
build_jobs="${LINGBOT_REALSENSE_BUILD_JOBS:-$(sysctl -n hw.logicalcpu)}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/lingbot-librealsense.XXXXXX")"

cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT

echo "Building pyrealsense2 $librealsense_version for Python $python_version ($python_arch)"
echo "Target environment: $venv_python"

git clone --depth 1 --branch "v$librealsense_version" \
  https://github.com/realsenseai/librealsense.git \
  "$work_dir/librealsense"

cmake \
  -S "$work_dir/librealsense" \
  -B "$work_dir/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$brew_prefix" \
  -DCMAKE_INSTALL_PREFIX="$work_dir/install" \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DBUILD_RS2_ALL=OFF \
  -DBUILD_SHARED_LIBS=OFF \
  -DBUILD_TOOLS=OFF \
  -DBUILD_UNIT_TESTS=OFF \
  -DBUILD_WITH_OPENMP=OFF \
  -DCHECK_FOR_UPDATES=OFF \
  -DFORCE_RSUSB_BACKEND=ON \
  -DPYTHON_EXECUTABLE="$venv_python" \
  -DPYTHON_INSTALL_DIR="$python_site/pyrealsense2"

cmake --build "$work_dir/build" \
  --target pyrealsense2 pyrsutils \
  --parallel "$build_jobs"
cmake --install "$work_dir/build" --config Release

"$venv_python" - <<'PY'
import platform
import pyrealsense2 as rs

print(f"Installed pyrealsense2 {rs.__version__} ({platform.machine()})")
print(f"Module: {rs.__file__}")
PY

echo "Installation complete."
echo "Next: $venv_python $script_dir/check_realsense_macos.py"
