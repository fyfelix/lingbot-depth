#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$app_dir/../.." && pwd)"
venv_python="$app_dir/.venv/bin/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this launcher is for macOS only" >&2
  exit 2
fi

if [[ ! -x "$venv_python" ]]; then
  echo "error: application virtual environment not found: $venv_python" >&2
  exit 2
fi

if ! "$venv_python" -c 'import pyrealsense2' >/dev/null 2>&1; then
  echo "error: pyrealsense2 is not installed in the application environment" >&2
  echo "run: $script_dir/install_pyrealsense2_macos.sh" >&2
  exit 2
fi

cat <<'EOF'
macOS 12+ requires elevated privileges for librealsense USB access.
This command elevates only the application virtual-environment Python process.
The first camera test uses mock inference; pass a later --inference-engine option to override it.
EOF

cd "$repo_dir"
exec sudo -H "$venv_python" -m lingbot_realtime \
  --source realsense \
  --inference-engine mock \
  "$@"
