#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
app_dir="$(cd "$script_dir/.." && pwd)"
repo_dir="$(cd "$app_dir/../.." && pwd)"
conda_bin="${LINGBOT_CONDA_BIN:-/home/asdepth/miniconda3/bin/conda}"
conda_env="${LINGBOT_CONDA_ENV:-asdepth-host-trt11-py312}"
bind="${LINGBOT_REALTIME_BIND:-0.0.0.0}"
port="${LINGBOT_REALTIME_PORT:-8000}"
record_root="${LINGBOT_REALTIME_RECORD_ROOT:-$app_dir/runs/recordings}"
engine="${LINGBOT_REALTIME_ENGINE:-}"
manifest="${LINGBOT_REALTIME_MANIFEST:-}"

if [[ ! -x "$conda_bin" ]]; then
  echo "error: conda executable not found: $conda_bin" >&2
  exit 2
fi

args=(
  python -m lingbot_realtime
  --source realsense
  --bind "$bind"
  --port "$port"
  --record-root "$record_root"
)

if [[ -n "$engine" ]]; then
  args+=(--backend tensorrt --engine "$engine")
  if [[ -n "$manifest" ]]; then
    args+=(--manifest "$manifest")
  fi
else
  args+=(--backend auto --no-inference)
fi

cd "$repo_dir"
export PYTHONPATH="$app_dir/src:$repo_dir${PYTHONPATH:+:$PYTHONPATH}"
exec "$conda_bin" run --no-capture-output -n "$conda_env" "${args[@]}" "$@"
