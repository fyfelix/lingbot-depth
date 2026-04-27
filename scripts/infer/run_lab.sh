#!/usr/bin/env bash

set -e
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
PYTHON_BIN=${PYTHON:-python3}

default_model_path="ckpts/model.pt"
model_path=${1:-${default_model_path}}
dataset_path=${2:-data/lab/test.jsonl}

model_name="$(basename "${model_path}")"
model_stub="${model_name%%.*}"
default_output_dir="logs/lab_infer/lab_${model_stub}"
output_dir=${3:-${default_output_dir}}

"${PYTHON_BIN}" scripts/infer/infer_lab.py \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --batch-size 16 \
    --num-workers 4 \
    --depth-scale 1000.0 \
    --max-depth 6.0 \
    --save-vis
