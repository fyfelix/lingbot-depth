#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
        PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
    else
        PYTHON_BIN="python"
    fi
fi

usage() {
    cat <<'EOF'
Usage:
  ./evaluation/run_eval.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [raw_type=d435] [cleanup_npy=false]

Environment overrides:
  DATASET_PATH       HAMMER or ClearPose JSONL path. Default: data/HAMMER/test.jsonl
  OUTPUT_DIR         Prediction/evaluation output directory.
                     Default: evaluation/output/<hammer|clearpose>_<timestamp>
  BATCH_SIZE         Dataloader batch size for inference. Default: 1
  NUM_WORKERS        Dataloader workers for inference. Default: 0
  PYTHON_BIN         Python executable. Default: .venv/bin/python if present, otherwise python
  DEVICE             Inference device: auto/cuda/mps/cpu. Default: auto
  RESOLUTION_LEVEL   LingBot-Depth resolution level. Default: 9

This wrapper is fixed for LingBot-Depth-v0.5 on HAMMER/ClearPose:
  - inference uses mdm.model.v2.MDMModel
  - input is RGB + selected raw depth (HAMMER: d435/l515/tof, ClearPose: d435 only)
  - predictions are saved as HxW float32 metric depth in meters
  - eval.py reads those .npy files and computes the original fixed metrics
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_path="${1:-ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt}"
raw_type="${2:-d435}"
cleanup_npy="${3:-false}"

dataset_path="${DATASET_PATH:-data/HAMMER/test.jsonl}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
device="${DEVICE:-auto}"
resolution_level="${RESOLUTION_LEVEL:-9}"
run_timestamp="$(date +%Y-%m-%d_%H-%M-%S)"

case "${dataset_path}" in
    *clearpose*|*ClearPose*)
        dataset_tag="clearpose"
        ;;
    *hammer*|*HAMMER*)
        dataset_tag="hammer"
        ;;
    *)
        echo "Error: DATASET_PATH must contain HAMMER or clearpose: ${dataset_path}" >&2
        exit 2
        ;;
esac

output_dir="${OUTPUT_DIR:-evaluation/output/${dataset_tag}_${run_timestamp}}"

cd "${PROJECT_ROOT}"

echo "model path: ${model_path}"
echo "fixed model class: mdm.model.v2.MDMModel"
echo "dataset path: ${dataset_path}"
echo "dataset tag: ${dataset_tag}"
echo "raw type: ${raw_type}"
echo "output dir: ${output_dir}"
echo "batch size: ${batch_size}"
echo "num workers: ${num_workers}"
echo "device: ${device}"
echo "resolution level: ${resolution_level}"
echo "cleanup npy: ${cleanup_npy}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --raw-type "${raw_type}" \
    --output "${output_dir}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --device "${device}" \
    --resolution-level "${resolution_level}"

echo "evaluating the model on ${dataset_tag}"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder vitl \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --raw-type "${raw_type}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}"
    find "${output_dir}" -maxdepth 1 -type f -name '*.npy' -delete
fi
