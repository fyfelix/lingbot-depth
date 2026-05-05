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
  ./evaluation/run_hammer.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [camera_type=d435] [cleanup_npy=false]

Arguments:
  camera_type        d435 | l515 | tof. Default: d435
  cleanup_npy        Delete predictions/*.npy after evaluation when true. Default: false

Environment overrides:
  DATASET_PATH       HAMMER JSONL path. Default: data/HAMMER/test.jsonl
  OUTPUT_DIR         Prediction/evaluation output directory.
                     Default: <checkpoint_dir>/hammer_<checkpoint_stub>_data_<camera_type>
  BATCH_SIZE         Dataloader batch size for inference. Default: 1
  NUM_WORKERS        Dataloader workers for inference. Default: 0
  PYTHON_BIN         Python executable. Default: .venv/bin/python if present, otherwise python
  DEVICE             Inference device: auto/cuda/mps/cpu. Default: auto
  RESOLUTION_LEVEL   LingBot-Depth resolution level. Default: 9
  SAVE_VIS           Save RGB/raw/pred/GT visualization images. Default: true
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_path="${1:-ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt}"
camera_type="${2:-d435}"
cleanup_npy="${3:-false}"

case "${camera_type}" in
    d435|l515|tof)
        ;;
    *)
        echo "unknown HAMMER camera_type: ${camera_type} (expected: d435 | l515 | tof)" >&2
        exit 2
        ;;
esac

dataset_path="${DATASET_PATH:-data/HAMMER/test.jsonl}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
device="${DEVICE:-auto}"
resolution_level="${RESOLUTION_LEVEL:-9}"
save_vis="${SAVE_VIS:-true}"

model_name="$(basename "${model_path}")"
model_stub="${model_name%.*}"
model_dir="$(dirname "${model_path}")"
output_dir="${OUTPUT_DIR:-${model_dir}/hammer_${model_stub}_data_${camera_type}}"

save_vis_arg=()
if [[ "${save_vis}" == "true" ]]; then
    save_vis_arg=(--save-vis)
fi

cd "${PROJECT_ROOT}"

echo "model path: ${model_path}"
echo "fixed model class: mdm.model.v2.MDMModel"
echo "dataset path: ${dataset_path}"
echo "camera type: ${camera_type}"
echo "output dir: ${output_dir}"
echo "batch size: ${batch_size}"
echo "num workers: ${num_workers}"
echo "device: ${device}"
echo "resolution level: ${resolution_level}"
echo "save visualization: ${save_vis}"
echo "cleanup npy: ${cleanup_npy}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --raw-type "${camera_type}" \
    --output "${output_dir}" \
    --batch-size "${batch_size}" \
    --num-workers "${num_workers}" \
    --device "${device}" \
    --resolution-level "${resolution_level}" \
    "${save_vis_arg[@]}"

echo "evaluating the model on HAMMER"
time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
    --encoder vitl \
    --model-path "${model_path}" \
    --dataset "${dataset_path}" \
    --output "${output_dir}" \
    --raw-type "${camera_type}"

if [[ "${cleanup_npy}" == "true" ]]; then
    echo "cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
    if [[ -d "${output_dir}/predictions" ]]; then
        find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
    fi
fi
