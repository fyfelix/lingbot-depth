#!/usr/bin/env bash

set -euo pipefail

export OPENCV_IO_ENABLE_OPENEXR=1

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
  ./evaluation/run_dreds.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [variant=all] [cleanup_npy=false]

Arguments:
  variant            catknown | catnovel | all. Default: all
  cleanup_npy        Delete predictions/*.npy after evaluation when true. Default: false

Environment overrides:
  DREDS_KNOWN_JSONL  DREDS catknown JSONL. Default: data/DREDS/test_std_catknown.jsonl
  DREDS_NOVEL_JSONL  DREDS catnovel JSONL. Default: data/DREDS/test_std_catnovel.jsonl
  OUTPUT_DIR         Prediction/evaluation output directory for a single variant.
  OUTPUT_ROOT        Root directory for default per-variant outputs. Default: checkpoint directory
  BATCH_SIZE         Dataloader batch size for inference. Default: 1
  NUM_WORKERS        Dataloader workers for inference. Default: 0
  PYTHON_BIN         Python executable. Default: .venv/bin/python if present, otherwise python
  DEVICE             Inference device: auto/cuda/mps/cpu. Default: auto
  RESOLUTION_LEVEL   LingBot-Depth resolution level. Default: 9
  SAVE_VIS           Save RGB/raw/pred/GT visualization images. Default: true

DREDS uses EXR floating-point depth in meters. raw-type is passed as d435 only
to satisfy the shared Python CLI and is ignored by the DREDS dataset loader.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

model_path="${1:-ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt}"
variant="${2:-all}"
cleanup_npy="${3:-false}"
camera_type="d435"

dreds_known_jsonl="${DREDS_KNOWN_JSONL:-data/DREDS/test_std_catknown.jsonl}"
dreds_novel_jsonl="${DREDS_NOVEL_JSONL:-data/DREDS/test_std_catnovel.jsonl}"
batch_size="${BATCH_SIZE:-1}"
num_workers="${NUM_WORKERS:-0}"
device="${DEVICE:-auto}"
resolution_level="${RESOLUTION_LEVEL:-9}"
save_vis="${SAVE_VIS:-true}"

model_name="$(basename "${model_path}")"
model_stub="${model_name%.*}"
model_dir="$(dirname "${model_path}")"
output_root="${OUTPUT_ROOT:-${model_dir}}"

if [[ "${variant}" == "all" && -n "${OUTPUT_DIR:-}" ]]; then
    echo "OUTPUT_DIR can only be used with variant=catknown or variant=catnovel; use OUTPUT_ROOT for variant=all." >&2
    exit 2
fi

save_vis_arg=()
if [[ "${save_vis}" == "true" ]]; then
    save_vis_arg=(--save-vis)
fi

cd "${PROJECT_ROOT}"

run_one_variant() {
    local label="$1"
    local jsonl_path="$2"
    local output_dir="${OUTPUT_DIR:-${output_root}/dreds_${label}_${model_stub}}"

    echo "[${label}] model path: ${model_path}"
    echo "[${label}] fixed model class: mdm.model.v2.MDMModel"
    echo "[${label}] dataset path: ${jsonl_path}"
    echo "[${label}] raw type placeholder: ${camera_type}"
    echo "[${label}] output dir: ${output_dir}"
    echo "[${label}] batch size: ${batch_size}"
    echo "[${label}] num workers: ${num_workers}"
    echo "[${label}] device: ${device}"
    echo "[${label}] resolution level: ${resolution_level}"
    echo "[${label}] save visualization: ${save_vis}"
    echo "[${label}] cleanup npy: ${cleanup_npy}"

    "${PYTHON_BIN}" "${SCRIPT_DIR}/infer.py" \
        --model-path "${model_path}" \
        --dataset "${jsonl_path}" \
        --raw-type "${camera_type}" \
        --output "${output_dir}" \
        --batch-size "${batch_size}" \
        --num-workers "${num_workers}" \
        --device "${device}" \
        --resolution-level "${resolution_level}" \
        "${save_vis_arg[@]}"

    echo "[${label}] evaluating the model on DREDS"
    time "${PYTHON_BIN}" "${SCRIPT_DIR}/eval.py" \
        --encoder vitl \
        --model-path "${model_path}" \
        --dataset "${jsonl_path}" \
        --output "${output_dir}" \
        --raw-type "${camera_type}"

    if [[ "${cleanup_npy}" == "true" ]]; then
        echo "[${label}] cleanup_npy is enabled, removing generated .npy files under ${output_dir}/predictions"
        if [[ -d "${output_dir}/predictions" ]]; then
            find "${output_dir}/predictions" -maxdepth 1 -type f -name '*.npy' -delete
        fi
    fi
}

case "${variant}" in
    catknown)
        run_one_variant catknown "${dreds_known_jsonl}"
        ;;
    catnovel)
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    all)
        run_one_variant catknown "${dreds_known_jsonl}"
        run_one_variant catnovel "${dreds_novel_jsonl}"
        ;;
    *)
        echo "unknown DREDS variant: ${variant} (expected: catknown | catnovel | all)" >&2
        exit 2
        ;;
esac
