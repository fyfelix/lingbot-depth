#!/usr/bin/env bash
# Compatibility entry point for AS-Depth's ten-position KITTI interface.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash scripts/infer/run_bs_kitti.sh [model_path] [arch] [raw_type]
      [resize_method] [is_disp] [cleanup_npy] [dataset_path] [save_vis]
      [intrinsics_path] [max_depth] [extra evaluation arguments...]

The arch, raw_type, resize_method, and is_disp positions are accepted for
AS-Depth compatibility. LingBot-Depth always runs mdm.model.v2.MDMModel and
metric-depth evaluation; raw_type must be d435 and is_disp=true is rejected.

Run after `conda activate lingbot-depth`, or set PYTHON_BIN explicitly.
Useful overrides: OUTPUT_DIR, BATCH_SIZE/BS, NUM_WORKERS, DEVICE,
RESOLUTION_LEVEL, RUN_EVAL, IMAGE_MIN, IMAGE_MAX, APPLY_MASK, USE_FP16,
PC_ROT_X_DEG, PC_ROT_Y_DEG, PC_KNN_K, PC_KNN_STD_RATIO.
EOF
}

is_true() {
    case "${1,,}" in
        1|true|yes|y|on) return 0 ;;
        *) return 1 ;;
    esac
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

MODEL_PATH="${1:-${MODEL_PATH:-ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt}}"
ARCH="${2:-${ARCH:-auto}}"
RAW_TYPE="${3:-${RAW_TYPE:-d435}}"
RESIZE_METHOD="${4:-${RESIZE_METHOD:-lower_bound}}"
IS_DISP="${5:-${IS_DISP:-auto}}"
CLEANUP_NPY="${6:-${CLEANUP_NPY:-true}}"
DATASET_PATH="${7:-${KITTI_JSONL:-${DATASET_PATH:-data/OpenDataLab___KITTI_depth_completion/val_selection_cropped.jsonl}}}"
SAVE_VIS="${8:-${SAVE_VIS:-false}}"
INTRINSICS_PATH="${9:-${INTRINSICS_PATH:-}}"
MAX_DEPTH="${10:-${MAX_DEPTH:-80.0}}"

if [[ "${RAW_TYPE,,}" != "d435" ]]; then
    echo "KITTI uses Velodyne depth; raw_type is a placeholder and must be d435" >&2
    exit 2
fi
if is_true "${IS_DISP}"; then
    echo "LingBot-Depth produces metric depth; is_disp=true is unsupported" >&2
    exit 2
fi
if [[ "${RESIZE_METHOD}" != "lower_bound" && "${RESIZE_METHOD}" != "upper_bound" ]]; then
    echo "resize_method must be lower_bound or upper_bound" >&2
    exit 2
fi

MODEL_NAME="$(basename -- "${MODEL_PATH}")"
MODEL_STUB="${MODEL_NAME%%.*}"
MODEL_DIR="$(dirname -- "${MODEL_PATH}")"
DATASET_NAME="$(basename -- "${DATASET_PATH}")"
DATASET_STUB="${DATASET_NAME%%.*}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_DIR}/kitti_${DATASET_STUB}_${MODEL_STUB}_data_d435}"
BATCH_SIZE="${BATCH_SIZE:-${BS:-4}}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-auto}"
RESOLUTION_LEVEL="${RESOLUTION_LEVEL:-9}"
RUN_EVAL="${RUN_EVAL:-true}"
IMAGE_MIN="${IMAGE_MIN:-0.1}"
IMAGE_MAX="${IMAGE_MAX:-${MAX_DEPTH}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if is_true "${RUN_EVAL}"; then
    STAGE="all"
else
    STAGE="infer"
fi

args=(
    "${PYTHON_BIN}" -m evaluation kitti
    --model-path "${MODEL_PATH}"
    --manifest "${DATASET_PATH}"
    --run-dir "${OUTPUT_DIR}"
    --stage "${STAGE}"
    --raw-max-depth "${MAX_DEPTH}"
    --visualization-min-depth "${IMAGE_MIN}"
    --visualization-max-depth "${IMAGE_MAX}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --device "${DEVICE}"
    --resolution-level "${RESOLUTION_LEVEL}"
    --pointcloud-rot-x-deg "${PC_ROT_X_DEG:-25.0}"
    --pointcloud-rot-y-deg "${PC_ROT_Y_DEG:-15.0}"
    --pointcloud-knn-k "${PC_KNN_K:-16}"
    --pointcloud-knn-std-ratio "${PC_KNN_STD_RATIO:-2.0}"
)

if [[ -n "${INTRINSICS_PATH}" ]]; then
    args+=(--intrinsics-path "${INTRINSICS_PATH}")
fi
if is_true "${SAVE_VIS}"; then
    args+=(--save-visualizations)
else
    args+=(--no-save-visualizations)
fi
if is_true "${CLEANUP_NPY}" && [[ "${STAGE}" == "all" ]]; then
    args+=(--cleanup-predictions)
fi
if is_true "${APPLY_MASK:-false}"; then
    args+=(--apply-mask)
fi
if is_true "${USE_FP16:-false}"; then
    args+=(--use-fp16)
fi
if is_true "${DISABLE_PC_KNN_FILTER:-false}"; then
    args+=(--disable-pointcloud-knn-filter)
fi
if [[ -n "${MAX_SAMPLES:-}" ]]; then
    args+=(--max-samples "${MAX_SAMPLES}")
fi
if (( $# > 10 )); then
    args+=("${@:11}")
fi

cd "${PROJECT_ROOT}"
echo "model_path=${MODEL_PATH}"
echo "model_class=mdm.model.v2.MDMModel"
echo "arch_compat=${ARCH} resize_method_compat=${RESIZE_METHOD} is_disp_compat=${IS_DISP}"
echo "benchmark=KITTI Depth Completion val_selection_cropped"
echo "dataset_path=${DATASET_PATH} run_dir=${OUTPUT_DIR} stage=${STAGE}"
"${args[@]}"

if is_true "${CLEANUP_NPY}" && [[ "${STAGE}" == "infer" ]]; then
    predictions_dir="${OUTPUT_DIR}/predictions"
    if [[ -d "${predictions_dir}" ]]; then
        find "${predictions_dir}" -type f -name '*.npy' -delete
    fi
fi
