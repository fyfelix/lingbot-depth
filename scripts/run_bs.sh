#!/bin/bash
set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="$PWD:$PYTHONPATH"

MODEL_PATH=${1:-"ckpts/model.pt"}
DATASET_PATH=${2:-"data/HAMMER/test.jsonl"}
CAMERA_TYPE=${3:-"d435"}

echo "Using Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Camera Type: ${CAMERA_TYPE}"

# 构建模型前缀
MODEL_NAME=$(basename "${MODEL_PATH}")
MODEL_STUB="${MODEL_NAME%%.*}" # 移除 .pt 等后缀
DIRNAME=$(dirname "${MODEL_PATH}")

# 兼容各种文件夹情况
OUTPUT_DIR="${DIRNAME}/eval_hammer_${MODEL_STUB}_${CAMERA_TYPE}"
echo "Output Directory: ${OUTPUT_DIR}"

BS=16
NUM_WORKERS=8

# 可选额外参数（如果用户想启用基准测试里的对齐校准，可传入此参数）
# 例如: bash run_bs.sh model.pt data/HAMMER/test.jsonl d435 --align
EXTRA_ARGS=${@:4}

# ===============
# 1. 批量推理部分
# ===============
echo "[1/2] 开始在数据集上推理模型..."
time python scripts/infer_dataset_bs.py \
    --model-path "${MODEL_PATH}" \
    --dataset "${DATASET_PATH}" \
    --raw-type "${CAMERA_TYPE}" \
    --output "${OUTPUT_DIR}" \
    --batch-size ${BS} \
    --num-workers ${NUM_WORKERS}

# ===============
# 2. 指标评测部分
# ===============
echo "[2/2] 开始在数据集上评估指标..."
time python scripts/eval_mp.py \
    --dataset "${DATASET_PATH}" \
    --output "${OUTPUT_DIR}" \
    --raw-type "${CAMERA_TYPE}" \
    --batch-size ${BS} \
    --num-workers ${NUM_WORKERS} \
    ${EXTRA_ARGS}

echo "全部评估完成！结果保存在 ${OUTPUT_DIR}"
