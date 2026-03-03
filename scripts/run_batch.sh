#!/bin/bash
# 批量处理某个文件夹下的所有 pt 模型文件

if [ "$#" -lt 1 ]; then
    echo "Usage: bash run_batch.sh <ckpt_directory> [dataset_jsonl] [camera_type]"
    exit 1
fi

ROOT_DIR=$1
DATASET_PATH=${2:-"data/HAMMER/test.jsonl"}
CAMERA_TYPE=${3:-"d435"}

echo "Processing all .pt files in ${ROOT_DIR}"

for ckpt in "$ROOT_DIR"/*.pt; do
    if [ -f "$ckpt" ]; then
        echo "=========================================================="
        echo "Processing checkpoint: $ckpt"
        bash scripts/run_bs.sh "$ckpt" "$DATASET_PATH" "$CAMERA_TYPE"
        echo "=========================================================="
    fi
done

echo "All batch jobs completed."
