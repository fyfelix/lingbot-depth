# LingBot-Depth HAMMER / ClearPose Evaluation

这个目录是在 `lingbot-depth` 外部项目根目录下的评估适配，不修改软链接来源 `run_bs_eval_pipeline`。

## 适配范围

- 数据集支持 HAMMER 和 ClearPose JSONL，默认 JSONL 为 `data/HAMMER/test.jsonl`。
- 模型固定为 LingBot-Depth 推荐主模型 `LingBot-Depth-v0.5`，默认 checkpoint 为 `ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt`。
- 推理入口固定使用当前项目官方模型类 `mdm.model.v2.MDMModel` 和 `MDMModel.infer(...)`。
- 输入为 RGB + raw depth；HAMMER `raw-type` 支持 `d435/l515/tof`，ClearPose only supports `d435`。
- 输出为 `HxW float32` metric depth，单位 meter，保存为逐样本 `.npy`。
- 当前模型直接输出 metric depth，不做 disparity / inverse depth 转换，也不启用 alignment。

## 运行方式

在项目根目录运行 HAMMER 默认评估：

```bash
./evaluation/run_eval.sh
```

等价于：

```bash
./evaluation/run_eval.sh ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt d435 false
```

参数：

```text
./evaluation/run_eval.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [raw_type=d435] [cleanup_npy=false]
```

常用环境变量：

```bash
DATASET_PATH=data/HAMMER/test.jsonl
OUTPUT_DIR=evaluation/output/hammer_<timestamp>
BATCH_SIZE=1
NUM_WORKERS=0
PYTHON_BIN=.venv/bin/python
DEVICE=auto
RESOLUTION_LEVEL=9
```

ClearPose 示例：

```bash
DATASET_PATH=data/clearpose/test.jsonl \
BATCH_SIZE=1 \
NUM_WORKERS=0 \
./evaluation/run_eval.sh ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt d435 false
```

如果显式设置 `OUTPUT_DIR`，脚本会完全使用该路径，不再自动追加 dataset tag 或 timestamp。

## 输出

默认输出目录按数据集和启动时间生成：

```text
evaluation/output/hammer_2026-05-03_15-30-00/
evaluation/output/clearpose_2026-05-03_15-30-00/
```

目录内容通常包含：

```text
args.json
eval_args.json
scene#frame.npy
set#scene#frame-stem.npy
all_metrics_<timestamp>_False.csv
mean_metrics_<timestamp>_False.json
```

注意：dataset tag 只出现在输出目录名中，指标文件名不再额外包含 `hammer` 或 `clearpose`。

如果第三个位置参数为 `true`，`run_eval.sh` 会在评估结束后删除逐样本 `.npy`，保留 CSV/JSON 指标。

## 样本命名

HAMMER 保持原命名规则：

```text
data/HAMMER/raw_data/scene12_traj1_1/polarization/rgb/000000.png
-> scene12_traj1_1#000000.npy
```

ClearPose 使用 `set#scene#frame-stem`：

```text
data/clearpose/set2/scene4/000709-color.png
-> set2#scene4#000709-color.npy
```

推理保存 `.npy` 和评估读取 `.npy` 使用同一套命名逻辑。

## 复用的原始评估逻辑

以下文件从 `run_bs_eval_pipeline` 复制后继续使用：

```text
dataset.py
eval.py
utils/metric.py
utils/img_utils.py
requirements.txt
```

`eval.py` 继续读取 GT depth、构造 valid mask、调用 `utils/metric.py` 中的固定指标，并保存 CSV/JSON。指标定义没有修改。

## 推理细节

- RGB 通过 OpenCV 读取后从 BGR 转 RGB，并归一化到 `[0, 1]`。
- raw depth 按 `depth_scale=1000` 转为 meter。
- raw depth 中 `<=0`、非有限值和超过数据集 `depth-range` 上限的输入会置为 `0`。
- `MDMModel.infer` 默认不传 intrinsics，因为评估只需要 depth，不需要 point cloud。
- `infer.py` 默认不应用模型 mask，以保存完整的 dense metric depth；如确需应用模型 mask，可直接调用 `infer.py --apply-mask`。
- `BATCH_SIZE` 只控制 DataLoader 取样分组；模型仍按单图循环推理。MacBook/CPU 建议保持 `BATCH_SIZE=1`。

## 已知限制

- 本目录不是通用评估框架，只适配当前 LingBot-Depth 模型与 HAMMER/ClearPose JSONL。
- ClearPose only supports `raw-type=d435`。
- MacBook 上只建议做参数、import、dataset smoke check；完整模型推理和评估建议在 GPU 环境运行。
- 当前环境若缺少 `pandas`，`eval.py` 无法完整运行；请安装 `evaluation/requirements.txt` 中依赖后再跑完整评估。
