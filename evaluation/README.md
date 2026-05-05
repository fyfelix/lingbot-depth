# LingBot-Depth HAMMER / ClearPose / DREDS Evaluation

这个目录是面向 LingBot-Depth 当前项目模型的评估导出目录。它复用 HAMMER / ClearPose / DREDS JSONL 数据组织和原有指标计算，但推理阶段固定使用项目内 `mdm.model.v2.MDMModel`，不使用 CDM 模板中的 `RGBDDepth`、`Resize`、`NormalizeImage` 或 `is_disp` 链路。

```text
evaluation/
├── dataset.py
├── infer.py
├── eval.py
├── run_hammer.sh
├── run_clearpose.sh
├── run_dreds.sh
├── requirements.txt
└── utils/
```

链路边界：

1. `infer.py` 读取 JSONL 和 LingBot checkpoint，写出 `predictions/*.npy`。
2. `eval.py` 从输出目录读取 `predictions/*.npy`，找不到时兼容读取旧版根目录 `.npy`。
3. 三个 `run_*.sh` wrapper 负责选择数据集、组织输出目录、顺序运行推理和评估。

## 数据集约定

HAMMER 每行 JSONL 是一个 frame，字段包含 `rgb`、`depth`、`d435_depth`、`l515_depth`、`tof_depth`、`depth-range`。GT 和 raw depth 是 16-bit PNG，`depth_scale=1000.0`，`raw-type` 支持 `d435/l515/tof`。

ClearPose 每行 JSONL 是一个 sequence，字段包含 `rgb`、`rgb-suffix`、`raw_depth-suffix`、`depth-suffix`、`depth-range`。脚本会在 sequence 目录下 glob 展开 frame；GT 和 raw depth 是 16-bit PNG，`depth_scale=1000.0`，固定使用 `raw-type=d435`。

DREDS 使用 sequence JSONL，字段包含 `rgb`、`rgb-suffix`、`raw_depth-suffix`、`depth-suffix`、`depth-range`。raw / GT depth 是 EXR float 深度，单位已经是 meter，`depth_scale=1.0`。DREDS 推理仍会把 `raw_depth-suffix` 对应的 EXR 作为 LingBot raw depth 输入；`raw-type=d435` 只是共享 CLI 的占位参数。

样本命名：

```text
HAMMER:    scene_name#filename
ClearPose: dir1#dir2#filename
DREDS:     dir1#dir2#filename
```

## 三条运行路线

### HAMMER

```bash
DATASET_PATH=data/HAMMER/test.jsonl \
./evaluation/run_hammer.sh ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt d435 false
```

参数：

```text
./evaluation/run_hammer.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [camera_type=d435] [cleanup_npy=false]
```

`camera_type` 支持 `d435`、`l515`、`tof`。

### ClearPose

```bash
DATASET_PATH=data/clearpose/test.jsonl \
./evaluation/run_clearpose.sh ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt false
```

参数：

```text
./evaluation/run_clearpose.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [cleanup_npy=false]
```

ClearPose 固定使用 `raw-type=d435`。

### DREDS

```bash
DREDS_KNOWN_JSONL=data/DREDS/test_std_catknown.jsonl \
DREDS_NOVEL_JSONL=data/DREDS/test_std_catnovel.jsonl \
OUTPUT_ROOT=/tmp/lingbot_dreds_eval \
./evaluation/run_dreds.sh ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt all false
```

参数：

```text
./evaluation/run_dreds.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [variant=all] [cleanup_npy=false]
```

说明：

- `variant=catknown` 使用 `DREDS_KNOWN_JSONL`。
- `variant=catnovel` 使用 `DREDS_NOVEL_JSONL`。
- `variant=all` 顺序运行 catknown 和 catnovel；此时只能使用 `OUTPUT_ROOT`，不能使用单目录 `OUTPUT_DIR`。
- `run_dreds.sh` 会设置 `OPENCV_IO_ENABLE_OPENEXR=1`，`infer.py` 和 `eval.py` 也会在 import `cv2` 前设置同一环境变量。

## 常用环境变量

```text
DATASET_PATH          HAMMER / ClearPose JSONL 路径
DREDS_KNOWN_JSONL     DREDS catknown JSONL 路径
DREDS_NOVEL_JSONL     DREDS catnovel JSONL 路径
OUTPUT_DIR            单数据集输出目录
OUTPUT_ROOT           DREDS all 模式的输出根目录
BATCH_SIZE            推理 DataLoader batch size，默认 1
NUM_WORKERS           推理 DataLoader worker 数，默认 0
DEVICE                推理设备 auto/cuda/mps/cpu，默认 auto
RESOLUTION_LEVEL      LingBot-Depth resolution level，默认 9
SAVE_VIS              true 时保存可视化图，默认 true
PYTHON_BIN            Python 可执行文件，默认优先使用项目 .venv/bin/python
```

## 输出目录

若没有设置 `OUTPUT_DIR` / `OUTPUT_ROOT`，默认写到 checkpoint 同级目录：

```text
<checkpoint_dir>/hammer_<checkpoint_stub>_data_<camera_type>/
<checkpoint_dir>/clearpose_<checkpoint_stub>_data_d435/
<checkpoint_dir>/dreds_catknown_<checkpoint_stub>/
<checkpoint_dir>/dreds_catnovel_<checkpoint_stub>/
```

输出内容：

```text
args.json
eval_args.json
predictions/*.npy
visualizations/*_vis.jpg
all_metrics_<timestamp>_False.csv
mean_metrics_<timestamp>_False.json
```

`predictions/*.npy` 是 `HxW float32` metric depth，单位 meter。`visualizations/*_vis.jpg` 是 RGB / raw depth / predicted depth / GT depth 四联图，仅当 `SAVE_VIS=true` 或直接给 `infer.py --save-vis` 时生成。

如果 `cleanup_npy=true`，wrapper 会在评估结束后删除 `predictions/*.npy`，保留 `args.json`、`eval_args.json`、CSV/JSON 指标和可视化图。

## 关键约定

- 模型固定为 `mdm.model.v2.MDMModel.from_pretrained(...)`，checkpoint 需要符合当前 LingBot-Depth 的 `model_config` + `model` 结构。
- 推理输入是 RGB `[0, 1]` 和 meter 单位 raw depth；resize、normalize、depth remap 由 `MDMModel` 内部处理。
- 推理默认不应用模型 mask；需要时可直接调用 `infer.py --apply-mask`。
- `eval.py` 使用数据集的 `depth_range` 覆盖评估 `min_depth/max_depth`，并从 dataset 的 `depth_scale` 决定 GT 读取缩放。
- DREDS 允许 prediction shape 与 GT shape 不一致，评估时用 nearest resize 对齐；HAMMER / ClearPose 遇到 shape mismatch 会直接报错。
- Mac 本地适合做 import、CLI、dataset smoke check；完整模型推理建议在 CUDA 环境运行。
