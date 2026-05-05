# evaluation 新版处理链路分析

## 文件清单

```text
evaluation/
├── dataset.py
├── infer.py
├── eval.py
├── run_hammer.sh
├── run_clearpose.sh
├── run_dreds.sh
├── README.md
├── requirements.txt
└── utils/
```

旧的单入口 `run_eval.sh` 已拆成三个数据集专属 wrapper。`infer.py` 负责生成 `predictions/*.npy`，`eval.py` 负责读取预测并写出 CSV/JSON 指标。

## 数据格式

HAMMER JSONL 每行是一个 frame，字段包含 `rgb`、`depth`、`d435_depth`、`l515_depth`、`tof_depth`、`depth-range`。路径相对 JSONL 所在目录解析，GT/raw depth 是 16-bit PNG，按 `depth_scale=1000.0` 转 meter；`raw-type` 支持 `d435/l515/tof`。

ClearPose JSONL 每行是一个 sequence，字段包含 `rgb`、`rgb-suffix`、`raw_depth-suffix`、`depth-suffix`、`depth-range`。dataset 会在 sequence 目录下 glob 展开 frame，最多取每个 sequence 前 300 帧；GT/raw depth 是 16-bit PNG，`depth_scale=1000.0`，固定使用 `raw-type=d435`。

DREDS JSONL 采用 ClearPose 类似的 sequence 展开模式，字段包含 `rgb`、`rgb-suffix`、`raw_depth-suffix`、`depth-suffix`、`depth-range`。本机样例使用 `_color.png`、`_depth_415.exr`、`_gt_depth.exr`；raw/GT depth 是 EXR float，单位 meter，`depth_scale=1.0`，每个 sequence 最多取前 50 帧。

样本命名统一由 `sample_name_for_dataset()` 生成：HAMMER 为 `scene_name#filename`，ClearPose/DREDS 为 `dir1#dir2#filename`。推理保存和评估读取使用同一命名规则。

## 模型与推理链路

当前 pipeline 固定加载 LingBot-Depth 项目模型：

```python
from mdm.model.v2 import MDMModel
model = MDMModel.from_pretrained(args.model_path).to(device).eval()
```

`MDMModel.from_pretrained(...)` 期望 checkpoint 含 `model_config` 和 `model`。推理输入为 RGB `[1, 3, H, W]`、范围 `[0, 1]`，以及 meter 单位 raw depth `[1, H, W]`；raw depth 中非有限值、`<=0` 和超过数据集 `depth-range` 上限的位置置为 0。推理调用：

```python
model.infer(
    image_tensor,
    depth_in=depth_tensor,
    intrinsics=None,
    resolution_level=args.resolution_level,
    apply_mask=args.apply_mask,
    use_fp16=use_fp16,
)
```

`BATCH_SIZE` 只影响 DataLoader 分组，模型仍逐样本调用 `MDMModel.infer()`。输出是 `HxW float32` metric depth，写入 `output/predictions/<sample>.npy`；可视化写入 `output/visualizations/<sample>_vis.jpg`。

## 与 CDM 模板差异

可复用的结构包括 dataset 工厂函数、DREDSDataset、三 wrapper、`predictions/` 与 `visualizations/` 子目录、eval 读取新版目录并 fallback 旧根目录、DREDS shape alignment。

必须保留项目适配的部分包括模型类、checkpoint 结构、推理 API、RGB/raw depth 预处理、`DEVICE`/`RESOLUTION_LEVEL`/`APPLY_MASK` 参数和当前 `utils/metric.py` 指标。不要把本目录改成 CDM 的 `rgbddepth.dpt.RGBDDepth`、`Resize`、`NormalizeImage`、`PrepareForNet` 或 `is_disp` 链路。

评估阶段从 dataset 读取 `depth_range` 和 `depth_scale`。DREDS 允许 prediction shape 与 GT shape 不一致，并用 nearest resize 对齐；HAMMER/ClearPose 遇到 shape mismatch 会直接报错。
