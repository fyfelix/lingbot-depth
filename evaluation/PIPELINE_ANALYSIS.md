# evaluation 与 run_bs_eval_pipeline 处理链路分析

## 结论摘要

当前 `evaluation` 目录是面向 LingBot-Depth 当前项目模型的适配版评估链路。它保留了原 `run_bs_eval_pipeline` 的数据集解析、预测落盘命名、指标计算和结果输出形式，但将推理阶段从原来的 `rgbddepth.dpt.RGBDDepth` 批量 4 通道输入，替换为当前项目的 `mdm.model.v2.MDMModel.infer(...)` 接口。

最关键的语义变化是：当前模型期望输入 `RGB + metric raw depth`，其中 RGB 是 `[0, 1]` 浮点张量，raw depth 是以 meter 为单位的二维深度图；原始链路则先把 RGB 和 depth resize/normalize 后拼成 `[B, 4, H', W']`，并可通过 `is_disp=true` 走 inverse depth/disparity 输入与输出转换。

## 当前 evaluation 处理 pipeline

### 1. 入口脚本

入口是 `evaluation/run_eval.sh`。

默认调用方式：

```bash
./evaluation/run_eval.sh [model_path=ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt] [raw_type=d435] [cleanup_npy=false]
```

默认数据集为 `data/HAMMER/test.jsonl`，默认输出目录为：

```text
evaluation/output/<hammer|clearpose>_<timestamp>
```

脚本主要做四件事：

1. 解析 `model_path`、`raw_type`、`cleanup_npy` 和环境变量。
2. 根据 `DATASET_PATH` 中是否包含 `HAMMER` 或 `clearpose` 判定数据集类型。
3. 调用 `evaluation/infer.py` 逐样本生成 `.npy` 预测和默认可视化图。
4. 调用 `evaluation/eval.py` 读取 `.npy` 并计算指标；若 `cleanup_npy=true`，最后删除逐样本 `.npy`。

可覆盖的主要环境变量：

```text
DATASET_PATH
OUTPUT_DIR
BATCH_SIZE
NUM_WORKERS
PYTHON_BIN
DEVICE
RESOLUTION_LEVEL
SAVE_VIS
```

### 2. 数据集加载

当前 `evaluation/dataset.py` 与 `run_bs_eval_pipeline/dataset.py` 内容一致。

#### HAMMER JSONL

每一行表示一个 frame，路径字段相对 JSONL 所在目录解析。样例字段如下：

```json
{
  "seq_name": "scene12_traj1_1",
  "depth": "raw_data/scene12_traj1_1/polarization/_gt/000000.png",
  "rgb": "raw_data/scene12_traj1_1/polarization/rgb/000000.png",
  "d435_depth": "raw_data/scene12_traj1_1/polarization/depth_d435/000000.png",
  "l515_depth": "raw_data/scene12_traj1_1/polarization/depth_l515/000000.png",
  "tof_depth": "raw_data/scene12_traj1_1/polarization/depth_tof/000000.png",
  "depth-range": [0.1, 6.0]
}
```

`HAMMERDataset.__getitem__` 返回：

```text
(rgb_path, raw_depth_path, gt_depth_path)
```

其中 `raw_depth_path` 由 `raw_type` 决定：

```text
d435 -> d435_depth
l515 -> l515_depth
tof  -> tof_depth
```

`depth_range` 使用第一条样本的 `"depth-range"`。

#### ClearPose JSONL

每一行表示一个 sequence，实际 frame 通过 glob 展开。样例字段如下：

```json
{
  "seq_name": "set2#scene4",
  "rgb": "set2/scene4",
  "rgb-suffix": "-color.png",
  "raw_depth-suffix": "-depth.png",
  "depth-suffix": "-depth_true.png",
  "depth-range": [0.3, 1.5]
}
```

`ClearPoseDataset` 会在 `root / item["rgb"]` 下分别按 suffix 查找 RGB、raw depth、GT depth 文件，并对每个 sequence 最多保留前 300 帧：

```text
rgb:       *-color.png
raw depth: *-depth.png
gt depth:  *-depth_true.png
```

ClearPose 当前只支持 `raw_type=d435`。`__getitem__` 同样返回：

```text
(rgb_path, raw_depth_path, gt_depth_path)
```

### 3. 当前模型推理输入格式

当前 `evaluation/infer.py` 固定加载：

```python
from mdm.model.v2 import MDMModel

model = MDMModel.from_pretrained(args.model_path).to(device).eval()
```

`MDMModel.from_pretrained(...)` 对 checkpoint 的结构有项目内约定：

```text
checkpoint["model_config"] -> 用于构建 MDMModel
checkpoint["model"]        -> 模型 state dict
```

推理时每张图单独进入 `model.infer(...)`：

```python
output = model.infer(
    image_tensor,
    depth_in=depth_tensor,
    intrinsics=None,
    resolution_level=args.resolution_level,
    apply_mask=args.apply_mask,
    use_fp16=use_fp16,
)
```

传入模型前的数据格式如下。

#### RGB

读取与变换流程：

```text
cv2.imread(..., IMREAD_COLOR)
BGR -> RGB
astype(float32) / 255.0
permute HWC -> CHW
unsqueeze batch
```

张量格式：

```text
shape: [1, 3, H, W]
dtype: float32
range: [0, 1]
color: RGB
device: args.device 解析后的 cuda/mps/cpu
```

#### raw depth

读取与变换流程：

```text
cv2.imread(..., IMREAD_UNCHANGED)
astype(float32) / depth_scale
过滤非有限值、<=0、>max_depth 的点
无效点置 0
unsqueeze batch
```

张量格式：

```text
shape: [1, H, W]
dtype: float32
unit: meter
invalid value: 0.0
```

`max_depth` 默认来自数据集 `"depth-range"` 的上限；显式传入 `--max-depth` 时才覆盖。推理前会检查 RGB 与 raw depth 的 `H, W` 完全一致。

#### MDMModel 内部输入约定

`MDMModel.forward(...)` 要求 `depth is not None`。如果 `depth_in` 是 `[B, H, W]`，内部会扩成 `[B, 1, H, W]`。

`DINOv2_RGBD_Encoder` 内部会执行：

1. 将 image resize 到 token grid 对应的 `token_rows * 14`、`token_cols * 14`。
2. 使用模型内置 ImageNet mean/std normalize image。
3. 使用 nearest resize depth。
4. 将 `inf`、`nan` depth 置 0。
5. 以 `depth > 0.01` 构造深度有效 mask。
6. 默认 `remap_depth_in="log"`，即对有效深度取 `log(depth)`，无效位置仍置 0。

因此，当前 evaluation wrapper 不应再额外构造 inverse depth；传入的是 metric raw depth，是否取 log 由模型内部配置控制。

#### 当前预测输出格式

`output["depth"]` 被规范化为：

```text
shape: [H, W]
dtype: float32
unit: meter
file: <sample_name>.npy
```

若模型输出大小和 raw depth 不一致，`evaluation/infer.py` 会用 bilinear resize 回 raw depth 的 `H, W`。

`run_eval.sh` 默认设置 `SAVE_VIS=true`，会额外传入 `--save-vis` 并保存：

```text
file: <sample_name>_vis.jpg
content: RGB / raw depth / pred depth 横向拼接
```

默认 `apply_mask=false`，因此会保存 dense metric depth。若显式启用 `--apply-mask`，模型 mask 外的位置可能变为 `inf`；后续 `eval.py` 会把预测中的 `nan/inf` 从 valid mask 中排除。

### 4. 样本命名与预测落盘

当前和原始链路使用同一命名约定。

HAMMER：

```text
data/HAMMER/raw_data/scene12_traj1_1/polarization/rgb/000000.png
-> scene12_traj1_1#000000.npy
```

ClearPose：

```text
data/clearpose/set2/scene4/000709-color.png
-> set2#scene4#000709-color.npy
```

`infer.py` 保存 `.npy`，`eval.py` 用同一套规则读取 `.npy`。

### 5. 指标计算

当前 `evaluation/eval.py` 复用原始指标定义：

```text
L1
rmse_linear
abs_relative_difference
delta4_acc_105
delta5_acc110
delta1_acc
```

GT depth 读取流程：

```text
cv2.imread(..., IMREAD_UNCHANGED)
astype(float32) / 1000.0
valid_mask = min_depth <= gt <= max_depth
无效 GT 位置写成 min_depth
```

注意：`eval.py` 虽然解析了 `--depth-scale`，但实际仍固定使用：

```python
depth_scale = 1000.0
```

`min_depth` 和 `max_depth` 会在 eval 阶段被数据集 `"depth-range"` 覆盖。当前 `ALIGN=False`，不做 scale/shift alignment，直接按 metric depth 计算。

当前版本相对原始 `eval.py` 的主要工程改动是设备选择：

```text
原始 eval.py: batch["pred"].cuda()
当前 eval.py:  batch["pred"].to(DEVICE)
```

因此当前评估可在 CUDA、MPS 或 CPU 上运行；原始版本默认要求 CUDA。

## 原始 run_bs_eval_pipeline 处理 pipeline

原始入口是 `run_bs_eval_pipeline/run_eval.sh`：

```bash
bash run_eval.sh <model_path> [arch=vitl] [camera_type=d435] [resize_method=lower_bound] [is_disp=false] [cleanup_npy=true]
```

原始推理固定加载：

```python
from rgbddepth.dpt import RGBDDepth

model = RGBDDepth(**model_configs[encoder])
```

checkpoint 支持三种结构：

```text
checkpoint["model"]
checkpoint["state_dict"]
直接 state dict
```

原始模型输入由 `batch_image2tensor(...)` 构造：

```text
RGB image:  HWC, RGB, uint8/float, 0-255
depth:      HxW, metric depth 或 inverse depth
resize:     keep_aspect_ratio=True, ensure_multiple_of=14
normalize:  RGB /255 后 ImageNet mean/std
concat:     [B, 4, H', W']
```

其中第 4 个通道的含义由 `is_disp` 控制：

```text
is_disp=false: 输入 metric raw depth，模型输出被当作 metric depth 保存
is_disp=true:  输入 inverse depth，模型输出被当作 inverse depth，保存前执行 pred = 1 / pred_disp
```

原始 `load_images(...)` 中 raw depth 的预处理较简单：

```text
depth = raw_depth / depth_scale
depth > max_depth 的位置置 0
simi_depth[depth > 0] = 1 / depth
```

原始 `max_depth` 推理默认是 `6.0`，`run_eval.sh` 没有按数据集 `"depth-range"` 传入 max depth。因此在 ClearPose 这类有效范围 `[0.3, 1.5]` 的数据上，原始推理输入和当前推理输入的深度截断策略不同；但 eval 阶段二者都会使用数据集 `"depth-range"` 作为 GT valid mask。

## 主要相同点

| 维度 | 相同点 |
| --- | --- |
| 数据集对象 | `HAMMERDataset` / `ClearPoseDataset` 逻辑一致 |
| dataset 返回值 | 都返回 `(rgb_path, raw_depth_path, gt_depth_path)` |
| 支持数据集 | 都支持 HAMMER 和 ClearPose JSONL |
| raw type | HAMMER 支持 `d435/l515/tof`，ClearPose 只支持 `d435` |
| 预测中间产物 | 都按样本保存 `.npy` |
| 样本命名 | HAMMER 用 `scene#frame`，ClearPose 用 `set#scene#frame-stem` |
| eval 输入 | 都只读预测目录中的 `.npy`，不重新跑模型 |
| 指标定义 | `utils/metric.py` 内容一致 |
| alignment | 默认 `ALIGN=False`，不做 scale/shift 对齐 |
| 输出指标 | 都输出 per-sample CSV 和 mean JSON |

## 主要差异

| 维度 | 当前 `evaluation` | 原始 `run_bs_eval_pipeline` | 影响 |
| --- | --- | --- | --- |
| 模型类 | `mdm.model.v2.MDMModel` | `rgbddepth.dpt.RGBDDepth` | checkpoint 结构和输入接口不兼容 |
| checkpoint 结构 | 需要 `model_config` 和 `model` | 支持 `model` / `state_dict` / 直接 state dict | 原始 RGBDDepth checkpoint 不能直接给当前 `MDMModel` 使用 |
| 推理 API | `model.infer(image, depth_in=...)` | `model.forward(batch_tensor)` | 当前走模型官方高层推理接口 |
| 模型输入 shape | RGB `[1,3,H,W]`，depth `[1,H,W]` | 拼接后 `[B,4,H',W']` | 当前 wrapper 的 `BATCH_SIZE` 不等于模型真实 batch |
| RGB 预处理 | wrapper 只做 RGB 转换和 `/255` | resize 后 `/255` + ImageNet normalize | 当前 normalize 和 resize 由 `MDMModel` 内部处理 |
| depth 预处理 | metric depth，过滤 `nan/inf/<=0/>max_depth` 后置 0 | metric depth 或 inverse depth，只显式截断 `>max_depth` | 当前输入更严格，且不支持 `is_disp` |
| depth 上限 | 默认使用数据集 `"depth-range"` 上限 | 推理默认 `6.0` | ClearPose 等短距离数据的 raw 输入截断不同 |
| resize 策略 | wrapper 不 resize，模型内部按 token 数 resize | wrapper 使用 `Resize(..., ensure_multiple_of=14)` | 两者模型看到的实际空间采样策略不同 |
| batch 推理 | DataLoader 可 batch，但模型逐图循环推理 | 真正 batch forward | 当前吞吐较低，但接口更贴近 MDMModel |
| 输出语义 | 直接 metric depth meter | `is_disp=false` 为 metric；`is_disp=true` 保存前 inverse 转 metric | 当前始终假定输出为 metric depth |
| mask | 默认不应用模型 mask，可选 `--apply-mask` | 无等价 mask 逻辑 | 启用 mask 会影响 eval valid pixels |
| intrinsics | 传 `None`，不输出 point cloud | 不使用 intrinsics | 当前只评 depth，不评 point cloud |
| 设备选择 | infer 支持 `auto/cuda/mps/cpu`；eval 支持 CUDA/MPS/CPU | infer 自动选择；eval 硬 `.cuda()` | 当前更适合本地 smoke check |
| CLI 默认 checkpoint | 有项目内默认 `ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt` | 必须传入 model path | 当前是项目内固定主模型适配 |
| 输出目录默认 | `evaluation/output/<dataset>_<timestamp>` | `<checkpoint_dir>/<dataset>_<checkpoint_stub>_data_<camera_type>` | 当前不会写到 checkpoint 目录旁 |
| 可视化默认 | `SAVE_VIS=true`，默认保存 `*_vis.jpg` | 需要显式传 `--save-vis` | 当前默认输出文件更多 |
| cleanup 默认 | `false` | `true` | 当前默认保留逐样本 `.npy` 方便复查 |

## 对结果可比性的影响

1. 当前和原始 eval 指标定义一致，因此如果 `.npy` 都是同尺寸、同单位的 metric depth，后处理指标具有可比性。
2. 推理结果本身不应被视为同一模型链路的公平替换，因为模型类、checkpoint 结构、输入归一化、resize 策略和 depth 表示都不同。
3. 对 HAMMER 默认 `depth-range=[0.1, 6.0]` 的场景，当前推理 `max_depth` 默认值与原始 `6.0` 基本一致。
4. 对 ClearPose 默认 `depth-range=[0.3, 1.5]` 的场景，当前推理会把 raw depth 中 `>1.5m` 的输入置 0，而原始推理默认只截断 `>6.0m`；这会改变模型可见的 raw depth 输入。
5. 当前默认不应用 `MDMModel` mask，保持 dense prediction；如果开启 `--apply-mask`，指标有效像素会因预测 `inf` 被进一步排除，结果不能直接和默认配置比较。

## 使用与迁移注意事项

1. 若评估 LingBot-Depth v0.5，使用当前 `evaluation`；若评估原 Camera Depth / RGBDDepth checkpoint，使用 `run_bs_eval_pipeline`。
2. 当前 `evaluation` 不支持 `is_disp=true` 的 inverse depth 输入输出语义。
3. 当前 `BATCH_SIZE` 只影响 DataLoader 取样分组，不会让 `MDMModel` 真正批量 forward；性能评估时不要把它和原始 batch pipeline 直接对比。
4. 两套 eval 都假设 depth 文件单位为 millimeter，并固定除以 `1000.0` 转 meter。
5. ClearPose 的 RGB、raw depth、GT depth 通过排序后的 glob 对齐，代码没有额外校验三类文件数量和 frame id 是否完全一致；如果数据目录不规范，两套 pipeline 都可能静默错配。
6. 当前 `evaluation` 的 `eval.py` 已支持 CPU/MPS，但完整模型推理仍建议在 CUDA 环境跑；本地更适合做 import、dataset 和小样本 smoke check。
