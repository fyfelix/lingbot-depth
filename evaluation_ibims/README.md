# LingBot-Depth iBims 官方评估

`evaluation_ibims/` 是当前项目专用的 iBims 官方评估适配目录。它只消费已有
synthetic raw depth manifest，并使用 `mdm.model.v2.MDMModel` 做推理；不会包含
或调用 `generate_raw_depth.py`、`validate_block_mask.py`。

## 前置条件

iBims 数据集目录默认是：

```text
data/ibims1
```

运行前需要已经存在 synthetic manifest：

```text
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_easy.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_medium.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_hard.jsonl
data/ibims1/ibims1_synthetic_raw_depth/manifests/ibims_extreme.jsonl
```

完整官方评估还需要数据集自带文件：

```text
data/ibims1/imagelist.txt
data/ibims1/ibims1_core_mat/
data/ibims1/evaluation_scripts/evaluate_ibims.py
```

当前 `.venv` 已满足项目推理依赖，但完整官方 evaluator 还需要
`scikit-image` 和 `scikit-learn`。

## 一站式运行

在项目根目录运行：

```bash
./evaluation_ibims/run_all.sh
```

等价于：

```bash
./evaluation_ibims/run_all.sh ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt
```

`run_all.sh` 会优先使用 `.venv/bin/python`。额外参数会透传给
`evaluation_ibims/run_all.py`：

```bash
./evaluation_ibims/run_all.sh ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt \
  --levels easy \
  --max-samples 1 \
  --skip-eval
```

## Python 入口

```bash
.venv/bin/python evaluation_ibims/run_all.py \
  --model-path ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt \
  --ibims-root data/ibims1 \
  --levels easy medium hard extreme \
  --batch-size 1 \
  --device auto \
  --resolution-level 9
```

常用参数：

```text
--run-dir <dir>          指定输出根目录
--use-fp16              CUDA 上启用模型 autocast
--max-samples <N>       每个 difficulty 只跑前 N 个样本
--skip-infer            跳过推理，使用 --run-dir 下已有 predictions
--skip-eval             跳过官方评估，只生成 MAT prediction
```

## 输出结构

默认输出目录：

```text
evaluation_ibims/output/ibims_<model_stem>_<YYYYMMDD_HHMMSS>/
```

主要内容：

```text
predictions/<level>/<sample>_results.mat
predictions/<level>/infer_args.json
official_eval/<level>/workspace/
official_eval/<level>/official_eval_stdout.txt
eval_summary.csv
eval_summary.json
```

每个 prediction MAT 包含变量 `pred_depths`：

```text
shape: 480x640
dtype: float32
unit: meter
invalid prediction: NaN
```

## 推理处理约定

- RGB 使用 OpenCV 读取，BGR 转 RGB，并归一化到 `[0, 1]`。
- raw depth 使用 manifest 中的 `depth_scale`，默认 `65535 / 50`，转换为 meter。
- raw depth 中非有限值、`<= 0`、超过 manifest `depth-range` 上限的点会置为 `0`。
- 模型固定为 `mdm.model.v2.MDMModel.from_pretrained(...)`。
- 调用 `MDMModel.infer(..., intrinsics=None, apply_mask=False)`，不做 disparity 或 inverse depth 转换，不做 alignment。
