# LingBot Realtime D435

LingBot-Depth 的连续 RGB-D 推理、WebGL 可视化、录制、snapshot 测量和 TensorRT FP16
部署应用。相机只由一个 worker 持有；实时页面、点云页面、录制和 snapshot 都消费同一条最新帧流，
不会为浏览器或抓拍重复打开 D435。

## 页面与控制

- `/`：连续 RGB、raw depth、predicted depth 和预测点云主界面。
- `/pointcloud`：共享相机 worker 的 raw D435 点云。
- `/snapshot`：从最新完整预测冻结不可变副本，保留两点测量、重试和逐次持久化。
- `/status`：相机、模型、FPS、录制和错误状态。

推理速度低于相机速度时不会创建无界队列。worker 每次只发布最新完成结果；WebSocket 使用单帧
`frame_ack`，每个浏览器最多有一帧在途。

## 安装与测试

推荐使用项目 Conda 环境，并安装应用为 editable package：

```bash
conda activate lingbot-depth
python -m pip install -e "apps/lingbot_realtime[test,realsense,deploy]"
pytest -q apps/lingbot_realtime/tests
```

不连接相机、不下载模型的本地闭环：

```bash
lingbot-realtime --source fixture --backend mock --bind 127.0.0.1
```

无模型或 engine 时可以仅启动传感器流：

```bash
lingbot-realtime --source realsense --backend auto --no-inference
```

## PyTorch 与 TensorRT

PyTorch BF16 回退：

```bash
lingbot-realtime \
  --source realsense \
  --backend torch \
  --model-path robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  --device cuda \
  --resolution-level 0 \
  --num-tokens 1200
```

正式 TensorRT FP16 路径：

```bash
lingbot-realtime \
  --source realsense \
  --backend tensorrt \
  --engine runs/deploy/d435-fp16/model.engine \
  --manifest runs/deploy/d435-fp16/deployment.json \
  --device cuda
```

`--backend auto` 的选择顺序是：传入 `--engine` 时使用 TensorRT，传入 `--model-path` 时使用
PyTorch，否则为 sensor-only。旧 `--inference-engine mdm|mock` 仍可使用，其中 `mdm` 映射到
`torch`。

默认部署规格固定为：

- 输入 `rgbd_input: float16 [1,4,480,640]`；RGB 为 `[0,1]`，深度为米，invalid 为 `0`。
- 输出 `depth: float16 [1,480,640]`；应用边界转换为 float32 米制深度。
- 1200 tokens、`resolution_level=0`、关闭按深度有效性动态删除 token、保留预测 mask。

导出的图保持 FP16 输入、输出、权重和主要 GEMM/卷积；序列长度固定不做动态 token 删除，
并用静态 attention key mask 排除无效深度 token。Transformer 的 LayerNorm、Softmax、Add 和
layer-scale 累计链保留 FP32，并在边界插入 Cast。两项策略分别记录在 manifest 的
`static_depth_attention_mask` 和 `fp32_stability_policy` 字段中。

## 导出与构建

`lingbot-realtime-deploy` 提供三个子命令：

```bash
# FP32 ONNX -> 图级 FP16 ONNX
lingbot-realtime-deploy export \
  --model robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  --output runs/deploy/d435-fp16 \
  --device cuda

# Strongly Typed FP16 engine、timing cache、build log
lingbot-realtime-deploy build \
  --onnx runs/deploy/d435-fp16/model.fp16.onnx \
  --output runs/deploy/d435-fp16 \
  --trtexec trtexec

# 完整导出、检查、构建和 smoke benchmark
lingbot-realtime-deploy all \
  --model robbyant/lingbot-depth-pretrain-vitl-14-v0.5 \
  --output runs/deploy/d435-fp16 \
  --device cuda \
  --trtexec trtexec
```

产物目录包含：

```text
model.fp32.onnx
model.fp32.onnx.data
model.fp16.onnx
model.fp16.onnx.data
model.engine
timing.cache
build.log
deployment.json
```

`deployment.json` 记录 checkpoint hash、tensor 语义、token 配置、precision、ONNX/TensorRT
版本、GPU capability、benchmark 和所有产物 checksum。TensorRT engine 必须在兼容的 TensorRT
major 环境中构建；传入 manifest 后，major、FP16 IO、固定 shape 或 engine checksum 不兼容都会在
runtime 加载阶段明确失败。

## 相机、推理和录制控制

常用选项：

```text
--no-auto-connect
--no-inference
--preview-fps 15
--ack-timeout 10
--cloud-stride 2
--cloud-point-budget 180000
--no-record
--record-root apps/lingbot_realtime/runs/recordings
--max-record-frames 0
```

连续录制每个 session 固定写：

```text
rgb.mp4
raw_depth.npy     # uint16 millimeter (N,H,W)
pred_depth.npy    # float32 meter (N,H,W), invalid=0
frames.jsonl
meta.json
```

开始录制前必须连接相机并启用推理；录制期间不允许切换推理。停止录制或正常退出时流式 NPY
header 会回填真实帧数，文件可以直接用 `numpy.load` 重新读取。

snapshot 单次保存使用：

```bash
lingbot-realtime ... --save-results --output-root apps/lingbot_realtime/runs
```

每次 capture 保存 RGB、raw/pred metric depth、深度可视化、PLY、intrinsics、metadata 和测量结果。

## RealSense

Linux 上安装 `realsense` extra 后，应用会在实际选择 `--source realsense` 时惰性导入
`pyrealsense2`。默认请求 `640x480@30`，失败时依次尝试 15 FPS 和 6 FPS；深度对齐到 color，
使用设备报告的 depth scale 和 color intrinsics。

本机 NVIDIA ARM 环境使用 RSUSB 后端启动，避免 native UVC/V4L2 路径触发 xHCI 故障：

```bash
apps/lingbot_realtime/scripts/run_realsense_rsusb.sh
```

脚本默认使用 `asdepth` Conda 环境、`/home/asdepth/librealsense/build/Release` 和
`6 FPS + mock` 推理；可用 `LINGBOT_REALTIME_RSUSB_ENV`、
`LINGBOT_LIBREALSENSE_RSUSB_ROOT`、`LINGBOT_REALTIME_FPS`、`LINGBOT_REALTIME_PORT` 覆盖。

### USB 稳定性与故障恢复

- D435 优先直连主机 USB 3.x 端口，使用短的、带屏蔽的 USB 3 数据线；不要在相机和主机之间串接无源 Hub。USB 2 线可以作为带宽降级排障手段。
- 启动服务前先运行 `rs-enumerate-devices`。确认设备可见后再启动推理服务，先用
  `--no-inference` 或 `--backend mock` 验证相机链路。
- 运行期间观察 `curl http://127.0.0.1:8000/status` 和 `dmesg -w`。应用会在启动前快速枚举设备，启动失败后检查设备是否仍在总线，并以最长 30 秒的退避重试；连续 3 次失败后自动熔断，需点击“连接相机”才会再次尝试，避免反复重启 pipeline。
- 如果内核出现 `xHCI host controller ... assume dead`、`HC died` 或连续的 UVC `-110` 超时，拔插相机通常无法恢复已经失效的 USB 主控；应停止服务并重启主机，再检查 `lsusb -d 8086:0b07` 和 `rs-enumerate-devices`。
- 保持 librealsense、D435 固件和主机内核在经过验证的版本组合；升级其中任一项后，先做 1 分钟 sensor-only 采集，再启用模型和录制。

macOS 的源码构建与 USB 检查工具仍位于 `scripts/install_pyrealsense2_macos.sh`、
`scripts/check_realsense_macos.py` 和 `scripts/run_realsense_macos.sh`。

受管 Conda 启动可使用：

```bash
LINGBOT_REALTIME_ENGINE=/srv/lingbot/deploy/model.engine \
LINGBOT_REALTIME_MANIFEST=/srv/lingbot/deploy/deployment.json \
apps/lingbot_realtime/scripts/run_managed.sh
```

模型、ONNX、engine、timing cache 和录制目录已在仓库 `.gitignore` 中排除。
