# LingBot-Depth Evaluation

This directory provides the repository-level evaluation pipeline for HAMMER,
ClearPose, DREDS, iBims, and KITTI Depth Completion. The implementation is layered into dataset
adapters, shared inference and I/O, output management, and evaluator backends.
It always runs `mdm.model.v2.MDMModel` and stores metric depth in meters.

## Environment

The project uses [uv](https://docs.astral.sh/uv/) for dependency and virtual
environment management:

```bash
uv sync --extra evaluation --group dev
```

This creates `.venv` from `pyproject.toml` and `uv.lock`. Run commands through
`uv run`; do not install a separate `evaluation/requirements.txt`.

xFormers is not required. Variable-length masked depth-token sequences fall
back to per-sample PyTorch scaled dot-product attention when xFormers is not
installed.

## Commands

All datasets share one Python entry point:

```bash
uv run --extra evaluation python -m evaluation <dataset> [options]
```

### HAMMER

```bash
uv run --extra evaluation python -m evaluation hammer \
  --model-path ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt \
  --manifest data/HAMMER/test.jsonl \
  --camera d435
```

`--camera` accepts `d435`, `l515`, or `tof`. HAMMER raw and ground-truth depth
are expected to be 16-bit images using a scale of 1000 units per meter.

### ClearPose

```bash
uv run --extra evaluation python -m evaluation clearpose \
  --model-path ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt \
  --manifest data/clearpose/test.jsonl
```

Each manifest row describes a sequence using `rgb`, `rgb-suffix`,
`raw_depth-suffix`, `depth-suffix`, and `depth-range`.

### DREDS

```bash
uv run --extra evaluation python -m evaluation dreds \
  --model-path ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt \
  --known-manifest data/DREDS/test_std_catknown.jsonl \
  --novel-manifest data/DREDS/test_std_catnovel.jsonl
```

Use `--variants catknown`, `--variants catnovel`, or both. DREDS raw and GT
depth are EXR float images already expressed in meters.

### iBims

```bash
uv run --extra evaluation python -m evaluation ibims \
  --model-path ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt \
  --ibims-root data/ibims1 \
  --levels easy medium hard extreme
```

The iBims root must contain:

```text
imagelist.txt
ibims1_core_mat/
evaluation_scripts/evaluate_ibims.py
ibims1_synthetic_raw_depth/manifests/ibims_<level>.jsonl
```

Canonical predictions are written as NPY files. During evaluation they are
converted to official `*_results.mat` files and staged with the official GT.

### KITTI Depth Completion

Use the complete `val_selection_cropped` benchmark (1000 paired frames). The
manifest can be generated from the OpenDataLab directory without copying data:

```bash
python evaluation/prepare_kitti_jsonl.py \
  --dataset-root data/KITTI_depth_completion \
  --output data/KITTI_depth_completion/val_selection_cropped.jsonl
```

Run the metric-depth pipeline with the KITTI-specific adapter:

```bash
uv run --extra evaluation python -m evaluation kitti \
  --model-path ckpts/lingbot-depth-pretrain-vitl-14-v0.5.pt \
  --manifest data/KITTI_depth_completion/val_selection_cropped.jsonl \
  --raw-max-depth 80
```

KITTI PNG values are divided by 256. Zero GT pixels are invalid, GT depth is
evaluated from `1/256` to infinity, and no scale/shift alignment is applied.
The raw Velodyne input is independently clipped to 80 m by default, matching
AS-Depth's `run_bs_kitti.sh -> eval_mp.py` semantics. With visualizations
enabled, every KITTI prediction produces both a colorized depth image under
`kitti_visualization/predictions/<scene>/<frame>.jpg` and a 3D point cloud under
`kitti_visualization/pointclouds/<scene>/<frame>.jpg`, in addition to the normal
comparison preview.

KITTI metrics use AS-Depth's float32 Torch implementation and per-sample
valid-mask semantics: only non-finite predictions are removed from valid GT
pixels. Finite zero and negative predictions therefore remain in MAE, RMSE,
absolute-relative, and delta calculations. Other dataset evaluators retain the
shared positive-depth mask.

For the legacy ten-positional-argument interface, use
`bash scripts/infer/run_bs_kitti.sh ...` after `conda activate lingbot-depth`.

## Common options

- `--stage all|infer|evaluate` selects the pipeline stage; default is `all`.
- `--run-dir` selects a run directory and is required for evaluate-only runs.
- `--device auto|cuda|mps|cpu` defaults to the best available device.
- `--resolution-level` defaults to `9`.
- `--batch-size` controls input loading batches; model inference remains
  per-sample to support variable image sizes.
- `--num-workers` controls parallel data loading.
- `--use-fp16` enables CUDA autocast; it is ignored on non-CUDA devices.
- `--apply-mask` applies the model-predicted validity mask.
- `--max-samples N` limits each subset independently for smoke testing.
- `--save-visualizations` and `--no-save-visualizations` control all
  visualizations. When enabled, KITTI produces both depth-map and 3D point-cloud
  visualizations for every prediction.
- `--cleanup-predictions` removes canonical NPY files only after evaluation
  succeeds.

Convenience wrappers are available in `evaluation/scripts/`; they only forward
arguments to `uv run ... python -m evaluation`.

## Output layout

When `--run-dir` is omitted, a new directory is created under:

```text
outputs/evaluation/<dataset>/<model_stem>_<YYYYMMDD_HHMMSS>/
```

Each run uses the following versioned layout:

```text
run.json
predictions/<subset>/<relative_sample_path>.npy
visualizations/<subset>/<relative_sample_path>_vis.jpg       # when enabled
kitti_visualization/predictions/<scene>/<frame>.jpg          # KITTI, when enabled
kitti_visualization/pointclouds/<scene>/<frame>.jpg          # KITTI, when enabled
metrics/per_sample.csv                # HAMMER, ClearPose, DREDS
metrics/summary.csv
metrics/summary.json
official/<subset>/predictions/*.mat   # iBims only
official/<subset>/workspace/          # iBims only
official/<subset>/evaluator.log       # iBims only
```

Subsets are camera names for HAMMER, `default` for ClearPose and KITTI,
`catknown`/`catnovel` for DREDS, and difficulty levels for iBims. Predictions
are `float32` metric depth; invalid values are represented as `NaN`.

Standard evaluation reports MAE, RMSE, absolute relative error, and delta
accuracy at 1.05, 1.10, and 1.25. iBims retains the metric names emitted by its
official evaluator.
