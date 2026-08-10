from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class EvaluationSample:
    """One RGB/raw-depth evaluation sample."""

    sample_id: str
    subset: str
    rgb_path: Path
    raw_depth_path: Path
    gt_depth_path: Optional[Path]
    depth_scale: float
    min_depth: float
    max_depth: float
    allow_evaluation_resize: bool = False
    expected_shape: Optional[Tuple[int, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True)
class RunConfig:
    dataset: str
    stage: str
    run_dir: Path
    model_path: Optional[str]
    device: str = "auto"
    resolution_level: int = 9
    batch_size: int = 1
    num_workers: int = 0
    use_fp16: bool = False
    apply_mask: bool = False
    save_visualizations: bool = True
    cleanup_predictions: bool = False
    max_samples: Optional[int] = None
    visualization_min_depth: float = 0.1
    visualization_max_depth: float = 5.0


@dataclass(frozen=True)
class LoadedSample:
    sample: EvaluationSample
    rgb: Any
    raw_depth: Any
    gt_depth: Any = None
