from __future__ import annotations

import time
from typing import Any

import numpy as np

from lingbot_realtime.domain import InferenceResult, RGBDFrame
from lingbot_realtime.inference.preprocessing import sanitize_metric_depth


class MDMInferenceEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        resolution_level: int = 9,
        apply_mask: bool = True,
        max_depth_m: float = 6.0,
    ) -> None:
        self.model_path = model_path
        self.requested_device = device
        self.resolution_level = int(resolution_level)
        self.apply_mask = bool(apply_mask)
        self.max_depth_m = float(max_depth_m)
        self._torch: Any = None
        self._device: Any = None
        self._model: Any = None

    @property
    def name(self) -> str:
        return "mdm"

    @property
    def device_name(self) -> str:
        return str(self._device) if self._device is not None else self.requested_device

    def _select_device(self, torch: Any) -> Any:
        if self.requested_device != "auto":
            device = torch.device(self.requested_device)
            if device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is unavailable")
            if device.type == "mps" and not torch.backends.mps.is_available():
                raise RuntimeError("MPS was requested but is unavailable")
            return device
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self) -> None:
        import torch
        from mdm.model.v2 import MDMModel

        self._torch = torch
        self._device = self._select_device(torch)
        self._model = MDMModel.from_pretrained(self.model_path).to(self._device).eval()

    def infer(self, frame: RGBDFrame) -> InferenceResult:
        if self._model is None or self._torch is None or self._device is None:
            raise RuntimeError("MDM inference engine is not loaded")
        torch = self._torch
        started = time.perf_counter()
        image = (
            torch.from_numpy(frame.color_rgb.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self._device)
        )
        depth_input = sanitize_metric_depth(frame.depth_m, self.max_depth_m)
        depth = torch.from_numpy(depth_input).unsqueeze(0).to(self._device)
        intrinsics = (
            torch.from_numpy(frame.intrinsics.normalized_matrix()).unsqueeze(0).to(self._device)
        )
        use_fp16 = self._device.type == "cuda"
        output = self._model.infer(
            image,
            depth_in=depth,
            intrinsics=intrinsics,
            resolution_level=self.resolution_level,
            apply_mask=self.apply_mask,
            use_fp16=use_fp16,
        )
        if "depth" not in output:
            raise RuntimeError("MDMModel.infer did not return depth")
        pred = output["depth"].squeeze().detach().float().cpu().numpy().astype(np.float32)
        if pred.shape != frame.depth_m.shape:
            import cv2

            pred = cv2.resize(
                pred,
                (frame.depth_m.shape[1], frame.depth_m.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        valid = np.isfinite(pred) & (pred > 0) & (pred <= self.max_depth_m)
        pred = np.where(valid, pred, 0.0).astype(np.float32)

        points_value = output.get("points")
        if points_value is not None:
            points = points_value.squeeze().detach().float().cpu().numpy().astype(np.float32)
            if points.shape[:2] != pred.shape:
                from lingbot_realtime.inference.mock import depth_to_points

                points = depth_to_points(pred, frame)
        else:
            from lingbot_realtime.inference.mock import depth_to_points

            points = depth_to_points(pred, frame)
        points[~valid] = np.inf
        return InferenceResult(
            pred_depth_m=np.ascontiguousarray(pred),
            points=np.ascontiguousarray(points),
            elapsed_sec=time.perf_counter() - started,
        )

    def close(self) -> None:
        self._model = None
        if self._torch is not None and self._device is not None and self._device.type == "cuda":
            self._torch.cuda.empty_cache()
