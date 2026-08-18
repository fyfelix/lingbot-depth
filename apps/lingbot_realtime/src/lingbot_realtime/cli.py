from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path
from typing import Sequence

import uvicorn

from lingbot_realtime.camera import FixtureFrameSource, RealSenseFrameSource
from lingbot_realtime.config import AppConfig
from lingbot_realtime.inference import (
    MDMInferenceEngine,
    MockInferenceEngine,
    TensorRTInferenceEngine,
)
from lingbot_realtime.services import PersistenceService, RuntimeController
from lingbot_realtime.web import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LingBot-Depth D435 continuous realtime inference and WebGL service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=("fixture", "realsense"), default="realsense")
    parser.add_argument("--backend", choices=("auto", "mock", "torch", "tensorrt"), default="auto")
    parser.add_argument(
        "--inference-engine",
        choices=("auto", "mock", "mdm", "torch", "tensorrt"),
        dest="legacy_backend",
        help="Compatibility alias for --backend (mdm maps to torch)",
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--vis-min", type=float, default=0.1)
    parser.add_argument("--vis-max", type=float, default=5.0)
    parser.add_argument("--pred-vis-percentile-min", type=float, default=1.0)
    parser.add_argument("--pred-vis-percentile-max", type=float, default=99.0)
    parser.add_argument("--resolution-level", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--no-auto-connect", action="store_true")
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview-fps", "--stream-fps", type=float, default=15.0)
    parser.add_argument("--ack-timeout", type=float, default=10.0)
    parser.add_argument("--send-timeout", type=float, default=2.0)
    parser.add_argument("--cloud-stride", type=int, default=2)
    parser.add_argument("--cloud-point-budget", type=int, default=180_000)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument(
        "--record-root", type=Path, default=Path("apps/lingbot_realtime/runs/recordings")
    )
    parser.add_argument("--record-session-id")
    parser.add_argument("--record-overwrite", action="store_true")
    parser.add_argument("--max-record-frames", type=int, default=0)
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("apps/lingbot_realtime/runs"))
    return parser


def build_config(argv: Sequence[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    backend = args.legacy_backend or args.backend
    if backend == "mdm":
        backend = "torch"
    if backend == "auto":
        if args.engine is not None:
            backend = "tensorrt"
        elif args.model_path:
            backend = "torch"
    config = AppConfig(
        source=args.source,
        inference_engine=backend,
        model_path=args.model_path,
        engine_path=args.engine,
        manifest_path=args.manifest,
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_depth_m=args.max_depth,
        vis_min_m=args.vis_min,
        vis_max_m=args.vis_max,
        pred_vis_percentile_min=args.pred_vis_percentile_min,
        pred_vis_percentile_max=args.pred_vis_percentile_max,
        resolution_level=args.resolution_level,
        num_tokens=args.num_tokens,
        apply_mask=not args.no_mask,
        auto_connect=not args.no_auto_connect,
        inference_enabled=not args.no_inference,
        bind=args.bind,
        port=args.port,
        preview_fps=args.preview_fps,
        ack_timeout_sec=args.ack_timeout,
        send_timeout_sec=args.send_timeout,
        cloud_stride=args.cloud_stride,
        cloud_point_budget=args.cloud_point_budget,
        record_enabled=not args.no_record,
        record_root=args.record_root,
        record_session_id=args.record_session_id,
        record_overwrite=args.record_overwrite,
        max_record_frames=args.max_record_frames,
        save_results=args.save_results,
        output_root=args.output_root,
    )
    config.validate()
    return config


def build_runtime(config: AppConfig) -> RuntimeController:
    source = (
        FixtureFrameSource(config.width, config.height, config.fps)
        if config.source == "fixture"
        else RealSenseFrameSource(config.width, config.height, config.fps)
    )
    engine = None
    if config.inference_engine == "mock":
        engine = MockInferenceEngine(max_depth_m=config.max_depth_m)
    elif config.inference_engine in {"torch", "mdm"}:
        assert config.model_path is not None
        engine = MDMInferenceEngine(
            config.model_path,
            device=config.device,
            resolution_level=config.resolution_level,
            num_tokens=config.num_tokens,
            apply_mask=config.apply_mask,
            max_depth_m=config.max_depth_m,
        )
    elif config.inference_engine == "tensorrt":
        assert config.engine_path is not None
        engine = TensorRTInferenceEngine(
            config.engine_path,
            manifest_path=config.manifest_path,
            device=config.device,
            max_depth_m=config.max_depth_m,
        )
    persistence = PersistenceService(
        config.save_results,
        config.output_root,
        config.max_depth_m,
        depth_viz=config.depth_viz_config(),
    )
    return RuntimeController(config, source, engine, persistence)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        config = build_config(argv)
    except ValueError as exc:
        build_parser().error(str(exc))
        return 2
    runtime = build_runtime(config)
    app = create_app(runtime)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=config.bind,
            port=config.port,
            log_level="info",
            ws="websockets-sansio",
            ws_ping_interval=None,
        )
    )

    def quit_watcher() -> None:
        while not runtime.quit_requested and not server.should_exit:
            time.sleep(0.2)
        if runtime.quit_requested:
            server.should_exit = True

    threading.Thread(target=quit_watcher, daemon=True, name="quit-watcher").start()
    logging.info("LingBot Realtime: http://%s:%d", config.bind, config.port)
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("Interrupted")
    finally:
        runtime.shutdown()
    return 0
