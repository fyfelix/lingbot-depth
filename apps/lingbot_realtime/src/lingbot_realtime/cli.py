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
from lingbot_realtime.inference import MDMInferenceEngine, MockInferenceEngine
from lingbot_realtime.services import PersistenceService, RuntimeController
from lingbot_realtime.web import create_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LingBot-Depth snapshot RGB-D web measurement application",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", choices=("fixture", "realsense"), default="realsense")
    parser.add_argument("--inference-engine", choices=("mock", "mdm"), default="mdm")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-depth", type=float, default=6.0)
    parser.add_argument("--vis-min", type=float, default=0.1)
    parser.add_argument("--vis-max", type=float, default=5.0)
    parser.add_argument("--pred-vis-percentile-min", type=float, default=1.0)
    parser.add_argument("--pred-vis-percentile-max", type=float, default=99.0)
    parser.add_argument("--resolution-level", type=int, default=9)
    parser.add_argument("--no-mask", action="store_true")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview-fps", type=float, default=15.0)
    parser.add_argument("--ack-timeout", type=float, default=10.0)
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("apps/lingbot_realtime/runs"))
    return parser


def build_config(argv: Sequence[str] | None = None) -> AppConfig:
    args = build_parser().parse_args(argv)
    config = AppConfig(
        source=args.source,
        inference_engine=args.inference_engine,
        model_path=args.model_path,
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
        apply_mask=not args.no_mask,
        bind=args.bind,
        port=args.port,
        preview_fps=args.preview_fps,
        ack_timeout_sec=args.ack_timeout,
        save_results=args.save_results,
        output_root=args.output_root,
    )
    config.validate()
    return config


def build_runtime(config: AppConfig) -> RuntimeController:
    if config.source == "fixture":
        source = FixtureFrameSource(config.width, config.height, config.fps)
    else:
        source = RealSenseFrameSource(config.width, config.height, config.fps)

    if config.inference_engine == "mock":
        engine = MockInferenceEngine(max_depth_m=config.max_depth_m)
    else:
        assert config.model_path is not None
        engine = MDMInferenceEngine(
            config.model_path,
            device=config.device,
            resolution_level=config.resolution_level,
            apply_mask=config.apply_mask,
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
