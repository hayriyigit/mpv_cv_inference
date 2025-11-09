from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import signal
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import zmq
import zmq.asyncio
from loguru import logger
from PIL import Image

from src.lib.mpv import MPV
from src.shared.models import (
    BoundingBox,
    ControlCommandType,
    Detection,
    FrameSyncMetadata,
    InferenceMetadata,
    InferenceResult,
    ModelControlCommand,
    ModelStatus,
    ServerStatus,
    Topic,
)


@dataclass
class FrameSample:
    """Container for the decoded frame returned by mpv."""

    width: int
    height: int
    stride: int
    format: str
    pixels: bytes
    timestamp: float
    frame_index: int
    captured_at: float

    def to_image(self) -> Image.Image:
        return Image.fromarray(self.to_numpy(), mode="RGBA")

    def to_numpy(self) -> np.ndarray:
        fmt = self.format.lower()
        if fmt not in {"bgra", "bgr0", "rgba", "rgb0"}:
            raise ValueError(f"Unsupported screenshot-raw format: {self.format}")

        bytes_per_pixel = 4
        row_pixels = self.stride // bytes_per_pixel
        buffer = np.frombuffer(self.pixels, dtype=np.uint8)
        try:
            array = buffer.reshape((self.height, row_pixels, bytes_per_pixel))
        except ValueError as exc:  # pragma: no cover - indicates corrupted buffer
            raise ValueError(
                f"Failed to reshape frame buffer (len={buffer.size}, height={self.height}, stride={self.stride})"
            ) from exc

        array = array[:, : self.width, :].copy()

        if fmt in {"bgra", "bgr0"}:
            array = array[:, :, [2, 1, 0, 3]]
            if fmt == "bgr0":
                array[:, :, 3] = 255
        elif fmt == "rgb0":
            array[:, :, 3] = 255
        # "rgba" already in correct order

        return array

    def digest(self, digest_size: int = 16) -> str:
        return hashlib.blake2b(self.pixels, digest_size=digest_size).hexdigest()


class BaseModelRunner:
    name: str = "base"
    version: str = "0.0"

    def __init__(self) -> None:
        self.enabled: bool = True

    def infer(self, frame_rgba: np.ndarray) -> List["ModelDetection"]:
        raise NotImplementedError

    def make_metadata(self) -> InferenceMetadata:
        return InferenceMetadata(
            model_name=self.name,
            version=self.version,
            parameters=self.current_parameters(),
        )

    def current_parameters(self) -> Dict[str, Any]:
        return {}

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        raise NotImplementedError


@dataclass
class ModelDetection:
    bbox: BoundingBox
    confidence: float
    extra: Dict[str, Any] = field(default_factory=dict)


class BrightnessHotspotModel(BaseModelRunner):
    name = "brightness_hotspot"
    version = "0.1"

    def __init__(self, sensitivity: float = 0.65, min_pixels: int = 2500) -> None:
        super().__init__()
        self.sensitivity = sensitivity
        self.min_pixels = min_pixels

    def current_parameters(self) -> Dict[str, Any]:
        return {"sensitivity": self.sensitivity, "min_pixels": self.min_pixels}

    def infer(self, frame_rgba: np.ndarray) -> List[ModelDetection]:
        grayscale = np.mean(frame_rgba[:, :, :3], axis=2)
        threshold = np.clip(self.sensitivity * 255.0, 0, 255)
        mask = grayscale >= threshold
        if mask.sum() < self.min_pixels:
            return []

        ys, xs = np.where(mask)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        coverage = mask.mean()
        bbox = BoundingBox(
            x=x_min,
            y=y_min,
            width=max(1, x_max - x_min + 1),
            height=max(1, y_max - y_min + 1),
        )
        confidence = float(np.clip(coverage / 0.5, 0.0, 1.0))
        extra = {"mean_brightness": float(grayscale.mean()), "coverage": float(coverage)}
        return [ModelDetection(bbox=bbox, confidence=confidence, extra=extra)]

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        if "sensitivity" in parameters:
            self.sensitivity = float(parameters["sensitivity"])
        if "min_pixels" in parameters:
            self.min_pixels = int(parameters["min_pixels"])


@dataclass
class ManagedModel:
    runner: BaseModelRunner
    enabled: bool = True
    last_timestamp: Optional[float] = None
    last_hash: Optional[str] = None

    def infer(self, frame: np.ndarray) -> List[ModelDetection]:
        if not self.enabled:
            return []
        return self.runner.infer(frame)

    def metadata(self) -> InferenceMetadata:
        return self.runner.make_metadata()

    def update_parameters(self, parameters: Dict[str, Any]) -> None:
        self.runner.update_parameters(parameters)

    def status(self) -> ModelStatus:
        return ModelStatus(
            model_name=self.runner.name,
            enabled=self.enabled,
            parameters=self.runner.current_parameters(),
            last_timestamp=self.last_timestamp,
            last_frame_hash=self.last_hash,
        )


class InferenceServer:
    def __init__(
        self,
        video_uri: str,
        pub_endpoint: str,
        ctrl_endpoint: str,
        sample_interval: float = 1 / 8,
        models: Optional[Iterable[BaseModelRunner]] = None,
    ) -> None:
        self.video_uri = video_uri
        self.pub_endpoint = pub_endpoint
        self.ctrl_endpoint = ctrl_endpoint
        self.sample_interval = sample_interval
        self._ctx = zmq.asyncio.Context.instance()
        self._pub_socket: Optional[zmq.asyncio.Socket] = None
        self._ctrl_socket: Optional[zmq.asyncio.Socket] = None
        self._mpv: Optional[MPV] = None
        self._frame_index = 0
        self._start_time = time.monotonic()
        self._running = asyncio.Event()
        self._running.clear()
        model_list = list(models or [BrightnessHotspotModel()])
        self.models: Dict[str, ManagedModel] = {
            model.name: ManagedModel(runner=model) for model in model_list
        }

    async def start(self) -> None:
        self._pub_socket = self._ctx.socket(zmq.PUB)
        self._pub_socket.bind(self.pub_endpoint)
        logger.info("Publishing inference stream on {}", self.pub_endpoint)

        self._ctrl_socket = self._ctx.socket(zmq.REP)
        self._ctrl_socket.bind(self.ctrl_endpoint)
        logger.info("Listening for control commands on {}", self.ctrl_endpoint)

        self._mpv = MPV(
            log_handler=self._log_mpv,
            start_event_thread=True,
            vo="gpu",
            gpu_context="auto",
            hwdec="auto-safe",
            audio="no",
            video_osd="no",
            keep_open="yes",
            idle="yes",
        )
        self._mpv['gpu-context'] = 'auto'
        self._mpv['vo'] = 'gpu'
        self._mpv['hwdec'] = 'auto-safe'
        self._mpv['aid'] = 'no'
        self._mpv['pause'] = 'no'
        self._mpv['osc'] = 'no'
        logger.info("Loading video {}", self.video_uri)
        self._mpv.loadfile(self.video_uri)
        try:
            self._mpv.wait_until_playing(timeout=15)
        except Exception as exc:  # noqa: BLE001
            logger.error("mpv failed to start playback: {}", exc)
            raise
        self._running.set()
        await self._publish_status()

    async def serve_forever(self) -> None:
        if self._running.is_set():
            raise RuntimeError("Server already running")
        await self.start()
        tasks = [
            asyncio.create_task(self._inference_loop(), name="inference"),
            asyncio.create_task(self._control_loop(), name="control"),
            asyncio.create_task(self._status_loop(), name="status"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Server cancellation requested")
        finally:
            for task in tasks:
                task.cancel()
            await self._cleanup()

    async def _cleanup(self) -> None:
        self._running.clear()
        if self._mpv is not None:
            logger.info("Shutting down mpv")
            self._mpv.terminate()
            self._mpv = None
        if self._pub_socket is not None:
            self._pub_socket.close()
            self._pub_socket = None
        if self._ctrl_socket is not None:
            self._ctrl_socket.close()
            self._ctrl_socket = None

    async def _status_loop(self) -> None:
        while self._running.is_set():
            await asyncio.sleep(5)
            await self._publish_status()

    async def _publish_status(self) -> None:
        if not self._pub_socket:
            return
        status = ServerStatus(
            video_uri=self.video_uri,
            models=[m.status() for m in self.models.values()],
            uptime_s=time.monotonic() - self._start_time,
            wallclock_time=time.time(),
        )
        await self._pub_socket.send_multipart(
            [Topic.STATUS.value.encode(), status.model_dump_json().encode()]
        )

    async def _control_loop(self) -> None:
        assert self._ctrl_socket is not None
        while True:
            raw = await self._ctrl_socket.recv()
            try:
                data = json.loads(raw.decode())
                command = ModelControlCommand.model_validate(data)
                reply = self._handle_control(command)
                await self._ctrl_socket.send_json(reply)
            except Exception as exc:  # noqa: BLE001
                logger.exception("Error handling control message: {}", exc)
                await self._ctrl_socket.send_json({"status": "error", "detail": str(exc)})

    def _handle_control(self, command: ModelControlCommand) -> Dict[str, Any]:
        model = self.models.get(command.model_name)

        if command.command == ControlCommandType.REQUEST_STATUS:
            if command.model_name == "*" or model is None:
                payload = [m.status().model_dump() for m in self.models.values()]
                return {"status": "ok", "models": payload}
            return {"status": "ok", "model": model.status().model_dump()}

        if model is None:
            return {"status": "error", "detail": f"Unknown model {command.model_name}"}

        if command.command == ControlCommandType.START_MODEL:
            model.enabled = True
            return {"status": "ok", "enabled": True}
        if command.command == ControlCommandType.STOP_MODEL:
            model.enabled = False
            return {"status": "ok", "enabled": False}
        if command.command == ControlCommandType.UPDATE_PARAMS:
            model.update_parameters(command.parameters)
            return {"status": "ok", "parameters": model.runner.current_parameters()}
        return {"status": "error", "detail": f"Unsupported command {command.command}"}

    async def _inference_loop(self) -> None:
        while self._running.is_set():
            await asyncio.sleep(self.sample_interval)
            sample = await asyncio.to_thread(self._capture_frame)
            if sample is None:
                continue
            if not self._pub_socket:
                continue
            frame_np = sample.to_numpy()
            frame_hash = sample.digest()
            timestamp = sample.timestamp
            sync = FrameSyncMetadata(
                video_uri=self.video_uri,
                timestamp=timestamp,
                frame_hash=frame_hash,
                frame_index=sample.frame_index,
            )
            for model in self.models.values():
                if not model.enabled:
                    continue
                try:
                    detections_model = model.infer(frame_np)
                except Exception as exc:  # noqa: BLE001
                    logger.exception(
                        "Model {} inference failed: {}", model.runner.name, exc
                    )
                    continue
                now = time.monotonic()
                if detections_model:
                    confidences = ", ".join(f"{det.confidence:.2f}" for det in detections_model)
                    logger.info(
                        "Model {} emitted {} detection(s) (hash={} conf=[{}])",
                        model.runner.name,
                        len(detections_model),
                        frame_hash[:12],
                        confidences,
                    )
                humannet_boxes = [
                    [
                        det.bbox.x,
                        det.bbox.y,
                        det.bbox.x + det.bbox.width,
                        det.bbox.y + det.bbox.height,
                        det.confidence,
                    ]
                    for det in detections_model
                ]
                inference_payload = InferenceResult(
                    sync=sync,
                    model=model.metadata(),
                    detections=[
                        Detection(
                            label=model.runner.name,
                            confidence=det.confidence,
                            bbox=det.bbox,
                            extra=det.extra,
                        )
                        for det in detections_model
                    ],
                    latency_ms=(now - sample.captured_at) * 1000.0,
                    server_time=now,
                    extra={
                        "frame_index": sample.frame_index,
                        "frame_shape": [sample.height, sample.width],
                        "humannet_results": humannet_boxes,
                        "othermodel_results": [],
                    },
                )
                model.last_timestamp = timestamp
                model.last_hash = frame_hash
                await self._pub_socket.send_multipart(
                    [Topic.INFERENCE.value.encode(), inference_payload.model_dump_json().encode()]
                )

    def _capture_frame(self) -> Optional[FrameSample]:
        if self._mpv is None:
            return None
        try:
            data = self._mpv.command("screenshot-raw", "video")
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to capture frame: {}", exc)
            return None
        if not data:
            return None
        timestamp = getattr(self._mpv, "time_pos", None)
        if timestamp is None:
            return None
        captured_at = time.monotonic()
        self._frame_index += 1
        return FrameSample(
            width=int(data["w"]),
            height=int(data["h"]),
            stride=int(data["stride"]),
            format=str(data["format"]),
            pixels=bytes(data["data"]),
            timestamp=float(timestamp),
            frame_index=self._frame_index,
            captured_at=captured_at,
        )

    def _log_mpv(self, loglevel: str, component: str, message: str) -> None:
        logger_opt = logger.opt(depth=1)
        logger_opt.debug("[MPV:{}] {} - {}", loglevel, component, message.strip())


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mpv inference server")
    parser.add_argument("--video", required=True, help="Path or URL of the video to decode")
    parser.add_argument("--pub", default="tcp://*:5556", help="PUB endpoint for inference stream")
    parser.add_argument(
        "--ctrl",
        default="tcp://*:5557",
        help="REP endpoint for control commands",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1 / 8,
        help="Seconds between frame samples for inference",
    )
    parser.add_argument(
        "--hotspot-sensitivity",
        type=float,
        default=0.5,
        help="Brightness threshold (0-1) for hotspot detection; lower values detect larger, darker regions",
    )
    parser.add_argument(
        "--hotspot-min-pixels",
        type=int,
        default=1500,
        help="Minimum number of bright pixels required before emitting a detection",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    hotspot_model = BrightnessHotspotModel(
        sensitivity=args.hotspot_sensitivity,
        min_pixels=args.hotspot_min_pixels,
    )
    server = InferenceServer(
        video_uri=args.video,
        pub_endpoint=args.pub,
        ctrl_endpoint=args.ctrl,
        sample_interval=args.sample_interval,
        models=[hotspot_model],
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def runner() -> None:
        await server.serve_forever()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(shutdown(loop, s)))

    try:
        loop.run_until_complete(runner())
    finally:
        loop.close()


async def shutdown(loop: asyncio.AbstractEventLoop, sig: signal.Signals) -> None:
    logger.info("Received signal {}: shutting down", sig.name)
    tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


if __name__ == "__main__":
    main()
