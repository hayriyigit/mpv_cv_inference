from __future__ import annotations

import argparse
import asyncio
import hashlib
import signal
from collections import deque
from typing import Any, Deque, Dict, Iterable, Optional, Tuple

import zmq
import zmq.asyncio
from loguru import logger
from PIL import Image, ImageDraw

from src.lib.mpv import MPV
from src.shared.models import (
    ControlCommandType,
    InferenceResult,
    ModelControlCommand,
    ServerStatus,
    Topic,
)


class InferenceCache:
    """Cache inference results to support seeking and delayed overlay drawing."""

    def __init__(self, max_items: int = 1024, tolerance: float = 0.05) -> None:
        self.max_items = max_items
        self.tolerance = tolerance
        self._by_hash: Dict[str, InferenceResult] = {}
        self._timeline: Deque[Tuple[float, str]] = deque()

    def add(self, result: InferenceResult) -> None:
        key = result.sync.frame_hash
        if key in self._by_hash:
            # Update timestamp if the same hash reappears (e.g. seeking backwards)
            self._timeline.append((result.sync.timestamp, key))
            self._by_hash[key] = result
        else:
            self._by_hash[key] = result
            self._timeline.append((result.sync.timestamp, key))
        self._evict_if_needed()

    def _evict_if_needed(self) -> None:
        while len(self._timeline) > self.max_items:
            ts, key = self._timeline.popleft()
            stored = self._by_hash.get(key)
            if stored is not None and stored.sync.timestamp == ts:
                self._by_hash.pop(key, None)

    def lookup(self, timestamp: float) -> Optional[InferenceResult]:
        cutoff = timestamp - self.tolerance
        nearest: Optional[Tuple[float, InferenceResult]] = None
        for ts, key in reversed(self._timeline):
            if ts < cutoff:
                break
            result = self._by_hash.get(key)
            if result is None:
                continue
            if abs(result.sync.timestamp - timestamp) <= self.tolerance:
                if nearest is None or abs(result.sync.timestamp - timestamp) < abs(nearest[0] - timestamp):
                    nearest = (result.sync.timestamp, result)
        return nearest[1] if nearest else None

    def latest(self) -> Optional[InferenceResult]:
        for _, key in reversed(self._timeline):
            result = self._by_hash.get(key)
            if result:
                return result
        return None

    def get_by_hash(self, frame_hash: str) -> Optional[InferenceResult]:
        return self._by_hash.get(frame_hash)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._by_hash)


class InferenceClient:
    def __init__(
        self,
        video_uri: str,
        sub_endpoint: str,
        ctrl_endpoint: str,
        tolerance: float = 0.05,
    ) -> None:
        self.video_uri = video_uri
        self.sub_endpoint = sub_endpoint
        self.ctrl_endpoint = ctrl_endpoint
        self.cache = InferenceCache(tolerance=tolerance)
        self.context = zmq.asyncio.Context.instance()
        self.sub_socket: Optional[zmq.asyncio.Socket] = None
        self.ctrl_socket: Optional[zmq.asyncio.Socket] = None
        self.ctrl_lock = asyncio.Lock()
        self._mpv: Optional[MPV] = None
        self._overlay = None
        self._running = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._status: Optional[ServerStatus] = None
        self._last_time_pos: Optional[float] = None
        self._last_hash_rendered: Optional[str] = None
        self._last_hash_verified: Optional[str] = None
        self._render_lock = asyncio.Lock()

    async def start(self) -> None:
        self.sub_socket = self.context.socket(zmq.SUB)
        self.sub_socket.connect(self.sub_endpoint)
        for topic in (Topic.INFERENCE.value, Topic.STATUS.value):
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        logger.info("Subscribed to inference stream at {}", self.sub_endpoint)

        self.ctrl_socket = self.context.socket(zmq.REQ)
        self.ctrl_socket.connect(self.ctrl_endpoint)
        logger.info("Control channel connected to {}", self.ctrl_endpoint)

        self._mpv = MPV(
            log_handler=self._log_mpv,
            start_event_thread=True,
            vo="gpu",
            gpu_context="auto",
            hwdec="auto-safe",
            idle="yes",
            keep_open="yes",
        )
        self._mpv['hwdec'] = 'auto-safe'
        self._mpv['vo'] = 'gpu'
        self._mpv['gpu-context'] = 'auto'
        self._mpv['osc'] = 'yes'
        self._mpv['sub-auto'] = 'no'
        self._mpv['osd-level'] = '1'
        self._overlay = self._mpv.create_image_overlay()
        logger.info("Loading video {}", self.video_uri)
        self._mpv.loadfile(self.video_uri)
        self._mpv.wait_until_playing(timeout=15)
        self._running.set()

    async def serve_forever(self) -> None:
        if self._running.is_set():
            raise RuntimeError("Client already running")
        await self.start()
        assert self._mpv is not None
        self._loop = asyncio.get_running_loop()
        self._register_mpv_observers()
        tasks = [
            asyncio.create_task(self._receive_loop(), name="receive"),
            asyncio.create_task(self._stdin_loop(), name="stdin"),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Client shutdown requested")
        finally:
            for task in tasks:
                task.cancel()
            await self._cleanup()

    def _register_mpv_observers(self) -> None:
        assert self._mpv is not None

        @self._mpv.property_observer('time-pos')
        def _time_pos_observer(_name: str, value: Any) -> None:
            if value is None or self._loop is None:
                return
            self._loop.call_soon_threadsafe(self._schedule_time_update, float(value))

        @self._mpv.event_callback('seek', 'playback-restart')
        def _seek_observer(_event) -> None:
            if self._loop is None:
                return
            current = getattr(self._mpv, 'time_pos', None)
            if current is not None:
                self._loop.call_soon_threadsafe(self._schedule_time_update, float(current))

    def _schedule_time_update(self, value: float) -> None:
        asyncio.create_task(self._handle_time_pos(value))

    async def _receive_loop(self) -> None:
        assert self.sub_socket is not None
        while self._running.is_set():
            try:
                topic_bytes, payload = await self.sub_socket.recv_multipart()
            except zmq.ZMQError as exc:
                if exc.errno == zmq.ETERM:
                    break
                raise
            topic = topic_bytes.decode()
            if topic == Topic.INFERENCE.value:
                result = InferenceResult.model_validate_json(payload)
                self.cache.add(result)
                if (
                    self._last_time_pos is not None
                    and abs(self._last_time_pos - result.sync.timestamp) <= self.cache.tolerance
                ):
                    asyncio.create_task(self._handle_time_pos(self._last_time_pos))
            elif topic == Topic.STATUS.value:
                self._status = ServerStatus.model_validate_json(payload)
                logger.debug(
                    "Server status: models={} uptime={:.1f}s",
                    [m.model_name for m in self._status.models],
                    self._status.uptime_s,
                )

    async def _stdin_loop(self) -> None:
        if not self.ctrl_socket:
            return
        while self._running.is_set():
            try:
                cmd = await asyncio.to_thread(input, "client> ")
            except (EOFError, KeyboardInterrupt):
                logger.info("Input closed, stopping client")
                self._running.clear()
                break
            stripped = cmd.strip()
            if not stripped:
                continue
            if stripped in {"quit", "exit"}:
                self._running.clear()
                break
            if stripped == "status":
                await self._request_status()
                continue
            if stripped.startswith("start "):
                _, model = stripped.split(maxsplit=1)
                await self._send_control(ModelControlCommand(command=ControlCommandType.START_MODEL, model_name=model))
                continue
            if stripped.startswith("stop "):
                _, model = stripped.split(maxsplit=1)
                await self._send_control(ModelControlCommand(command=ControlCommandType.STOP_MODEL, model_name=model))
                continue
            if stripped.startswith("set "):
                parts = stripped.split()
                if len(parts) < 3:
                    logger.warning("Usage: set <model> key=value [key=value ...]")
                    continue
                model = parts[1]
                params: Dict[str, Any] = {}
                for chunk in parts[2:]:
                    if '=' not in chunk:
                        logger.warning("Ignoring malformed parameter '{}'.", chunk)
                        continue
                    key, value = chunk.split('=', maxsplit=1)
                    params[key] = _coerce_value(value)
                await self._send_control(
                    ModelControlCommand(
                        command=ControlCommandType.UPDATE_PARAMS,
                        model_name=model,
                        parameters=params,
                    )
                )
                continue
            logger.info("Commands: status | start <model> | stop <model> | set <model> key=value ... | quit")

    async def _handle_time_pos(self, time_pos: float) -> None:
        if not self._running.is_set():
            return
        async with self._render_lock:
            self._last_time_pos = time_pos
            frame_hash = await asyncio.to_thread(self._capture_frame_hash)
            if frame_hash and frame_hash == self._last_hash_rendered:
                return

            result = None
            if frame_hash:
                result = self.cache.get_by_hash(frame_hash)

            if result is None:
                result = self.cache.lookup(time_pos)

            if result is None:
                return

            expected_hash = result.sync.frame_hash
            if frame_hash and frame_hash != expected_hash:
                delta_ts = time_pos - result.sync.timestamp
                if abs(delta_ts) <= self.cache.tolerance * 2:
                    logger.warning(
                        "Frame hash mismatch: client={} server={} (Δt={:.3f}s) – using server hash",
                        frame_hash,
                        expected_hash,
                        delta_ts,
                    )
                    frame_hash = expected_hash
                else:
                    logger.warning(
                        "Frame hash mismatch outside tolerance: client={} server={} (Δt={:.3f}s)",
                        frame_hash,
                        expected_hash,
                        delta_ts,
                    )
                    return

            if frame_hash is None:
                frame_hash = expected_hash

            frame_shape = result.extra.get("frame_shape") if isinstance(result.extra, dict) else None
            target_size, pos, scale = self._compute_overlay_geometry(frame_shape)
            overlay_image = await asyncio.to_thread(
                self._generate_overlay_image,
                result,
                target_size,
                scale,
            )
            if overlay_image is None:
                return

            await self._update_overlay(overlay_image, pos)
            self._last_hash_verified = frame_hash
            self._last_hash_rendered = expected_hash

    async def _clear_overlay(self) -> None:
        if self._overlay is None or self._mpv is None:
            return
        self._overlay.remove()
        self._overlay = self._mpv.create_image_overlay()
        self._last_hash_rendered = None
        self._last_hash_verified = None

    async def _update_overlay(self, img: Optional[Image.Image], pos: Tuple[int, int]) -> None:
        if self._overlay is None or self._mpv is None:
            return
        if img is None:
            await self._clear_overlay()
            return
        self._overlay.update(img=img, pos=pos)

    def _generate_overlay_image(
        self,
        result: InferenceResult,
        target_size: Tuple[int, int],
        scale: Tuple[float, float],
    ) -> Optional[Image.Image]:
        frame_shape = result.extra.get("frame_shape") if isinstance(result.extra, dict) else None
        if not frame_shape:
            logger.debug("Inference result missing frame shape; skipping overlay")
            return None
        target_width, target_height = target_size
        overlay = Image.new("RGBA", (target_width, target_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        boxes = self._extract_overlay_boxes(result, scale, target_size)
        if not boxes:
            return None

        for label, confidence, x1, y1, x2, y2 in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0, 200), width=4)
            text = f"{label} {confidence:.2f}" if confidence is not None else label
            text_pos = (x1 + 4, max(0, y1 - 20))
            draw.text(text_pos, text, fill=(255, 255, 255, 220))
        return overlay

    def _extract_overlay_boxes(
        self,
        result: InferenceResult,
        scale: Tuple[float, float],
        target_size: Tuple[int, int],
    ) -> list[tuple[str, Optional[float], int, int, int, int]]:
        boxes: list[tuple[str, Optional[float], int, int, int, int]] = []
        extras = result.extra if isinstance(result.extra, dict) else {}
        sx, sy = scale
        max_w, max_h = target_size
        max_w = max(1, max_w) - 1
        max_h = max(1, max_h) - 1

        for key, entries in extras.items():
            if not key.endswith("_results"):
                continue
            if not isinstance(entries, (list, tuple)):
                continue
            label = key.replace("_results", "")
            for item in entries:
                if isinstance(item, dict):
                    x1 = item.get("x1")
                    y1 = item.get("y1")
                    x2 = item.get("x2")
                    y2 = item.get("y2")
                    confidence = item.get("confidence")
                elif isinstance(item, (list, tuple)):
                    if len(item) < 4:
                        continue
                    x1, y1, x2, y2 = item[:4]
                    confidence = float(item[4]) if len(item) > 4 else None
                else:
                    continue

                try:
                    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                except (TypeError, ValueError):
                    continue

                x1_s = int(round(x1_i * sx))
                y1_s = int(round(y1_i * sy))
                x2_s = int(round(x2_i * sx))
                y2_s = int(round(y2_i * sy))

                x1_s = max(0, min(max_w, x1_s))
                y1_s = max(0, min(max_h, y1_s))
                x2_s = max(0, min(max_w, x2_s))
                y2_s = max(0, min(max_h, y2_s))

                boxes.append((label, confidence, x1_s, y1_s, x2_s, y2_s))

        if not boxes and result.detections:
            for detection in result.detections:
                bbox = detection.bbox
                x1_s = int(round(bbox.x * sx))
                y1_s = int(round(bbox.y * sy))
                x2_s = int(round((bbox.x + bbox.width) * sx))
                y2_s = int(round((bbox.y + bbox.height) * sy))
                x1_s = max(0, min(max_w, x1_s))
                y1_s = max(0, min(max_h, y1_s))
                x2_s = max(0, min(max_w, x2_s))
                y2_s = max(0, min(max_h, y2_s))
                boxes.append((detection.label, detection.confidence, x1_s, y1_s, x2_s, y2_s))

        return boxes

    def _compute_overlay_geometry(
        self,
        frame_shape: Optional[Iterable[int]],
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[float, float]]:
        frame_h, frame_w = (0, 0)
        if frame_shape and len(frame_shape) == 2:
            frame_h, frame_w = int(frame_shape[0]), int(frame_shape[1])

        if frame_w <= 0 or frame_h <= 0:
            frame_w = frame_w or 1
            frame_h = frame_h or 1

        target_w, target_h = frame_w, frame_h
        offset = (0, 0)

        if self._mpv is not None:
            dims = None
            try:
                dims = self._mpv.osd_dimensions
            except Exception:  # pragma: no cover - property unavailable
                dims = None

            if isinstance(dims, dict) and all(k in dims for k in ("w", "h")):
                target_w = int(dims.get("w", frame_w)) or frame_w
                target_h = int(dims.get("h", frame_h)) or frame_h
                offset = (
                    int(dims.get("ml", 0)),
                    int(dims.get("mt", 0)),
                )
            else:
                target_w = int(getattr(self._mpv, "dwidth", frame_w)) or frame_w
                target_h = int(getattr(self._mpv, "dheight", frame_h)) or frame_h

        scale_x = target_w / frame_w if frame_w else 1.0
        scale_y = target_h / frame_h if frame_h else 1.0

        return (target_w, target_h), offset, (scale_x, scale_y)

    def _capture_frame_hash(self) -> Optional[str]:
        if self._mpv is None:
            return None
        try:
            data = self._mpv.command('screenshot-raw', 'video')
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to capture client frame: {}", exc)
            return None
        if not data:
            return None
        return hashlib.blake2b(bytes(data['data']), digest_size=16).hexdigest()

    async def _send_control(self, command: ModelControlCommand) -> None:
        if not self.ctrl_socket:
            logger.warning("Control socket not initialized")
            return
        payload = command.model_dump_json()
        async with self.ctrl_lock:
            await self.ctrl_socket.send_string(payload)
            reply = await self.ctrl_socket.recv_json()
            logger.info("Control reply: {}", reply)

    async def _request_status(self) -> None:
        await self._send_control(
            ModelControlCommand(command=ControlCommandType.REQUEST_STATUS, model_name="*")
        )

    async def _cleanup(self) -> None:
        self._running.clear()
        if self._overlay is not None:
            try:
                self._overlay.remove()
            except Exception:  # noqa: BLE001
                pass
            self._overlay = None
        if self._mpv is not None:
            self._mpv.terminate()
            self._mpv = None
        if self.sub_socket is not None:
            self.sub_socket.close()
            self.sub_socket = None
        if self.ctrl_socket is not None:
            self.ctrl_socket.close()
            self.ctrl_socket = None

    def _log_mpv(self, loglevel: str, component: str, message: str) -> None:
        logger.debug("[MPV:{}] {} - {}", loglevel, component, message.strip())


def _coerce_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MPV inference client")
    parser.add_argument("--video", required=True, help="Path or URL of the video to play")
    parser.add_argument("--sub", default="tcp://localhost:5556", help="PUB endpoint to subscribe to")
    parser.add_argument("--ctrl", default="tcp://localhost:5557", help="Control endpoint")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Timestamp tolerance in seconds")
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    client = InferenceClient(
        video_uri=args.video,
        sub_endpoint=args.sub,
        ctrl_endpoint=args.ctrl,
        tolerance=args.tolerance,
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def runner() -> None:
        await client.serve_forever()

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
