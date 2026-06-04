from __future__ import annotations

import hashlib
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image


def _ensure_rgba(frame: np.ndarray) -> np.ndarray:
    if frame.ndim != 3:
        raise ValueError("Frame array must have shape (H, W, C)")
    channels = frame.shape[2]
    if channels == 4:
        return frame
    if channels == 3:
        alpha = np.full(frame.shape[:2] + (1,), 255, dtype=frame.dtype)
        return np.concatenate((frame, alpha), axis=2)
    raise ValueError(f"Unsupported channel count: {channels}")


def compute_luma_plane(frame_rgba: np.ndarray) -> np.ndarray:
    frame = _ensure_rgba(frame_rgba).astype(np.float32)
    r = frame[:, :, 0]
    g = frame[:, :, 1]
    b = frame[:, :, 2]
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y.astype(np.uint8)


def compute_luma_hash(frame_rgba: np.ndarray, digest_size: int = 8) -> str:
    y_plane = compute_luma_plane(frame_rgba)
    return hashlib.blake2s(y_plane.tobytes(), digest_size=digest_size).hexdigest()


def compute_dhash(frame_rgba: np.ndarray, hash_size: int = 8) -> str:
    frame = _ensure_rgba(frame_rgba)
    image = Image.fromarray(frame[:, :, :3], mode="RGB")
    image = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    width = hash_size * hash_size
    return f"{value:0{width // 4}x}"


def hamming_distance_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        raise ValueError("Hex strings must have equal length")
    diff = int(a, 16) ^ int(b, 16)
    return diff.bit_count()


@dataclass(frozen=True)
class FrameFingerprint:
    luma_hash: Optional[str] = None
    dhash: Optional[str] = None

    def matches(self, other: Optional[FrameFingerprint], *, hamming_threshold: int = 6) -> bool:
        if other is None:
            return False
        if self.luma_hash and other.luma_hash:
            return self.luma_hash == other.luma_hash
        if self.dhash and other.dhash:
            return hamming_distance_hex(self.dhash, other.dhash) <= hamming_threshold
        return False

    def to_dict(self) -> Dict[str, Optional[str]]:
        return {
            "luma_hash": self.luma_hash,
            "dhash": self.dhash,
        }

    @staticmethod
    def from_dict(data: Optional[Dict[str, str]]) -> Optional[FrameFingerprint]:  # noqa: D401
        if not data:
            return None
        return FrameFingerprint(
            luma_hash=data.get("luma_hash"),
            dhash=data.get("dhash"),
        )


@dataclass(frozen=True)
class FrameSignature:
    frame_hash: str
    fingerprint: Optional[FrameFingerprint] = None

    def matches(self, other: Optional[FrameSignature], *, hamming_threshold: int = 6) -> bool:
        if other is None:
            return False
        if self.frame_hash == other.frame_hash:
            return True
        if self.fingerprint and other.fingerprint:
            return self.fingerprint.matches(other.fingerprint, hamming_threshold=hamming_threshold)
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_hash": self.frame_hash,
            "fingerprint": self.fingerprint.to_dict() if self.fingerprint else None,
        }

    @staticmethod
    def from_numpy(frame_rgba: np.ndarray, *, frame_hash: Optional[str] = None) -> FrameSignature:
        luma_hash = compute_luma_hash(frame_rgba)
        dhash = compute_dhash(frame_rgba)
        fingerprint = FrameFingerprint(luma_hash=luma_hash, dhash=dhash)
        if frame_hash is None:
            frame_hash = hashlib.blake2b(frame_rgba.tobytes(), digest_size=16).hexdigest()
        return FrameSignature(frame_hash=frame_hash, fingerprint=fingerprint)

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> Optional[FrameSignature]:
        if not data:
            return None
        fingerprint = FrameFingerprint.from_dict(data.get("fingerprint"))
        frame_hash = data.get("frame_hash")
        if frame_hash is None:
            return None
        return FrameSignature(frame_hash=frame_hash, fingerprint=fingerprint)


@dataclass
class FrameSyncPacket:
    stream_id: str
    pts: float
    monotonic_ts: float
    frame_index: Optional[int]
    signature: Optional[FrameSignature] = None
    payload: Any = None


@dataclass
class SyncedPair:
    reference: FrameSyncPacket
    other: FrameSyncPacket
    delta: float

    def ordered(self, reference_stream: str) -> Tuple[FrameSyncPacket, FrameSyncPacket]:
        if self.reference.stream_id == reference_stream:
            return self.reference, self.other
        if self.other.stream_id == reference_stream:
            return self.other, self.reference
        raise ValueError("Reference stream missing in pair")


class DriftEstimator:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self._estimate = 0.0
        self._initialized = False

    def update(self, measurement: float) -> float:
        if not self._initialized:
            self._estimate = measurement
            self._initialized = True
        else:
            self._estimate = (1.0 - self.alpha) * self._estimate + self.alpha * measurement
        return self._estimate

    @property
    def estimate(self) -> float:
        return self._estimate


class FrameSynchronizer:
    def __init__(
        self,
        *,
        reference_stream: str,
        tolerance: float = 0.03,
        buffer_duration: float = 0.6,
        drift_alpha: float = 0.05,
    ) -> None:
        self.reference_stream = reference_stream
        self.tolerance = tolerance
        self.buffer_duration = buffer_duration
        self.buffers: Dict[str, Deque[FrameSyncPacket]] = {}
        self.latest_pts: Dict[str, float] = {}
        self.drift: Dict[str, DriftEstimator] = {}
        self.drift_alpha = drift_alpha

    def add_packet(self, packet: FrameSyncPacket) -> List[SyncedPair]:
        buffer = self.buffers.setdefault(packet.stream_id, deque())
        self._insert_sorted(buffer, packet)
        self.latest_pts[packet.stream_id] = max(packet.pts, self.latest_pts.get(packet.stream_id, packet.pts))
        self._prune(packet.stream_id)

        if packet.stream_id == self.reference_stream:
            return self._match_reference_stream()
        return self._match_non_reference(packet.stream_id)

    def _insert_sorted(self, buffer: Deque[FrameSyncPacket], packet: FrameSyncPacket) -> None:
        if not buffer or buffer[-1].pts <= packet.pts:
            buffer.append(packet)
            return
        temp = deque()
        inserted = False
        while buffer:
            head = buffer.popleft()
            if not inserted and packet.pts < head.pts:
                temp.append(packet)
                inserted = True
            temp.append(head)
        if not inserted:
            temp.append(packet)
        buffer.extend(temp)

    def _prune(self, stream_id: str) -> None:
        buffer = self.buffers[stream_id]
        if not buffer:
            return
        current_limit = self.latest_pts[stream_id] - self.buffer_duration
        while buffer and buffer[0].pts < current_limit:
            buffer.popleft()

    def _match_reference_stream(self) -> List[SyncedPair]:
        pairs: List[SyncedPair] = []
        ref_buffer = self.buffers.get(self.reference_stream)
        if not ref_buffer:
            return pairs
        if len(self.buffers) < 2:
            return pairs
        ref_packet = ref_buffer[0]
        for stream_id, buffer in self.buffers.items():
            if stream_id == self.reference_stream or not buffer:
                continue
            pair = self._attempt_pair(ref_packet, stream_id)
            if pair:
                pairs.append(pair)
        if pairs:
            ref_buffer.popleft()
        return pairs

    def _match_non_reference(self, stream_id: str) -> List[SyncedPair]:
        buffer = self.buffers[stream_id]
        if not buffer:
            return []
        ref_buffer = self.buffers.get(self.reference_stream)
        if not ref_buffer:
            return []
        ref_packet = ref_buffer[0]
        pair = self._attempt_pair(ref_packet, stream_id)
        if pair:
            ref_buffer.popleft()
            return [pair]
        return []

    def _attempt_pair(self, ref_packet: FrameSyncPacket, other_stream: str) -> Optional[SyncedPair]:
        other_buffer = self.buffers.get(other_stream)
        if not other_buffer:
            return None
        estimator = self.drift.setdefault(other_stream, DriftEstimator(alpha=self.drift_alpha))
        best_idx = -1
        best_packet: Optional[FrameSyncPacket] = None
        best_delta = math.inf
        for idx, packet in enumerate(other_buffer):
            adjusted_pts = packet.pts + estimator.estimate
            delta = adjusted_pts - ref_packet.pts
            abs_delta = abs(delta)
            if abs_delta < best_delta:
                best_delta = abs_delta
                best_packet = packet
                best_idx = idx
        if best_packet is None:
            return None
        adjusted_delta = (best_packet.pts + estimator.estimate) - ref_packet.pts
        if abs(adjusted_delta) > self.tolerance:
            # Drop stale packets that are too far in the past
            if adjusted_delta < -self.tolerance:
                other_buffer.popleft()
            return None
        estimator.update(ref_packet.pts - best_packet.pts)
        del other_buffer[best_idx]
        return SyncedPair(reference=ref_packet, other=best_packet, delta=adjusted_delta)

    def flush(self) -> Iterator[FrameSyncPacket]:
        for buffer in self.buffers.values():
            while buffer:
                yield buffer.popleft()


def capture_timestamp() -> float:
    return time.monotonic()
