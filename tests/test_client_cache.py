from __future__ import annotations

import pytest

pytest.importorskip("zmq")

from src.client.inference_client import InferenceCache
from src.shared.models import (
    BoundingBox,
    Detection,
    FrameSyncMetadata,
    InferenceMetadata,
    InferenceResult,
)


def make_result(timestamp: float, frame_hash: str) -> InferenceResult:
    frame_index = int(round(timestamp * 1000))
    return InferenceResult(
        sync=FrameSyncMetadata(
            video_uri="sample.mp4",
            timestamp=timestamp,
            frame_hash=frame_hash,
            frame_index=frame_index,
            capture_time=timestamp,
        ),
        model=InferenceMetadata(model_name="brightness_hotspot"),
        detections=[
            Detection(
                label="hotspot",
                confidence=0.5,
                bbox=BoundingBox(x=0, y=0, width=10, height=10),
            )
        ],
        latency_ms=None,
        server_time=timestamp,
        extra={"frame_shape": [480, 640], "frame_index": frame_index},
    )


def test_cache_lookup_within_tolerance() -> None:
    cache = InferenceCache(max_items=10, tolerance=0.1)
    cache.add(make_result(1.0, "hash_a0000"))
    cache.add(make_result(2.0, "hash_b0000"))
    found = cache.lookup(2.05)
    assert found is not None
    assert found.sync.frame_hash == "hash_b0000"


def test_cache_eviction() -> None:
    cache = InferenceCache(max_items=2, tolerance=0.1)
    cache.add(make_result(1.0, "hash_a0000"))
    cache.add(make_result(2.0, "hash_b0000"))
    cache.add(make_result(3.0, "hash_c0000"))
    assert cache.lookup(1.0) is None
    assert cache.lookup(3.0).sync.frame_hash == "hash_c0000"


def test_cache_handles_hash_collisions_with_temporal_metadata() -> None:
    cache = InferenceCache(max_items=10, tolerance=0.15)
    early = make_result(1.0, "hash_dup000")
    late = make_result(1.12, "hash_dup000")
    cache.add(early)
    cache.add(late)

    match_early = cache.get_by_hash("hash_dup000", timestamp=1.02)
    assert match_early is not None
    assert match_early.sync.timestamp == pytest.approx(early.sync.timestamp)

    match_late = cache.get_by_hash("hash_dup000", timestamp=1.12)
    assert match_late is not None
    assert match_late.sync.timestamp == pytest.approx(late.sync.timestamp)


def test_lookup_prefers_non_future_results() -> None:
    cache = InferenceCache(max_items=10, tolerance=0.1)
    first = make_result(5.0, "hash_time00")
    second = make_result(5.08, "hash_time00")
    cache.add(first)
    cache.add(second)

    rewind_match = cache.lookup(5.01)
    assert rewind_match is not None
    assert rewind_match.sync.timestamp == pytest.approx(first.sync.timestamp)

    forward_match = cache.lookup(5.08)
    assert forward_match is not None
    assert forward_match.sync.timestamp == pytest.approx(second.sync.timestamp)
