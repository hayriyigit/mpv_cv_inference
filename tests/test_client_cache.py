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
    return InferenceResult(
        sync=FrameSyncMetadata(
            video_uri="sample.mp4",
            timestamp=timestamp,
            frame_hash=frame_hash,
            frame_index=int(timestamp * 10),
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
        extra={"frame_shape": [480, 640]},
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
