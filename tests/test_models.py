from __future__ import annotations

import json
from time import time

from src.shared.models import (
    BoundingBox,
    Detection,
    FrameSyncMetadata,
    InferenceMetadata,
    InferenceResult,
)


def test_inference_result_roundtrip() -> None:
    sync = FrameSyncMetadata(
        video_uri="sample.mp4",
        timestamp=12.5,
        frame_hash="abc123def456",
        frame_index=42,
    )
    metadata = InferenceMetadata(model_name="brightness_hotspot", version="0.1")
    result = InferenceResult(
        sync=sync,
        model=metadata,
        detections=[
            Detection(
                label="hotspot",
                confidence=0.87,
                bbox=BoundingBox(x=10, y=20, width=30, height=40),
                extra={"mean_brightness": 210.5},
            )
        ],
        latency_ms=12.3,
        server_time=time(),
        extra={"frame_index": 42, "frame_shape": [720, 1280]},
    )

    payload = result.model_dump_json()
    parsed = InferenceResult.model_validate_json(payload)
    assert parsed.sync == result.sync
    assert parsed.model == result.model
    assert parsed.detections[0].bbox.width == result.detections[0].bbox.width
    assert json.loads(payload)["extra"]["frame_shape"] == [720, 1280]
