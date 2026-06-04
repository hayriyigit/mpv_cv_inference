from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Axis-aligned bounding box expressed in pixel coordinates."""

    x: int = Field(..., ge=0, description="Left coordinate in pixels")
    y: int = Field(..., ge=0, description="Top coordinate in pixels")
    width: int = Field(..., gt=0, description="Bounding box width in pixels")
    height: int = Field(..., gt=0, description="Bounding box height in pixels")

    def as_xywh(self) -> tuple[int, int, int, int]:
        """Convenience tuple representation."""
        return self.x, self.y, self.width, self.height


class Detection(BaseModel):
    """A single model detection result."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BoundingBox
    extra: Dict[str, Any] = Field(default_factory=dict)


class FrameSyncMetadata(BaseModel):
    """Frame identity shared between server and client for loose synchronization."""

    video_uri: str
    timestamp: float = Field(..., ge=0.0, description="Presentation timestamp in seconds")
    frame_hash: str = Field(..., min_length=10, description="Digest computed from decoded pixels")
    frame_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="Optional monotonically increasing frame counter maintained by the server",
    )
    capture_time: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Server monotonic timestamp recorded when the frame pixels were captured.",
    )


class InferenceMetadata(BaseModel):
    """Metadata about the model that produced a detection batch."""

    model_name: str
    version: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)


class InferenceResult(BaseModel):
    """Inference payload that the server publishes for consumption by clients."""

    sync: FrameSyncMetadata
    model: InferenceMetadata
    detections: List[Detection] = Field(default_factory=list)
    latency_ms: Optional[float] = Field(
        default=None,
        description="Optional end-to-end latency measurement for diagnostics",
    )
    server_time: float = Field(
        ...,
        description="Server monotonic time when the inference finished (seconds).",
    )
    extra: Dict[str, Any] = Field(default_factory=dict)


class ControlCommandType(str, Enum):
    """Supported control verbs sent from client to server."""

    START_MODEL = "start_model"
    STOP_MODEL = "stop_model"
    UPDATE_PARAMS = "update_params"
    REQUEST_STATUS = "request_status"


class ModelControlCommand(BaseModel):
    """A command issued by the client to manage inference models on the server."""

    command: ControlCommandType
    model_name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class ModelStatus(BaseModel):
    """Status report for a single model instance."""

    model_name: str
    enabled: bool
    parameters: Dict[str, Any] = Field(default_factory=dict)
    last_timestamp: Optional[float] = None
    last_frame_hash: Optional[str] = None


class ServerStatus(BaseModel):
    """Aggregated status snapshot returned by the server."""

    video_uri: str
    models: List[ModelStatus] = Field(default_factory=list)
    uptime_s: float
    wallclock_time: float


class Topic(str, Enum):
    """Topic names for ZeroMQ PUB/SUB routing."""

    INFERENCE = "inference"
    STATUS = "status"