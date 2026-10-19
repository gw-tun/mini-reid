"""Events that cross the API/worker boundary."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RecordSegmentPayload(BaseModel):
    """Accepted HTTP payload from MediaMTX or a replay tool."""

    model_config = ConfigDict(extra="allow")

    camera_key: str | None = None
    camera: str | None = None
    segment_path: str
    duration_seconds: float | None = Field(default=None, gt=0)
    duration: float | None = Field(default=None, gt=0)
    event_id: str | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_aliases(self) -> "RecordSegmentPayload":
        if not (self.camera_key or self.camera):
            raise ValueError("camera_key or camera is required")
        if self.duration_seconds is None and self.duration is None:
            raise ValueError("duration_seconds or duration is required")
        return self


class RecordSegmentEvent(BaseModel):
    """Durable event consumed by the recording-index worker."""

    model_config = ConfigDict(extra="forbid")

    event_type: Literal["record.segment.completed"] = "record.segment.completed"
    event_id: str
    camera_key: str
    segment_path: str
    duration_seconds: float = Field(gt=0)
    completed_at: datetime
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: RecordSegmentPayload) -> "RecordSegmentEvent":
        camera_key = payload.camera_key or payload.camera
        assert camera_key is not None
        duration = payload.duration_seconds or payload.duration
        assert duration is not None
        event_id = payload.event_id or sha256(
            f"{camera_key}\n{payload.segment_path}".encode("utf-8")
        ).hexdigest()
        completed_at = payload.completed_at or datetime.now(timezone.utc)
        return cls(
            event_id=event_id,
            camera_key=camera_key,
            segment_path=payload.segment_path,
            duration_seconds=duration,
            completed_at=completed_at,
            metadata={"source": "mediamtx-webhook"},
        )
