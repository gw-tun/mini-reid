from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any


class ObjectType(StrEnum):
    PERSON = "person"
    VEHICLE = "vehicle"


@dataclass(frozen=True)
class Frame:
    camera_key: str
    captured_at: datetime
    image: Any
    source_path: str


@dataclass(frozen=True)
class Detection:
    object_type: ObjectType
    confidence: float
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class TrackObservation:
    camera_key: str
    track_id: str
    object_type: ObjectType
    observed_at: datetime
    bbox: tuple[float, float, float, float]
    confidence: float
    image: Any


@dataclass(frozen=True)
class CropSample:
    track_id: str
    object_type: ObjectType
    image: Any
    quality: float


@dataclass(frozen=True)
class Embedding:
    vector_space: str
    values: tuple[float, ...]
    model_version: str


@dataclass(frozen=True)
class MatchCandidate:
    camera_key: str
    track_id: str
    score: float


@dataclass(frozen=True)
class CameraTransition:
    source_camera: str
    destination_camera: str
    min_seconds: float
    max_seconds: float

    def window(self, observed_at: datetime) -> tuple[datetime, datetime]:
        return (
            observed_at + timedelta(seconds=self.min_seconds),
            observed_at + timedelta(seconds=self.max_seconds),
        )


@dataclass
class FollowSession:
    session_id: str
    source_camera: str
    current_camera: str
    started_at: datetime
    active: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
