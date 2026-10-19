from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime
from typing import Protocol

from services.common.events import RecordSegmentEvent

from .models import (
    CameraTransition,
    CropSample,
    Detection,
    Embedding,
    Frame,
    FollowSession,
    MatchCandidate,
    TrackObservation,
)


class SegmentReader(Protocol):
    def frames(self, segment_path: str, sample_fps: float) -> Iterable[Frame]: ...


class Detector(Protocol):
    def detect(self, frame: Frame) -> Sequence[Detection]: ...


class Tracker(Protocol):
    def update(self, frame: Frame, detections: Sequence[Detection]) -> Sequence[TrackObservation]: ...


class CropSelector(Protocol):
    def select(self, observation: TrackObservation) -> CropSample | None: ...


class ReIDModel(Protocol):
    def embed(self, crop: CropSample) -> Embedding | None: ...


class VectorIndex(Protocol):
    def upsert(self, event: RecordSegmentEvent, observation: TrackObservation, embedding: Embedding) -> None: ...

    def search(
        self,
        embedding: Embedding,
        *,
        camera_key: str | None = None,
        top_k: int = 20,
    ) -> Sequence[MatchCandidate]: ...


class RecordingStore(Protocol):
    def save_index_result(self, event: RecordSegmentEvent, frames_seen: int, embeddings_written: int) -> None: ...


class FollowEventSink(Protocol):
    def publish(self, session: FollowSession, candidates: Sequence[MatchCandidate]) -> None: ...


class TransitionGraph(Protocol):
    def outgoing(self, source_camera: str) -> Sequence[CameraTransition]: ...


class Clock(Protocol):
    def now(self) -> datetime: ...
