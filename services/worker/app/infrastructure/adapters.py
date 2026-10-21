from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import datetime

from services.common.events import RecordSegmentEvent

from ..domain.models import (
    CameraTransition,
    CropSample,
    Detection,
    Embedding,
    Frame,
    FollowSession,
    MatchCandidate,
    TrackObservation,
)


class EmptySegmentReader:
    def frames(
        self, segment_path: str, sample_fps: float, started_at: datetime | None = None
    ) -> Iterable[Frame]:
        return ()


class EmptyDetector:
    def detect(self, frame: Frame) -> Sequence[Detection]:
        return ()


class EmptyTracker:
    def update(self, frame: Frame, detections: Sequence[Detection]) -> Sequence[TrackObservation]:
        return ()


class EmptyCropSelector:
    def select(self, observation: TrackObservation) -> CropSample | None:
        return None


class EmptyReIDModel:
    def embed(self, crop: CropSample) -> Embedding | None:
        return None


class EmptyVectorIndex:
    def upsert(self, event: RecordSegmentEvent, observation: TrackObservation, embedding: Embedding) -> None:
        pass

    def search(
        self,
        embedding: Embedding,
        *,
        camera_key: str | None = None,
        object_type: str | None = None,
        top_k: int = 20,
        exclude_track_db_id: int | None = None,
    ) -> Sequence[MatchCandidate]:
        return ()


class EmptyRecordingStore:
    def save_index_result(self, event: RecordSegmentEvent, frames_seen: int, embeddings_written: int) -> None:
        pass


class EmptyFollowEventSink:
    def publish(self, session: FollowSession, candidates: Sequence[MatchCandidate]) -> None:
        pass


class StaticTransitionGraph:
    def __init__(self, transitions: Sequence[CameraTransition] = ()) -> None:
        self._transitions = tuple(transitions)

    def outgoing(self, source_camera: str) -> Sequence[CameraTransition]:
        return tuple(t for t in self._transitions if t.source_camera == source_camera)
