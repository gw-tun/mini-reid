from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from ..domain.models import CropSample, FollowSession, MatchCandidate, TrackObservation
from ..domain.ports import FollowEventSink, ReIDModel, TransitionGraph, VectorIndex


class LiveFollowService:
    """Schedules route-aware cross-camera matching for an active target."""

    def __init__(
        self,
        graph: TransitionGraph,
        reid: ReIDModel,
        index: VectorIndex,
        sink: FollowEventSink,
        similarity_threshold: float = 0.65,
    ) -> None:
        self._graph = graph
        self._reid = reid
        self._index = index
        self._sink = sink
        self._similarity_threshold = similarity_threshold

    def start(self, session: FollowSession) -> Sequence[tuple[str, datetime, datetime]]:
        """Return downstream ReID windows; activation is owned by the scheduler."""

        windows = []
        for transition in self._graph.outgoing(session.current_camera):
            start, end = transition.window(session.started_at)
            windows.append((transition.destination_camera, start, end))
        return windows

    def observe(
        self,
        session: FollowSession,
        observation: TrackObservation,
        crop: CropSample,
    ) -> Sequence[MatchCandidate]:
        """Embed a candidate observation and publish ranked cross-camera matches."""

        if not session.active:
            return ()
        embedding = self._reid.embed(crop)
        if embedding is None:
            return ()
        candidates = tuple(
            candidate
            for candidate in self._index.search(
                embedding,
                camera_key=observation.camera_key,
            )
            if candidate.score >= self._similarity_threshold
        )
        self._sink.publish(session, candidates)
        return candidates
