from __future__ import annotations

from dataclasses import dataclass

from services.common.events import RecordSegmentEvent

from ..domain.ports import (
    CropSelector,
    Detector,
    RecordingStore,
    ReIDModel,
    SegmentReader,
    Tracker,
    VectorIndex,
)


@dataclass(frozen=True)
class RecordIndexResult:
    frames_seen: int
    detections_seen: int
    tracks_seen: int
    embeddings_written: int


class RecordIndexingService:
    """Orchestrates one finalized segment without owning model implementations."""

    def __init__(
        self,
        reader: SegmentReader,
        detector: Detector,
        tracker: Tracker,
        crop_selector: CropSelector,
        reid: ReIDModel,
        index: VectorIndex,
        store: RecordingStore,
        sample_fps: float = 2.0,
    ) -> None:
        self._reader = reader
        self._detector = detector
        self._tracker = tracker
        self._crop_selector = crop_selector
        self._reid = reid
        self._index = index
        self._store = store
        self._sample_fps = sample_fps

    def index_segment(self, event: RecordSegmentEvent) -> RecordIndexResult:
        """Run the pipeline; adapters currently return empty results by design."""

        frames_seen = detections_seen = tracks_seen = embeddings_written = 0
        for frame in self._reader.frames(event.segment_path, self._sample_fps):
            frames_seen += 1
            detections = self._detector.detect(frame)
            detections_seen += len(detections)
            observations = self._tracker.update(frame, detections)
            tracks_seen += len(observations)
            for observation in observations:
                crop = self._crop_selector.select(observation)
                if crop is None:
                    continue
                embedding = self._reid.embed(crop)
                if embedding is None:
                    continue
                self._index.upsert(event, observation, embedding)
                embeddings_written += 1

        result = RecordIndexResult(
            frames_seen=frames_seen,
            detections_seen=detections_seen,
            tracks_seen=tracks_seen,
            embeddings_written=embeddings_written,
        )
        self._store.save_index_result(event, frames_seen, embeddings_written)
        return result
