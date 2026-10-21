from __future__ import annotations

import unittest
from datetime import datetime, timezone

from services.common.events import RecordSegmentEvent
from services.worker.app.application.backbone import (
    CropQualityPolicy,
    EmbeddingContractError,
    EmbeddingService,
    InMemoryVectorIndex,
    MatchPolicy,
    ModelRegistry,
    ModelSpec,
    RepresentativeCropSelector,
)
from services.worker.app.application.record_indexer import RecordIndexingService
from services.worker.app.application.record_search import RecordSearchService
from services.worker.app.application.live_follow import LiveFollowService
from services.worker.app.domain.models import (
    CropSample,
    Detection,
    Embedding,
    Frame,
    MatchCandidate,
    CameraTransition,
    FollowSession,
    ObjectType,
    TrackObservation,
)
from services.worker.app.infrastructure.adapters import EmptyFollowEventSink, StaticTransitionGraph


NOW = datetime(2026, 10, 19, tzinfo=timezone.utc)


def event(event_id: str = "segment-1") -> RecordSegmentEvent:
    return RecordSegmentEvent(
        event_id=event_id,
        camera_key="cam01",
        segment_path="/recordings/cam01/segment.mp4",
        duration_seconds=300,
        completed_at=NOW,
    )


def observation(
    track_id: str = "track-1",
    *,
    camera_key: str = "cam01",
    object_type: ObjectType = ObjectType.PERSON,
    bbox: tuple[float, float, float, float] = (0, 0, 80, 160),
    confidence: float = 0.9,
) -> TrackObservation:
    return TrackObservation(
        camera_key=camera_key,
        track_id=track_id,
        object_type=object_type,
        observed_at=NOW,
        bbox=bbox,
        confidence=confidence,
        image=f"crop:{track_id}",
    )


class FakeReID:
    def __init__(self, values: tuple[float, ...], vector_space: str = "person-v1") -> None:
        self.values = values
        self.vector_space = vector_space

    def embed(self, crop: CropSample) -> Embedding:
        return Embedding(self.vector_space, self.values, "fake-1")


class BackboneTests(unittest.TestCase):
    def test_embedding_service_normalizes_and_tags_vector_space(self) -> None:
        registry = ModelRegistry(
            {
                ObjectType.PERSON: ModelSpec("person-model", ObjectType.PERSON, "person-v1", 2)
            }
        )
        service = EmbeddingService(registry, {ObjectType.PERSON: FakeReID((3.0, 4.0))})

        embedding = service.embed(CropSample("track-1", ObjectType.PERSON, "image", 1.0))

        self.assertEqual(embedding.vector_space, "person-v1")
        self.assertAlmostEqual(embedding.values[0], 0.6)
        self.assertAlmostEqual(embedding.values[1], 0.8)

    def test_embedding_contract_rejects_wrong_dimension(self) -> None:
        registry = ModelRegistry(
            {
                ObjectType.PERSON: ModelSpec("person-model", ObjectType.PERSON, "person-v1", 2)
            }
        )
        service = EmbeddingService(registry, {ObjectType.PERSON: FakeReID((1.0, 2.0, 3.0))})

        with self.assertRaises(EmbeddingContractError):
            service.embed(CropSample("track-1", ObjectType.PERSON, "image", 1.0))

    def test_quality_gate_keeps_best_crop_per_track(self) -> None:
        selector = RepresentativeCropSelector(CropQualityPolicy())

        self.assertIsNone(selector.select(observation(bbox=(0, 0, 20, 50))))
        first = selector.select(observation(bbox=(0, 0, 80, 160)))
        second = selector.select(observation(bbox=(0, 0, 120, 240)))
        third = selector.select(observation(bbox=(0, 0, 100, 200)))

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNone(third)
        assert first is not None and second is not None
        self.assertGreater(second.quality, first.quality)

    def test_exact_vector_index_filters_object_type_and_ranks(self) -> None:
        index = InMemoryVectorIndex()
        index.upsert(
            event("person-a"),
            observation("person-a", camera_key="cam01"),
            Embedding("person-v1", (1.0, 0.0), "fake-1"),
        )
        index.upsert(
            event("person-b"),
            observation("person-b", camera_key="cam02"),
            Embedding("person-v1", (0.0, 1.0), "fake-1"),
        )
        index.upsert(
            event("vehicle-a"),
            observation("vehicle-a", camera_key="cam03", object_type=ObjectType.VEHICLE),
            Embedding("vehicle-v1", (1.0, 0.0), "fake-1"),
        )

        results = index.search(
            Embedding("person-v1", (1.0, 0.0), "fake-1"),
            object_type="person",
        )

        self.assertEqual([candidate.track_id for candidate in results], ["person-a", "person-b"])
        self.assertAlmostEqual(results[0].score, 1.0)

    def test_record_search_returns_video_evidence(self) -> None:
        index = InMemoryVectorIndex()
        index.upsert(
            event("segment-2"),
            observation("track-2", camera_key="cam01"),
            Embedding("person-v1", (1.0, 0.0), "fake-1"),
        )
        results = RecordSearchService(FakeReID((1.0, 0.0)), index, min_similarity=0.6).search(
            "query-image", ObjectType.PERSON, top_k=5
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].segment_path, "/recordings/cam01/segment.mp4")
        self.assertEqual(results[0].track_id, "track-2")

    def test_match_policy_abstains_when_top_two_are_too_close(self) -> None:
        policy = MatchPolicy(min_similarity=0.65, min_margin=0.05)
        close = policy.accept(
            [
                MatchCandidate("cam01", "a", 0.81),
                MatchCandidate("cam02", "b", 0.79),
            ]
        )
        clear = policy.accept(
            [
                MatchCandidate("cam01", "a", 0.81),
                MatchCandidate("cam02", "b", 0.70),
            ]
        )

        self.assertIsNone(close)
        self.assertIsNotNone(clear)

    def test_live_follow_routes_to_downstream_camera_and_searches_gallery(self) -> None:
        index = InMemoryVectorIndex()
        index.upsert(event("history-01"), observation("track-01", camera_key="cam01"), Embedding("person-v1", (1.0, 0.0), "fake-1"))
        index.upsert(event("history-02"), observation("track-02", camera_key="cam02"), Embedding("person-v1", (0.8, 0.2), "fake-1"))
        service = LiveFollowService(
            StaticTransitionGraph((CameraTransition("cam01", "cam02", 10, 45),)),
            FakeReID((1.0, 0.0)), index, EmptyFollowEventSink(), similarity_threshold=0.65,
        )
        session = FollowSession("follow-1", "cam01", "cam01", NOW)

        windows = service.start(session)
        matches = service.observe(session, observation("live", camera_key="cam02"), CropSample("live", ObjectType.PERSON, "image", 1.0))

        self.assertEqual(windows[0][0], "cam02")
        self.assertEqual([item.camera_key for item in matches], ["cam01", "cam02"])


class RecordIndexingTests(unittest.TestCase):
    def test_record_indexing_orchestrates_frame_to_embedding(self) -> None:
        frame = Frame("cam01", NOW, "frame", "/recordings/cam01/segment.mp4")
        track = observation()

        class Reader:
            def frames(self, segment_path: str, sample_fps: float, started_at=None):
                return [frame]

        class Detector:
            def detect(self, frame: Frame):
                return [Detection(ObjectType.PERSON, 0.9, track.bbox)]

        class Tracker:
            def update(self, frame: Frame, detections):
                return [track]

        class Selector:
            def select(self, observation):
                return CropSample(observation.track_id, observation.object_type, observation.image, 1.0)

        class Index:
            def __init__(self):
                self.calls = []

            def upsert(self, event, observation, embedding):
                self.calls.append((event.event_id, observation.track_id, embedding.vector_space))

        class Store:
            def __init__(self):
                self.result = None

            def save_index_result(self, event, frames_seen, embeddings_written):
                self.result = (frames_seen, embeddings_written)

        index = Index()
        store = Store()
        service = RecordIndexingService(
            Reader(), Detector(), Tracker(), Selector(), FakeReID((1.0, 0.0)), index, store
        )

        result = service.index_segment(event())

        self.assertEqual(result.frames_seen, 1)
        self.assertEqual(result.detections_seen, 1)
        self.assertEqual(result.tracks_seen, 1)
        self.assertEqual(result.embeddings_written, 1)
        self.assertEqual(index.calls, [("segment-1", "track-1", "person-v1")])
        self.assertEqual(store.result, (1, 1))

    def test_record_indexing_uses_event_camera_key_not_reader_stem(self) -> None:
        # The shared worker reader does not know the camera, so it falls back to
        # the segment filename stem. The indexer must attribute observations to
        # the event's camera instead.
        frame = Frame("segment-stem", NOW, "frame", "/recordings/cam01/segment.mp4")

        class Reader:
            def frames(self, segment_path: str, sample_fps: float, started_at=None):
                return [frame]

        class Detector:
            def detect(self, frame: Frame):
                return [Detection(ObjectType.PERSON, 0.9, (0, 0, 80, 160))]

        class Tracker:
            def update(self, frame: Frame, detections):
                return [
                    TrackObservation(
                        frame.camera_key, "track-1", ObjectType.PERSON, NOW,
                        (0, 0, 80, 160), 0.9, "image",
                    )
                ]

        class Selector:
            def select(self, observation):
                return CropSample(observation.track_id, observation.object_type, observation.image, 1.0)

        class Index:
            def __init__(self):
                self.camera_keys = []

            def upsert(self, event, observation, embedding):
                self.camera_keys.append(observation.camera_key)

        class Store:
            def save_index_result(self, event, frames_seen, embeddings_written):
                pass

        index = Index()
        service = RecordIndexingService(
            Reader(), Detector(), Tracker(), Selector(), FakeReID((1.0, 0.0)), index, Store()
        )

        service.index_segment(event())

        self.assertEqual(index.camera_keys, ["cam01"])


if __name__ == "__main__":
    unittest.main()
