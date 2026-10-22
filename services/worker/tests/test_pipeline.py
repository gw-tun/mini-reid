"""RecordIndexingService end to end with the real reader, tracker and crop rules.

Only the detector and the ReID model are fakes, so this runs in about a second
without weights, and proves the person adapters fit the use case as written.
"""

import datetime as dt

from services.common.events import RecordSegmentEvent
from services.worker.app.application.record_indexer import RecordIndexingService
from services.worker.app.domain.models import ObjectType
from services.worker.app.indexing.person_crops import PersonCropSelector
from services.worker.app.indexing.routing import ClassRoutedCropSelector, ClassRoutedReID
from services.worker.app.indexing.track_crops import TrackCropSelector
from services.worker.app.infrastructure.adapters import EmptyRecordingStore, EmptyVectorIndex
from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader
from services.worker.app.tracking.bytetrack import ByteTrackTracker

from fakes import RED, ColourDetector, CountingReID, write_video


def test_one_person_walking_is_tracked_and_embedded_once_not_every_frame(tmp_path):
    clip = tmp_path / "cam01" / "2026-10-20_08-00-00.mp4"
    clip.parent.mkdir()
    write_video(clip, {RED: (0.0, 3.0, 40)}, seconds=3.0)
    person_reid = CountingReID()
    service = RecordIndexingService(
        reader=PyAVSegmentReader(),
        detector=ColourDetector(),
        tracker=ByteTrackTracker(),
        crop_selector=ClassRoutedCropSelector({ObjectType.PERSON: TrackCropSelector(PersonCropSelector())}),
        reid=ClassRoutedReID({ObjectType.PERSON: person_reid}),
        index=EmptyVectorIndex(),
        store=EmptyRecordingStore(),
        sample_fps=2,
    )
    event = RecordSegmentEvent(event_id="e1", camera_key="cam01", segment_path=str(clip),
                               duration_seconds=3, completed_at=dt.datetime.now(dt.timezone.utc))
    result = service.index_segment(event)
    assert result.frames_seen == 6 and result.detections_seen == 6
    assert result.tracks_seen == 5                  # 6 sightings, the first unconfirmed
    assert result.embeddings_written == 1           # same-quality crops: refreshed, not re-embedded
    assert {c.track_id for c in person_reid.calls} == {"cam01-1"}
