"""Index people in a run of frames: detect -> track -> best crops -> barcodes -> database.

One implementation for every caller (today scripts/index_clip.py, with a
whole file). The frames come from a `SegmentReader` -- a whole file or a
stretch of one; where they came from does not matter here.

Writes the draft tables (infra/postgres/init/002, 003) through PgPersonStore,
in the caller's transaction, and each best crop as a JPEG under `crops_root`.
Indexing the same frames again updates rows instead of doubling them.
"""

from __future__ import annotations

import collections
import datetime as dt
import hashlib
import pathlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

from PIL import Image

from ..domain.models import Frame
from ..indexing.person_crops import PersonCropRules, PersonCropSelector, judge_person_crop
from ..indexing.track_crops import TrackCropSelector
from ..infrastructure.pg_person_store import PgPersonStore
from ..tracking.bytetrack import ByteTrackTracker


@dataclass
class IndexedTrack:
    track_id: int
    """The database id."""
    local_key: str
    first_seen: dt.datetime
    last_seen: dt.datetime
    crops: int


@dataclass
class IndexResult:
    camera_id: int
    recording_id: int
    frames: int = 0
    detections: int = 0
    kept: int = 0
    tracks: list[IndexedTrack] = field(default_factory=list)
    barcodes: dict[str, tuple[int, int]] = field(default_factory=dict)
    """model key -> (new rows, rows)."""
    last_seen: dt.datetime | None = None


class PersonIndexer:
    def __init__(self, detector, models: Sequence, crops_root: pathlib.Path,
                 rules: PersonCropRules = PersonCropRules()) -> None:
        """`detector`: a person Detector with `model_key` (YoloDetector); `models`:
        ReID adapters with `card` and `vectors()` (reid/base.py)."""
        self.detector = detector
        self.models = list(models)
        self.crops_root = pathlib.Path(crops_root)
        self.rules = rules
        weights = getattr(detector, "weights", None)
        self.detector_version = (hashlib.sha256(pathlib.Path(weights).read_bytes()).hexdigest()
                                 if weights else detector.model_key)

    def index(self, conn, frames: Iterable[Frame], *, camera_key: str, media_uri: str,
              recording_started_at: dt.datetime, file_size: int | None = None,
              key_prefix: str = "") -> IndexResult:
        """`key_prefix` keeps two stretches of one recording apart: the tracker
        numbers its tracks from 1 on every run."""
        name, stem = pathlib.PurePath(media_uri).name, pathlib.PurePath(media_uri).stem
        tracker, crops = ByteTrackTracker(), TrackCropSelector(PersonCropSelector(self.rules))

        # 1. Frames -> every person box, the tracks they form, each track's best crops.
        detections: list[dict] = []
        box_key: dict[tuple, str] = {}
        observations = collections.defaultdict(list)
        ended = []
        n_frames, last_seen = 0, recording_started_at
        for frame in frames:
            n_frames += 1
            ms = int(round(frame.captured_at.timestamp() * 1000))
            found = self.detector.detect(frame)
            for i, d in enumerate(found):
                verdict = judge_person_crop(frame.image, d.bbox, d.confidence, self.rules)
                key = f"{camera_key}/{name}:{ms}:{i}"  # recording + frame time + box number
                detections.append(dict(ingest_key=key, observed_at=frame.captured_at, bbox=d.bbox,
                                       score=d.confidence, quality=verdict.quality,
                                       payload={"label": d.label, "rejected": verdict.reason}))
                box_key[(ms, d.bbox)] = key
            for o in tracker.update(frame, found):
                observations[o.track_id].append(dict(observed_at=o.observed_at, bbox=o.bbox,
                                                     score=o.confidence, ingest_key=box_key.get((ms, o.bbox))))
                crops.select(o)
            ended += [(e, crops.end_with_times(e.track_id)) for e in tracker.ended()]
            last_seen = frame.captured_at
        ended += [(e, crops.end_with_times(e.track_id)) for e in tracker.finish()]

        # 2. Best crops -> files (the evidence a web page shows).
        by_ingest = {d["ingest_key"]: d for d in detections}
        saved: dict[str, list[dict]] = {}
        for e, best in ended:
            rows = []
            for crop, at in best:
                ms = int(round(at.timestamp() * 1000))
                uri = f"{camera_key}/{stem}/{key_prefix}{e.track_id}/{ms}.jpg"
                (self.crops_root / uri).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(crop.image).save(self.crops_root / uri, quality=92)
                ingest = next((o["ingest_key"] for o in observations[e.track_id] if o["observed_at"] == at), None)
                if ingest in by_ingest:
                    by_ingest[ingest] |= dict(crop_uri=uri, crop_height=crop.image.shape[0],
                                              crop_width=crop.image.shape[1])
                rows.append(dict(uri=uri, quality=crop.quality, observed_at=at.isoformat(), ingest_key=ingest))
            saved[e.track_id] = rows

        # 3. Best crops -> barcodes, every crop of the run in one batch per model.
        images = [crop.image for _, best in ended for crop, _ in best]
        vectors = {m.card.model_key: (m.card, m.vectors(images)) for m in self.models} if images else {}

        # 4. Everything -> the database (the caller's transaction).
        store = PgPersonStore(conn)
        camera_id = store.camera_id(camera_key)
        recording_id = store.recording_id(camera_id, media_uri, recording_started_at, last_seen, file_size)
        result = IndexResult(camera_id, recording_id, n_frames, len(detections),
                             sum(1 for d in detections if d["payload"]["rejected"] is None), last_seen=last_seen)
        detection_ids = store.add_detections(camera_id, recording_id, self.detector.model_key,
                                             self.detector_version, detections)
        track_ids = {}
        for e, best in ended:
            local_key = key_prefix + e.track_id
            track_ids[e.track_id] = store.add_track(
                camera_id, recording_id, local_key, e.first_seen, e.last_seen, saved[e.track_id],
                {"observations": e.observations, "seconds": round((e.last_seen - e.first_seen).total_seconds(), 1)})
            store.add_observations(track_ids[e.track_id],
                                   [o | {"detection_id": detection_ids.get(o["ingest_key"])}
                                    for o in observations[e.track_id]])
            result.tracks.append(IndexedTrack(track_ids[e.track_id], local_key, e.first_seen, e.last_seen, len(best)))
        for key, (card, matrix) in vectors.items():
            model_id = store.model_id(card)
            rows, n = [], 0
            for e, best in ended:
                for (crop, at), row in zip(best, saved[e.track_id]):
                    ms = int(round(at.timestamp() * 1000))
                    rows.append(dict(track_id=track_ids[e.track_id], detection_id=detection_ids.get(row["ingest_key"]),
                                     camera_id=camera_id, observed_at=at, quality=crop.quality, vector=matrix[n],
                                     sample_key=f"{track_ids[e.track_id]}:{card.model_key}:{card.sha256[:12]}:{ms}"))
                    n += 1
            result.barcodes[key] = (store.add_embeddings(card, model_id, rows), len(rows))
        return result
