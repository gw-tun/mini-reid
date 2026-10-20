"""ByteTrack tracker adapter (implements the `Tracker` port).

A detector sees one frame at a time; the tracker connects the same person
across frames inside ONE camera, giving each a track id (spec §6.3). It never
connects cameras -- that is ReID's job later.

- One ByteTrack per camera, one id counter for all of them, so a track id is
  unique: "cam01-17".
- Boxes are handed to ByteTrack normalised to 0..1 (as semantic-va does) and
  come back in frame pixels, the `Detection.bbox` convention.
- A person and a car never share a track: ByteTrack matches within one class.
- Only CONFIRMED tracks (seen `min_hits` times) produce observations, so a
  one-frame flicker never reaches the crop selector or ReID.
- A track the tracker drops after `max_age` missed frames has ended; `ended()`
  hands those over (e.g. to write the track's best crops).
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from ..domain.models import Detection, Frame, ObjectType, TrackObservation
from .bytetrack_core import ByteTrack
from .bytetrack_core import Detection as CoreDetection


@dataclass(frozen=True)
class EndedTrack:
    camera_key: str
    track_id: str
    object_type: ObjectType
    first_seen: dt.datetime
    last_seen: dt.datetime
    observations: int


@dataclass
class _Live:
    object_type: ObjectType
    first_seen: dt.datetime
    last_seen: dt.datetime
    observations: int = 0


class ByteTrackTracker:
    def __init__(self, *, max_age: int = 10, min_hits: int = 2,
                 iou_threshold: float = 0.3, high_score: float = 0.6) -> None:
        """Defaults are semantic-va's. At 2 sampled FPS, `max_age=10` keeps a
        track alive through 5 s of missed detections (a person behind a pillar)."""
        self._settings = dict(max_age=max_age, min_hits=min_hits,
                              iou_threshold=iou_threshold, high_score=high_score)
        self._ids = itertools.count(1)
        self._by_camera: dict[str, ByteTrack] = {}
        self._live: dict[str, dict[str, _Live]] = {}
        self._ended: list[EndedTrack] = []

    def update(self, frame: Frame, detections: Sequence[Detection]) -> Sequence[TrackObservation]:
        camera = frame.camera_key
        tracker = self._by_camera.get(camera)
        if tracker is None:
            tracker = self._by_camera[camera] = ByteTrack(ids=self._ids, **self._settings)
            self._live[camera] = {}
        h, w = frame.image.shape[:2]
        core = [CoreDetection(d.object_type.value, d.confidence,
                              (d.bbox[0] / w, d.bbox[1] / h, d.bbox[2] / w, d.bbox[3] / h))
                for d in detections]
        tracker.update(core, frame.captured_at.timestamp())

        live = self._live[camera]
        observations = []
        for source, d in zip(detections, core):
            if d.track_id is None or not d.confirmed:
                continue
            track_id = f"{camera}-{d.track_id}"
            state = live.get(track_id)
            if state is None:
                state = live[track_id] = _Live(source.object_type, frame.captured_at, frame.captured_at)
            state.last_seen = frame.captured_at
            state.observations += 1
            observations.append(TrackObservation(camera, track_id, source.object_type,
                                                 frame.captured_at, source.bbox,
                                                 source.confidence, frame.image))

        # ByteTrack forgets a track after `max_age` misses: that is its end.
        # (`_tracks` is the copied tracker's own list; it has no public one.)
        alive = {f"{camera}-{t.track_id}" for t in tracker._tracks}
        for track_id in [k for k in live if k not in alive]:
            s = live.pop(track_id)
            self._ended.append(EndedTrack(camera, track_id, s.object_type,
                                          s.first_seen, s.last_seen, s.observations))
        return observations

    def ended(self) -> list[EndedTrack]:
        """Tracks that ended since the last call (each is returned once)."""
        out, self._ended = self._ended, []
        return out

    def finish(self) -> list[EndedTrack]:
        """End everything still open: the segment is over."""
        for camera, live in self._live.items():
            for track_id, s in live.items():
                self._ended.append(EndedTrack(camera, track_id, s.object_type,
                                              s.first_seen, s.last_seen, s.observations))
            live.clear()
        return self.ended()
