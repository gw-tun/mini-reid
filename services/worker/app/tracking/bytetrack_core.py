# COPIED VERBATIM from semantic-va src/core/tracking.py (commit bb1e782, 2026-10-06).
#
# mini-reid tracks people with the same ByteTrack semantic-va uses, so results
# here predict results there. Don't edit this copy: change semantic-va's and copy
# it again. The `Tracker` port adapter is tracking/bytetrack.py.
#
"""ByteTrack-style association over detections.

Numpy and the standard library only. Tracking raises no dependency question --
it never touches the GPU and it is a few hundred lines of association logic, so
vendoring it is cheaper than taking a package that drags a training framework
into a process pinned to `transformers==4.57.1`.

ByteTrack rather than BoT-SORT because these are fixed cameras: BoT-SORT's
camera-motion compensation buys nothing here, and its optional re-identification
is a capability this project excludes on purpose.

What this exists for is **duration**. A 2 s window cannot answer "loitering for
two minutes" no matter which model reads it, and `Track.dwell` is the only
place that answer can come from.

The honest limit, which gate C.2 measures rather than removes: an id switch
splits one two-minute loiter into two one-minute ones, and the rule does not
fire. At the default 2 fps there may simply not be enough temporal density to
associate reliably, which is why 2.0 is a starting point to measure and not a
recommendation.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    cls: str
    score: float
    bbox: tuple[float, float, float, float]
    """Normalised xyxy."""

    track_id: int | None = None
    zones: set = field(default_factory=set)
    track: "Track | None" = field(default=None, repr=False, compare=False)
    """The track this detection was assigned to, kept as a reference so the
    question "was it ever confirmed?" can be asked at window time -- after the
    track has been confirmed by a later frame, or dropped from the tracker.
    `Track.confirmed` never goes back to False."""

    @property
    def confirmed(self) -> bool:
        """Part of a track the tracker confirmed (`min_hits` sightings).

        What has()/count()/zone() mean by "the detector saw". A detection that
        never joined a confirmed track is a one-frame flicker, and counting it
        made `has(bicycle)` match 16 clips of a camera on which the tracker
        never confirmed a single bicycle (2026-10-06): 66% of bicycle clips and
        83% of motorcycle clips in the index rested on no confirmed track."""
        return self.track is not None and self.track.confirmed

    @property
    def centroid(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


@dataclass
class Track:
    track_id: int
    cls: str
    bbox: tuple[float, float, float, float]
    first_pts: float
    last_pts: float
    first_wall: float
    last_wall: float
    hits: int = 1
    misses: int = 0
    zones: set = field(default_factory=set)
    start_centroid: tuple[float, float] = (0.0, 0.0)
    max_displacement: float = 0.0
    confirmed: bool = False
    best_crop: object = None
    """The largest, highest-scoring sighting of this object, as a uint8 array.

    NOT the first. A track's first frame is by construction the moment the
    object was smallest or most occluded, and crop height varies from 77 to 773
    pixels within one class on this footage -- so the choice of sighting is
    worth more than most of the modelling on top of it.
    probes/RESULTS-ATTRIBUTE.md.
    """

    best_px: int = 0
    """Height of `best_crop` in SOURCE pixels, which is what makes it
    comparable across tracks and the criterion for replacing it."""

    best_pts: float = 0.0

    path: list = field(default_factory=list)
    """(pts, cx, cy) samples -- NORMALISED and BOUNDED, which is what the
    schema's own comment for the column already asks for: "decimated centroids,
    bounded length".

    Normalised because `bbox` is. Pixel coordinates would change meaning the
    moment `input_size` does, and a stored path outlives the config that made
    it.

    This is the substrate for every relation -- `near`, `in`, and anything that
    needs to know WHERE something was rather than only that it was there. The
    column has existed since the schema was written and has never been
    populated: 0 of 30,397 rows. doc/PLAN-PHASE1-DETAIL.md 1.5.
    """

    @property
    def dwell(self) -> float:
        """Seconds between the first and last sighting.

        On the stream's own timeline, like everything else in the pipeline --
        `first_pts`/`last_pts`, not wall time, so a file decoded at 5x
        realtime reports the dwell the footage actually shows rather than the
        time the machine took to read it.
        """
        return max(self.last_pts - self.first_pts, 0.0)

    @property
    def is_static(self) -> bool:
        """Moved less than a tenth of the frame since first seen.

        Crude on purpose: it is a hint for a VLM prompt, not a rule anyone
        should alert on. A stationary object and a stalled tracker look
        identical from here.
        """
        return self.max_displacement < 0.1


PATH_LIMIT = 240
"""Two minutes at the default 2 detect-fps.

A cap that DECIMATES rather than truncates. Dropping the oldest samples would
lose where a track came from, which is exactly what "had been near that car for
three minutes" needs to know. Halving the resolution instead keeps the whole
span: a three-minute track ends up sampled every two seconds rather than cut
back to its last two minutes.
"""


def _extend_path(track: "Track", pts: float) -> None:
    """One sample: (pts, x, y, h) -- the GROUND CONTACT point and the apparent
    height, both normalised.

    The foot of the box, not its centre. Objects stand on a ground plane, and
    where they touch it is what their position on that plane depends on; a
    centroid also moves when a person raises an arm or a boot opens.

    The height rides along because it is the only depth information a single
    fixed camera gives away for free. For a class of roughly constant real
    height, apparent height is inversely proportional to distance -- which is
    what lets a pixel gap be turned into something like metres with no
    calibration at all. Measured on this footage, a person spans 49 to 717
    pixels, a depth range of about 25x, over which one fixed pixel threshold
    means 1.3 m at the front of the scene and 10.0 m at the back.
    """
    # From the box rather than a `centroid` property: that one is Detection's,
    # and a Track's box is updated in place on every hit.
    x0, y0, x1, y1 = track.bbox
    cx = (x0 + x1) / 2.0
    track.path.append((round(pts, 3), round(cx, 4), round(y1, 4),
                       round(y1 - y0, 4)))
    if len(track.path) > PATH_LIMIT:
        track.path[:] = track.path[::2]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two (N, 4) and (M, 4) xyxy arrays."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def greedy_match(cost: np.ndarray, threshold: float):
    """Highest-IoU-first assignment.

    Greedy rather than Hungarian: at these densities -- a handful of objects
    per frame on a fixed camera -- the assignments agree, and scipy is not a
    dependency this process should take for a 10x10 matrix.
    """
    pairs, used_rows, used_cols = [], set(), set()
    if cost.size == 0:
        return pairs, used_rows, used_cols
    order = np.argsort(cost, axis=None)[::-1]
    for flat in order:
        i, j = divmod(int(flat), cost.shape[1])
        if cost[i, j] < threshold:
            break
        if i in used_rows or j in used_cols:
            continue
        pairs.append((i, j))
        used_rows.add(i)
        used_cols.add(j)
    return pairs, used_rows, used_cols


class ByteTrack:
    """Association by IoU, in two passes over confidence.

    The idea ByteTrack is named for: match high-confidence detections first,
    then give the leftovers a second chance against the tracks that went
    unmatched. A person half-occluded for a second drops in confidence rather
    than disappearing, and the second pass is what keeps their id.
    """

    def __init__(self, max_age: int = 10, min_hits: int = 2,
                 iou_threshold: float = 0.3, high_score: float = 0.6,
                 ids=None) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_score = high_score
        self._tracks: list[Track] = []
        self._ids = ids if ids is not None else itertools.count(1)
        """Shared across every tracker in one DetectStage when the caller
        passes one.

        A per-tracker counter restarts at 1 for each stream and again whenever
        `forget_stream` drops one, so `(run_id, track_key)` -- which the schema
        calls the real key -- collided between unrelated tracks. The upsert
        then merged them: measured on MEVA, a track showed 318 seconds of
        dwell across clips that are at most eight seconds long, which the
        loiter rule would have fired on happily."""

    def tracks(self) -> list[Track]:
        return [t for t in self._tracks if t.confirmed]

    def update(self, detections: list[Detection], pts: float) -> list[Detection]:
        """Assign track ids. Returns the same detections, annotated."""
        wall = time.time()
        high = [d for d in detections if d.score >= self.high_score]
        low = [d for d in detections if d.score < self.high_score]

        unmatched = list(self._tracks)
        for pool in (high, low):
            unmatched = self._associate(pool, unmatched, pts, wall)

        for track in unmatched:
            track.misses += 1
        self._tracks = [t for t in self._tracks if t.misses <= self.max_age]
        return detections

    def _associate(self, dets: list[Detection], tracks: list[Track],
                   pts: float, wall: float) -> list[Track]:
        if not dets:
            return tracks
        det_boxes = np.array([d.bbox for d in dets], dtype=np.float32)
        trk_boxes = np.array([t.bbox for t in tracks], dtype=np.float32) \
            if tracks else np.zeros((0, 4), dtype=np.float32)

        cost = iou_matrix(det_boxes, trk_boxes)
        # A track only competes for a detection of its own class. Without
        # this a car and a person that overlap will swap ids, and the swap is
        # invisible in every aggregate afterwards.
        for i, det in enumerate(dets):
            for j, track in enumerate(tracks):
                if det.cls != track.cls:
                    cost[i, j] = 0.0

        pairs, used_dets, used_trks = greedy_match(cost, self.iou_threshold)
        for i, j in pairs:
            self._hit(tracks[j], dets[i], pts, wall)

        for i, det in enumerate(dets):
            if i in used_dets:
                continue
            if det.score < self.high_score:
                # Low-confidence detections recover tracks; they do not start
                # them. Otherwise every flicker becomes a one-frame track and
                # the counts are noise.
                continue
            self._spawn(det, pts, wall)

        return [t for j, t in enumerate(tracks) if j not in used_trks]

    def _hit(self, track: Track, det: Detection, pts: float, wall: float) -> None:
        track.bbox = det.bbox
        _extend_path(track, pts)
        track.last_pts = pts
        track.last_wall = wall
        track.hits += 1
        track.misses = 0
        track.zones |= det.zones
        cx, cy = det.centroid
        sx, sy = track.start_centroid
        track.max_displacement = max(track.max_displacement,
                                     float(np.hypot(cx - sx, cy - sy)))
        if track.hits >= self.min_hits:
            track.confirmed = True
        det.track_id = track.track_id
        det.track = track

    def _spawn(self, det: Detection, pts: float, wall: float) -> None:
        track = Track(
            track_id=next(self._ids), cls=det.cls, bbox=det.bbox,
            first_pts=pts, last_pts=pts, first_wall=wall, last_wall=wall,
            start_centroid=det.centroid, zones=set(det.zones),
            confirmed=self.min_hits <= 1,
        )
        _extend_path(track, pts)
        self._tracks.append(track)
        det.track_id = track.track_id
        det.track = track
