import datetime as dt

import numpy as np

from services.worker.app.domain.models import CropSample, ObjectType, TrackObservation
from services.worker.app.indexing.track_crops import TrackCropSelector

T0 = dt.datetime(2026, 10, 20, 8, 0, tzinfo=dt.timezone.utc)


class GateByScore:
    """Fake person gate: quality = the observation's confidence."""

    def select(self, o):
        return CropSample(o.track_id, ObjectType.PERSON, np.zeros((2, 1, 3)), o.confidence)


def obs(n, conf=0.8, track="cam01-1"):
    return TrackObservation("cam01", track, ObjectType.PERSON, T0 + dt.timedelta(seconds=0.5 * n),
                            (0, 0, 1, 2), conf, None)


def test_first_good_crop_is_embedded_then_only_every_refresh_interval():
    s = TrackCropSelector(GateByScore(), keep=10, refresh_every=4)
    emitted = [s.select(obs(n)) is not None for n in range(9)]
    assert emitted == [True, False, False, False, True, False, False, False, True]


def test_a_crop_that_is_not_among_the_best_is_never_embedded():
    s = TrackCropSelector(GateByScore(), keep=2, refresh_every=1)
    assert s.select(obs(0, 0.9)) is not None
    assert s.select(obs(1, 0.8)) is not None
    assert s.select(obs(2, 0.5)) is None        # worse than both kept crops
    assert s.select(obs(3, 0.95)) is not None   # better: replaces 0.8


def test_end_returns_best_crops_best_first_and_forgets_the_track():
    s = TrackCropSelector(GateByScore(), keep=3)
    for n, q in enumerate([0.6, 0.9, 0.7, 0.8]):
        s.select(obs(n, q))
    assert [round(c.quality, 2) for c in s.end("cam01-1")] == [0.9, 0.8, 0.7]
    assert s.best("cam01-1") == []


def test_near_duplicate_frames_are_skipped():
    s = TrackCropSelector(GateByScore(), keep=4, min_gap=dt.timedelta(seconds=1))
    s.select(obs(0))
    s.select(obs(1))      # 0.5 s later: too close
    s.select(obs(2))      # 1.0 s later: kept
    assert len(s.best("cam01-1")) == 2
