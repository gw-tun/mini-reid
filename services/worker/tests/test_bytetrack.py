import datetime as dt

import numpy as np

from services.worker.app.domain.models import Detection, Frame, ObjectType
from services.worker.app.tracking.bytetrack import ByteTrackTracker

T0 = dt.datetime(2026, 10, 20, 8, 0, tzinfo=dt.timezone.utc)
IMAGE = np.zeros((480, 640, 3), np.uint8)


def frame(camera, n):
    return Frame(camera, T0 + dt.timedelta(seconds=0.5 * n), IMAGE, "x.mp4")


def person(x, score=0.9):
    return Detection(ObjectType.PERSON, score, (x, 100.0, x + 60, 300.0), "person")


def test_one_walking_person_keeps_one_track_after_confirmation():
    tracker = ByteTrackTracker()
    seen = [tracker.update(frame("cam01", n), [person(100 + 5 * n)]) for n in range(6)]
    assert seen[0] == []                               # first sighting: not confirmed yet
    ids = {o.track_id for obs in seen[1:] for o in obs}
    assert ids == {"cam01-1"}                          # one track, camera-prefixed
    assert seen[3][0].bbox == (115.0, 100.0, 175.0, 300.0)   # back in frame pixels


def test_cameras_track_separately_with_unique_ids():
    tracker = ByteTrackTracker()
    for n in range(3):
        a = tracker.update(frame("cam01", n), [person(100)])
        b = tracker.update(frame("cam02", n), [person(100)])
    assert a[0].track_id != b[0].track_id
    assert a[0].track_id.startswith("cam01-") and b[0].track_id.startswith("cam02-")


def test_a_person_and_a_car_in_the_same_place_get_different_tracks():
    tracker = ByteTrackTracker()
    car = Detection(ObjectType.VEHICLE, 0.9, (100.0, 100.0, 160.0, 300.0), "car")
    for n in range(3):
        out = tracker.update(frame("cam01", n), [person(100), car])
    assert len({o.track_id for o in out}) == 2


def test_a_track_ends_after_max_age_missed_frames():
    tracker = ByteTrackTracker(max_age=2)
    for n in range(3):
        tracker.update(frame("cam01", n), [person(100)])
    for n in range(3, 6):
        tracker.update(frame("cam01", n), [])
    [ended] = tracker.ended()
    assert ended.track_id == "cam01-1" and ended.observations == 2   # frames 1 and 2
    assert tracker.ended() == []                                     # handed over once


def test_finish_ends_open_tracks():
    tracker = ByteTrackTracker()
    for n in range(3):
        tracker.update(frame("cam01", n), [person(100)])
    assert [e.track_id for e in tracker.finish()] == ["cam01-1"]
