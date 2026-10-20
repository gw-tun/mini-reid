"""YOLO on a real frame. Skipped when Ultralytics or the weights aren't installed."""

import datetime as dt
import importlib.util
import pathlib

import numpy as np
import pytest

from services.worker.app.domain.models import Frame, ObjectType

ROOT = pathlib.Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "models" / "yolo" / "yolo11m.pt"
CLIP = ROOT / "data" / "recordings" / "test-stairs" / "2018-03-13_15-50-00.mp4"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or not WEIGHTS.is_file() or not CLIP.is_file(),
    reason="needs ultralytics, models/yolo/yolo11m.pt and the stairwell test clip")


@pytest.fixture(scope="module")
def detector():
    from services.worker.app.detection.yolo import YoloDetector

    return YoloDetector(WEIGHTS, device="cpu")


def first_frame():
    from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader

    return next(iter(PyAVSegmentReader().frames(str(CLIP), 2)))


def test_finds_people_with_boxes_inside_the_frame(detector):
    frame = first_frame()
    found = detector.detect(frame)
    people = [d for d in found if d.object_type is ObjectType.PERSON]
    assert people, "the stairwell clip has a person in its first frame"
    h, w = frame.image.shape[:2]
    for d in found:
        x1, y1, x2, y2 = d.bbox
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and 0 < d.confidence <= 1
        assert d.label in {"person", "car", "motorcycle", "bus", "truck"}


def test_rgb_frames_are_passed_as_bgr(detector):
    """Same pixels, channel order flipped by hand: if the adapter forgot to
    convert, the two runs would differ."""
    frame = first_frame()
    a = detector.detect(frame)
    flipped = Frame(frame.camera_key, frame.captured_at, np.ascontiguousarray(frame.image[..., ::-1]),
                    frame.source_path)
    b = detector.detect(flipped)
    assert len(a) >= 1 and [d.label for d in a] != [] and (len(a) != len(b) or
                                                          any(abs(x.confidence - y.confidence) > 1e-3
                                                              for x, y in zip(a, b)))


def test_person_only_detector_ignores_vehicles():
    from services.worker.app.detection.yolo import YoloDetector

    person_only = YoloDetector(WEIGHTS, device="cpu", object_types={ObjectType.PERSON})
    assert person_only.classes == [0]
    assert all(d.object_type is ObjectType.PERSON for d in person_only.detect(first_frame()))
