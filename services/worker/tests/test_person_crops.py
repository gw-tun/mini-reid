import datetime as dt

import numpy as np

from services.worker.app.domain.models import ObjectType, TrackObservation
from services.worker.app.indexing.person_crops import (
    PersonCropRules, PersonCropSelector, judge_person_crop, sharpness)

FRAME = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)


def test_a_clear_upright_person_passes_with_quality_from_size_and_confidence():
    v = judge_person_crop(FRAME, (600, 200, 680, 456), 0.9)   # 80 x 256
    assert v.reason is None and v.quality == 0.9


def test_rejections_say_why():
    assert judge_person_crop(FRAME, (600, 200, 640, 260), 0.9).reason == "too small"
    assert judge_person_crop(FRAME, (600, 200, 680, 456), 0.3).reason == "low confidence"
    assert judge_person_crop(FRAME, (300, 200, 900, 456), 0.9).reason == "odd shape"  # h/w 0.43: a group


def test_a_seated_person_is_kept_with_lower_quality():
    standing = judge_person_crop(FRAME, (600, 200, 680, 456), 0.9)   # h/w 3.2
    seated = judge_person_crop(FRAME, (500, 200, 800, 456), 0.9)     # h/w 0.85
    assert seated.reason is None and seated.quality < standing.quality
    assert seated.quality == round(0.9 * 0.6, 4)


def test_cut_off_person_counts_half():
    inside = judge_person_crop(FRAME, (600, 200, 680, 456), 0.9).quality
    at_edge = judge_person_crop(FRAME, (0, 200, 80, 456), 0.9).quality
    assert at_edge == inside / 2


def test_blur_check_when_enabled():
    flat = np.full((720, 1280, 3), 128, np.uint8)
    assert sharpness(flat[0:256, 0:80]) == 0.0
    rules = PersonCropRules(min_sharpness=10.0)
    assert judge_person_crop(flat, (600, 200, 680, 456), 0.9, rules).reason == "blurry"
    assert judge_person_crop(FRAME, (600, 200, 680, 456), 0.9, rules).reason is None


def observation(object_type, bbox, conf=0.9):
    return TrackObservation("cam01", "t1", object_type, dt.datetime(2026, 1, 1), bbox, conf, FRAME)


def test_selector_returns_person_crops_only_and_counts_rejections():
    s = PersonCropSelector()
    crop = s.select(observation(ObjectType.PERSON, (600, 200, 680, 456)))
    assert crop is not None and crop.image.shape == (256, 80, 3) and crop.object_type is ObjectType.PERSON
    assert s.select(observation(ObjectType.VEHICLE, (600, 200, 680, 456))) is None
    assert s.select(observation(ObjectType.PERSON, (600, 200, 640, 260))) is None
    assert s.rejected == {"too small": 1}
