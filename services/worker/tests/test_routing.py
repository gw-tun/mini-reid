import datetime as dt

import numpy as np

from services.worker.app.domain.models import CropSample, Embedding, ObjectType, TrackObservation
from services.worker.app.indexing.routing import ClassRoutedCropSelector, ClassRoutedReID


class Tag:
    """Fake selector / ReID that says which class pipeline handled the object."""

    def __init__(self, name):
        self.name = name

    def select(self, o):
        return CropSample(o.track_id, o.object_type, self.name, 1.0)

    def embed(self, crop):
        return Embedding(f"{self.name}-space", (1.0,), "v1")


def obs(t):
    return TrackObservation("cam", "1", t, dt.datetime(2026, 1, 1), (0, 0, 1, 1), 0.9, np.zeros((1, 1, 3)))


def test_each_class_goes_to_its_own_crop_rules():
    r = ClassRoutedCropSelector({ObjectType.PERSON: Tag("person"), ObjectType.VEHICLE: Tag("vehicle")})
    assert r.select(obs(ObjectType.PERSON)).image == "person"
    assert r.select(obs(ObjectType.VEHICLE)).image == "vehicle"


def test_each_class_goes_to_its_own_reid_model_and_vector_space():
    r = ClassRoutedReID({ObjectType.PERSON: Tag("msmt17"), ObjectType.VEHICLE: Tag("veri")})
    person = r.embed(CropSample("1", ObjectType.PERSON, None, 1.0))
    vehicle = r.embed(CropSample("2", ObjectType.VEHICLE, None, 1.0))
    assert person.vector_space == "msmt17-space" and vehicle.vector_space == "veri-space"


def test_a_class_without_a_model_is_skipped_not_misrouted():
    r = ClassRoutedReID({ObjectType.PERSON: Tag("msmt17")})
    assert r.embed(CropSample("2", ObjectType.VEHICLE, None, 1.0)) is None
