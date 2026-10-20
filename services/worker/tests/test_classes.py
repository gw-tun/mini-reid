from services.worker.app.detection.classes import coco_ids_for, object_type_for
from services.worker.app.domain.models import ObjectType


def test_person_and_vehicle_classes_map_to_their_pipelines():
    assert object_type_for(0) == ("person", ObjectType.PERSON)
    assert object_type_for(2) == ("car", ObjectType.VEHICLE)
    assert object_type_for(7) == ("truck", ObjectType.VEHICLE)


def test_ignored_classes_return_none():
    assert object_type_for(1) is None   # bicycle: the rider is the person
    assert object_type_for(16) is None  # dog


def test_detector_is_asked_only_for_wanted_classes():
    assert coco_ids_for({ObjectType.PERSON}) == [0]
    assert coco_ids_for({ObjectType.PERSON, ObjectType.VEHICLE}) == [0, 2, 3, 5, 7]
