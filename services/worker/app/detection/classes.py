"""Which detector classes the system keeps, and which pipeline each one goes to.

YOLO's COCO model has no "vehicle" class: it detects car, bus, truck, and so on.
This table turns a detector class into an `ObjectType`, and the object type
decides everything downstream: crop rules, ReID model, vector space and
thresholds (see `indexing/routing.py`). Person and vehicle vectors never meet.

    person (0)                          -> PERSON   -> person ReID   (VIP-9)
    car (2), motorcycle (3),
    bus (5), truck (7)                  -> VEHICLE  -> vehicle ReID  (VIP-8)

Bicycle (1) is left out on purpose: the rider is detected as a person, and the
vehicle ReID model (VeRi) is trained on cars. VIP-8 owns the vehicle rows.
"""

from __future__ import annotations

from ..domain.models import ObjectType

COCO_CLASSES: dict[int, tuple[str, ObjectType]] = {
    0: ("person", ObjectType.PERSON),
    2: ("car", ObjectType.VEHICLE),
    3: ("motorcycle", ObjectType.VEHICLE),
    5: ("bus", ObjectType.VEHICLE),
    7: ("truck", ObjectType.VEHICLE),
}


def coco_ids_for(object_types: set[ObjectType]) -> list[int]:
    """COCO class ids to ask the detector for, given the object types we want."""
    return sorted(i for i, (_, t) in COCO_CLASSES.items() if t in object_types)


def object_type_for(coco_id: int) -> tuple[str, ObjectType] | None:
    """(COCO label, object type), or None for a class the system ignores."""
    return COCO_CLASSES.get(coco_id)
