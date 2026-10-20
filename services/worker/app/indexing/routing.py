"""Send each object to its own class's crop rules and ReID model.

The use cases (`RecordIndexingService`, `LiveCameraService`) take ONE crop
selector and ONE ReID model. These composites fill those slots and dispatch on
the object type, so person and vehicle pipelines plug in side by side without
changing the use cases (open/closed):

    crop_selector = ClassRoutedCropSelector({PERSON: PersonCropSelector(),
                                             VEHICLE: VehicleCropSelector()})   # VIP-8
    reid          = ClassRoutedReID({PERSON: person_reid,                      # VIP-9
                                     VEHICLE: vehicle_reid})                    # VIP-8

A class with no entry is skipped: nothing is embedded rather than embedded by
the wrong model. Person and vehicle vectors live in different vector spaces
and must never be compared.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..domain.models import CropSample, Embedding, ObjectType, TrackObservation
from ..domain.ports import CropSelector, ReIDModel


class ClassRoutedCropSelector:
    def __init__(self, by_type: Mapping[ObjectType, CropSelector]) -> None:
        self._by_type = dict(by_type)

    def select(self, observation: TrackObservation) -> CropSample | None:
        selector = self._by_type.get(observation.object_type)
        return selector.select(observation) if selector else None


class ClassRoutedReID:
    def __init__(self, by_type: Mapping[ObjectType, ReIDModel]) -> None:
        self._by_type = dict(by_type)

    def embed(self, crop: CropSample) -> Embedding | None:
        model = self._by_type.get(crop.object_type)
        return model.embed(crop) if model else None
