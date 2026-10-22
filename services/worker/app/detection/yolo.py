"""YOLO detector adapter (implements the `Detector` port).

One YOLO pass finds every kept class; `classes.py` maps each box to PERSON or
VEHICLE. Ultralytics is imported only when the detector is built, so the rest
of the worker (and its tests) run without it.

Licence note: Ultralytics YOLO is AGPL-3.0 -- see docs/VIP-9-PERSON-PLAN.md §5.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import numpy as np

from ..device import pick_device
from ..domain.models import Detection, Frame, ObjectType
from .classes import coco_ids_for, object_type_for

DEFAULT_WEIGHTS = pathlib.Path("models/yolo/yolo11m.pt")


class YoloDetector:
    def __init__(
        self,
        weights: str | pathlib.Path = DEFAULT_WEIGHTS,
        *,
        device: str | None = None,
        object_types: set[ObjectType] | None = None,
        min_confidence: float = 0.25,
        image_size: int = 640,
    ) -> None:
        """`min_confidence` is low on purpose: weak boxes help a tracker keep a
        half-hidden person (ByteTrack's second pass). Each crop selector applies
        its own, stricter rule before anything is embedded."""
        from ultralytics import YOLO  # heavy: only when a detector is built

        path = pathlib.Path(weights)
        if not path.is_absolute():  # relative to the repository, wherever we were started
            path = pathlib.Path(__file__).resolve().parents[4] / path
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: run scripts/download_models.sh")
        self.device = pick_device(device)
        self.model = YOLO(str(path))
        self.classes = coco_ids_for(object_types or {ObjectType.PERSON, ObjectType.VEHICLE})
        self.min_confidence = min_confidence
        self.image_size = image_size
        self.model_key = f"{path.stem}@{image_size}"
        self.weights = path

    def detect(self, frame: Frame) -> Sequence[Detection]:
        # Ultralytics reads NumPy images as BGR (OpenCV's order); frames are RGB.
        bgr = np.ascontiguousarray(frame.image[..., ::-1])
        result = self.model.predict(
            bgr,
            classes=self.classes,
            conf=self.min_confidence,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )[0]
        boxes = result.boxes
        found: list[Detection] = []
        for (x1, y1, x2, y2), conf, cls in zip(
            boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()
        ):
            mapped = object_type_for(int(cls))
            if mapped is None or x2 <= x1 or y2 <= y1:
                continue
            label, object_type = mapped
            found.append(Detection(object_type, float(conf),
                                   (float(x1), float(y1), float(x2), float(y2)), label))
        return found
