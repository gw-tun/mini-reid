from __future__ import annotations

import unittest
from datetime import datetime, timezone

from services.worker.app.domain.models import Frame, ObjectType
from services.worker.app.infrastructure.media import UltralyticsDetector


class _FakeBoxes:
    def __init__(self, rows: list[tuple[list[float], float, float]]) -> None:
        self.xyxy = _Listish([row[0] for row in rows])
        self.conf = _Listish([row[1] for row in rows])
        self.cls = _Listish([row[2] for row in rows])


class _Listish:
    def __init__(self, values: list) -> None:
        self._values = values

    def tolist(self) -> list:
        return self._values


class _FakeResult:
    def __init__(self, rows: list[tuple[list[float], float, float]]) -> None:
        self.boxes = _FakeBoxes(rows)


class _FakeModel:
    def __init__(self, results: list[_FakeResult]) -> None:
        self._results = results
        self.calls: list[tuple[int, float]] = []

    def predict(self, images: list, conf: float = 0.0, verbose: bool = False):
        self.calls.append((len(images), conf))
        return self._results


def make_frame(name: str) -> Frame:
    return Frame("cam01", datetime(2026, 1, 1, tzinfo=timezone.utc), object(), name)


def make_detector(results: list[_FakeResult]) -> UltralyticsDetector:
    detector = object.__new__(UltralyticsDetector)  # skip YOLO loading in __init__
    detector._model = _FakeModel(results)
    detector._confidence = 0.35
    return detector


class DetectorBatchTests(unittest.TestCase):
    def test_detect_batch_maps_labels_and_skips_unknown_classes(self) -> None:
        person = (10.0, 20.0, 50.0, 120.0)
        vehicle = (100.0, 10.0, 300.0, 200.0)
        bicycle = (5.0, 5.0, 40.0, 40.0)
        results = [
            _FakeResult([(person, 0.91, 0.0), (bicycle, 0.8, 1.0)]),
            _FakeResult([(vehicle, 0.7, 2.0)]),
        ]
        detector = make_detector(results)

        batched = detector.detect_batch([make_frame("a"), make_frame("b")])

        self.assertEqual(len(batched), 2)
        self.assertEqual(len(batched[0]), 1)  # bicycle dropped
        self.assertIs(batched[0][0].object_type, ObjectType.PERSON)
        self.assertAlmostEqual(batched[0][0].confidence, 0.91)
        self.assertEqual(batched[0][0].bbox, person)
        self.assertIs(batched[1][0].object_type, ObjectType.VEHICLE)
        self.assertEqual(detector._model.calls, [(2, 0.35)])

    def test_detect_delegates_to_single_element_batch(self) -> None:
        results = [_FakeResult([((0.0, 0.0, 40.0, 80.0), 0.6, 0.0)])]
        detector = make_detector(results)

        detections = detector.detect(make_frame("only"))

        self.assertEqual(len(detections), 1)
        self.assertIs(detections[0].object_type, ObjectType.PERSON)
        self.assertEqual(detector._model.calls, [(1, 0.35)])

    def test_empty_batch_short_circuits_without_model_call(self) -> None:
        detector = make_detector([])
        self.assertEqual(detector.detect_batch([]), [])
        self.assertEqual(detector._model.calls, [])


if __name__ == "__main__":
    unittest.main()
