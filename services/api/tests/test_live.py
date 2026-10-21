from __future__ import annotations

import json
import unittest

from fastapi import HTTPException

from services.api.app.api import live
from services.worker.app.domain.models import Detection, Frame, ObjectType


class _FakeImage:
    shape = (720, 1280, 3)


class _FakeCapture:
    def __init__(self, frames: list[tuple[bool, object]], opened: bool = True) -> None:
        self._frames = list(frames)
        self._opened = opened
        self.released = False

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, object]:
        if not self._frames:
            return False, None
        return self._frames.pop(0)

    def release(self) -> None:
        self.released = True


class NormalizeBoxesTest(unittest.TestCase):
    def test_pixel_boxes_become_clamped_frame_fractions(self) -> None:
        detections = [
            Detection(ObjectType.PERSON, 0.871234, (128.0, 72.0, 640.0, 576.0)),
            Detection(ObjectType.VEHICLE, 0.5, (-10.0, 0.0, 2000.0, 900.0)),
        ]

        boxes = live._normalize_boxes(detections, 1280, 720)

        self.assertEqual(len(boxes), 2)
        self.assertEqual(
            boxes[0],
            {"label": "person", "score": 0.871, "x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.8},
        )
        self.assertEqual(boxes[1]["label"], "vehicle")
        self.assertEqual(boxes[1]["x1"], 0.0)
        self.assertEqual(boxes[1]["x2"], 1.0)
        self.assertEqual(boxes[1]["y2"], 1.0)

    def test_zero_frame_size_yields_no_boxes(self) -> None:
        detections = [Detection(ObjectType.PERSON, 0.9, (0.0, 0.0, 10.0, 10.0))]

        self.assertEqual(live._normalize_boxes(detections, 0, 0), [])


class DetectionEventsTest(unittest.TestCase):
    def test_streams_detections_then_error_and_releases_capture(self) -> None:
        capture = _FakeCapture([(True, _FakeImage()), (False, None)])

        def detect(frame: Frame) -> list[Detection]:
            self.assertEqual(frame.camera_key, "cam01")
            return [Detection(ObjectType.PERSON, 0.9, (640.0, 0.0, 1280.0, 720.0))]

        events = list(
            live._detection_events(
                "cam01",
                2.0,
                open_capture=lambda: capture,
                detect=detect,
                sleep=lambda _seconds: None,
            )
        )

        self.assertTrue(capture.released)
        self.assertEqual(len(events), 2)
        self.assertTrue(events[0].startswith("data: ") and events[0].endswith("\n\n"))
        payload = json.loads(events[0].removeprefix("data: "))
        self.assertEqual(payload["type"], "detections")
        self.assertEqual(payload["frame"], {"width": 1280, "height": 720})
        self.assertEqual(payload["boxes"][0]["label"], "person")
        self.assertEqual(payload["boxes"][0]["x1"], 0.5)
        self.assertEqual(payload["boxes"][0]["x2"], 1.0)
        failure = json.loads(events[1].removeprefix("data: "))
        self.assertEqual(failure["type"], "error")

    def test_unopened_capture_reports_error_and_releases(self) -> None:
        capture = _FakeCapture([], opened=False)

        events = list(
            live._detection_events(
                "cam01",
                2.0,
                open_capture=lambda: capture,
                detect=lambda _frame: [],
                sleep=lambda _seconds: None,
            )
        )

        self.assertTrue(capture.released)
        self.assertEqual(len(events), 1)
        payload = json.loads(events[0].removeprefix("data: "))
        self.assertEqual(payload["type"], "error")
        self.assertEqual(payload["detail"], "camera stream unavailable")

    def test_frames_surface_before_queued_failure(self) -> None:
        capture = _FakeCapture([(True, _FakeImage()), (True, _FakeImage()), (False, None)])

        events = list(
            live._detection_events(
                "cam01",
                2.0,
                open_capture=lambda: capture,
                detect=lambda _frame: [],
                sleep=lambda _seconds: None,
            )
        )

        self.assertTrue(capture.released)
        self.assertGreaterEqual(len(events), 2)
        self.assertEqual(json.loads(events[0].removeprefix("data: "))["type"], "detections")
        self.assertEqual(json.loads(events[-1].removeprefix("data: "))["type"], "error")

    def test_latest_frame_keeps_newest_and_orders_failure_last(self) -> None:
        holder = live._LatestFrame()

        self.assertEqual(holder.await_next(0, timeout=0.01), ("timeout",))
        holder.put(object())
        holder.put("newest")
        self.assertEqual(holder.await_next(0, timeout=0.01), ("frame", 2, "newest"))
        holder.fail()
        self.assertEqual(holder.await_next(2, timeout=0.01), ("failed",))
        holder.close()
        self.assertEqual(holder.await_next(2, timeout=0.01), ("closed",))

    def test_closing_mid_stream_releases_capture(self) -> None:
        capture = _FakeCapture([(True, _FakeImage()) for _ in range(10)])
        stream = live._detection_events(
            "cam01",
            2.0,
            open_capture=lambda: capture,
            detect=lambda _frame: [],
            sleep=lambda _seconds: None,
        )

        first = next(stream)
        self.assertEqual(json.loads(first.removeprefix("data: "))["type"], "detections")
        stream.close()

        self.assertTrue(capture.released)


class LiveDetectionsRouteTest(unittest.TestCase):
    def test_rejects_invalid_camera_and_fps_before_runtime_load(self) -> None:
        with self.assertRaises(HTTPException) as camera_error:
            live.live_detections(camera="../etc", fps=2.0)
        self.assertEqual(camera_error.exception.status_code, 422)

        with self.assertRaises(HTTPException) as fps_error:
            live.live_detections(camera="cam01", fps=50.0)
        self.assertEqual(fps_error.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
