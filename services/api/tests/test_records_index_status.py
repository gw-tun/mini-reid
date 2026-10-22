from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from services.api.app.api import records


class QueueSegmentStatesTest(unittest.TestCase):
    def _write_job(self, directory: Path, name: str, segment_path: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        payload = {
            "job_id": name,
            "event": {
                "event_type": "record.segment.completed",
                "segment_path": segment_path,
            },
        }
        (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_reports_queued_processing_and_failed_states(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_job(root / "pending", "a", "/recordings/cam01/2026-10-21_07-00-00.mp4")
            self._write_job(root / "processing", "b", "/recordings/cam01/2026-10-21_07-01-00.mp4")
            self._write_job(root / "failed", "c", "/recordings/cam02/2026-10-21_07-02-00.mp4")
            original = records._QUEUE_DIR
            records._QUEUE_DIR = root
            try:
                states = records._queue_segment_states()
            finally:
                records._QUEUE_DIR = original

        self.assertEqual(states[("cam01", "2026-10-21_07-00-00.mp4")], "queued")
        self.assertEqual(states[("cam01", "2026-10-21_07-01-00.mp4")], "processing")
        self.assertEqual(states[("cam02", "2026-10-21_07-02-00.mp4")], "failed")

    def test_processing_wins_over_queued_for_same_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segment = "/recordings/cam01/2026-10-21_07-00-00.mp4"
            self._write_job(root / "pending", "a", segment)
            self._write_job(root / "processing", "b", segment)
            original = records._QUEUE_DIR
            records._QUEUE_DIR = root
            try:
                states = records._queue_segment_states()
            finally:
                records._QUEUE_DIR = original

        self.assertEqual(states[("cam01", "2026-10-21_07-00-00.mp4")], "processing")

    def test_unreadable_job_files_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pending").mkdir(parents=True, exist_ok=True)
            (root / "pending" / "broken.json").write_text("{not json", encoding="utf-8")
            (root / "pending" / "empty.json").write_text("{}", encoding="utf-8")
            original = records._QUEUE_DIR
            records._QUEUE_DIR = root
            try:
                states = records._queue_segment_states()
            finally:
                records._QUEUE_DIR = original

        self.assertEqual(states, {})
