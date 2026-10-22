"""Host/container media path resolution."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from services.common.paths import resolve_media_path


class ResolveMediaPathTests(unittest.TestCase):
    def test_returns_existing_path_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "seg.mp4"
            target.write_bytes(b"x")
            self.assertEqual(resolve_media_path(str(target), recordings_root=temp_dir), str(target))

    def test_maps_container_recording_path_onto_host_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            host_file = Path(temp_dir) / "cam01" / "seg.mp4"
            host_file.parent.mkdir(parents=True)
            host_file.write_bytes(b"x")
            resolved = resolve_media_path("/recordings/cam01/seg.mp4", recordings_root=temp_dir)
            self.assertEqual(resolved, str(host_file))

    def test_returns_original_when_nothing_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resolved = resolve_media_path("/recordings/cam09/missing.mp4", recordings_root=temp_dir)
            self.assertEqual(resolved, "/recordings/cam09/missing.mp4")

    def test_default_root_env_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            host_file = Path(temp_dir) / "cam01" / "seg.mp4"
            host_file.parent.mkdir(parents=True)
            host_file.write_bytes(b"x")
            previous = os.environ.get("RECORDINGS_DIR")
            os.environ["RECORDINGS_DIR"] = temp_dir
            try:
                self.assertEqual(
                    resolve_media_path("/recordings/cam01/seg.mp4"), str(host_file)
                )
            finally:
                if previous is None:
                    os.environ.pop("RECORDINGS_DIR", None)
                else:
                    os.environ["RECORDINGS_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
