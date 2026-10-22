"""Worker runtime wiring: PostgreSQL/pgvector is the only vector store."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch


class WorkerRuntimeTests(unittest.TestCase):
    def test_build_runtime_requires_fastreid_for_postgres_gallery(self) -> None:
        from services.worker.app import worker as worker_module

        with tempfile.TemporaryDirectory() as temp_dir:
            weights = os.path.join(temp_dir, "yolo11m.pt")
            with open(weights, "wb"):
                pass
            env = {
                "REID_BACKEND": "histogram",
                "YOLO_WEIGHTS": weights,
                "JOB_QUEUE_DIR": os.path.join(temp_dir, "job-queue"),
                "YOLO_CONFIG_DIR": os.path.join(temp_dir, "ultralytics"),
            }
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaises(RuntimeError) as ctx:
                    worker_module.build_runtime()
        self.assertIn("REID_BACKEND=fastreid", str(ctx.exception))

    def test_json_vector_index_is_removed(self) -> None:
        from services.worker.app.infrastructure import media

        self.assertFalse(hasattr(media, "JsonVectorIndex"))


if __name__ == "__main__":
    unittest.main()
