from __future__ import annotations

import threading
import time
import unittest
from unittest.mock import patch

from services.api.app import embedder_cache


class EmbedderCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        embedder_cache.clear()

    def tearDown(self) -> None:
        embedder_cache.clear()

    @staticmethod
    def _kwargs(**overrides) -> dict:
        params = dict(
            device="cpu",
            person_weights="p.pth",
            person_config="p.yml",
            vehicle_weights="v.pth",
            vehicle_config="v.yml",
        )
        params.update(overrides)
        return params

    def test_second_call_reuses_instance(self) -> None:
        sentinel = object()
        with patch.object(embedder_cache, "build_embedding_service", return_value=sentinel) as build:
            first = embedder_cache.get_embedder("fastreid", **self._kwargs())
            second = embedder_cache.get_embedder("fastreid", **self._kwargs())
        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        build.assert_called_once()

    def test_different_configuration_builds_separate_instance(self) -> None:
        outputs = iter([object(), object()])
        with patch.object(embedder_cache, "build_embedding_service", side_effect=lambda *a, **k: next(outputs)) as build:
            first = embedder_cache.get_embedder("fastreid", **self._kwargs(device="cpu"))
            second = embedder_cache.get_embedder("fastreid", **self._kwargs(device="cuda"))
        self.assertIsNot(first, second)
        self.assertEqual(build.call_count, 2)

    def test_build_failure_is_not_cached(self) -> None:
        calls: list[int] = []

        def failing_then_ok(*args, **kwargs):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("weights missing")
            return "ok"

        with patch.object(embedder_cache, "build_embedding_service", side_effect=failing_then_ok):
            with self.assertRaises(RuntimeError):
                embedder_cache.get_embedder("fastreid", **self._kwargs())
            self.assertEqual(embedder_cache.get_embedder("fastreid", **self._kwargs()), "ok")
        self.assertEqual(len(calls), 2)

    def test_concurrent_requests_build_once(self) -> None:
        started = threading.Barrier(8)
        results: list[object] = []
        errors: list[BaseException] = []

        def slow_build(*args, **kwargs):
            time.sleep(0.05)
            return object()

        with patch.object(embedder_cache, "build_embedding_service", side_effect=slow_build) as build:
            def call() -> None:
                try:
                    started.wait(timeout=5)
                    results.append(embedder_cache.get_embedder("fastreid", **self._kwargs()))
                except BaseException as exc:  # surface thread failures in the main assertion
                    errors.append(exc)

            threads = [threading.Thread(target=call) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

        self.assertEqual(errors, [])
        self.assertEqual(build.call_count, 1)
        self.assertEqual(len(results), 8)
        self.assertEqual(len(set(map(id, results))), 1)

    def test_embedder_from_env_forwards_configuration(self) -> None:
        captured: dict[str, dict] = {}

        def fake_build(backend, **kwargs):
            captured["backend"] = backend
            captured.update(kwargs)
            return "service"

        env = {
            "REID_BACKEND": "fastreid",
            "DEVICE": "cpu",
            "PERSON_REID_WEIGHTS": "pw.pth",
            "PERSON_REID_CONFIG": "pw.yml",
            "VEHICLE_REID_WEIGHTS": "vw.pth",
            "VEHICLE_REID_CONFIG": "vw.yml",
        }
        with patch.object(embedder_cache, "build_embedding_service", side_effect=fake_build):
            with patch.dict(embedder_cache.os.environ, env, clear=False):
                self.assertEqual(embedder_cache.embedder_from_env(), "service")
        self.assertEqual(captured["backend"], "fastreid")
        self.assertEqual(captured["person_weights"], "pw.pth")
        self.assertEqual(captured["vehicle_config"], "vw.yml")
        self.assertEqual(captured["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
