from __future__ import annotations

import argparse
import random
import unittest

import numpy as np

import scripts.calibrate_thresholds as calib


def make_args(**overrides) -> argparse.Namespace:
    params = dict(
        max_genuine_pairs=20000,
        max_impostor_pairs=20000,
        target_precision=0.95,
    )
    params.update(overrides)
    return argparse.Namespace(**params)


def separable_tracks(rng: random.Random, tracks: int = 4, samples: int = 8, dim: int = 16) -> dict[str, list]:
    """Each track clusters around its own unit direction so classes separate."""

    per_track: dict[str, list] = {}
    for track in range(tracks):
        base = np.zeros(dim)
        base[track * (dim // tracks) : (track + 1) * (dim // tracks)] = 1.0
        base /= np.linalg.norm(base)
        vectors = []
        for _ in range(samples):
            noisy = base + np.array([rng.gauss(0.0, 0.05) for _ in range(dim)])
            vectors.append(noisy / np.linalg.norm(noisy))
        per_track[f"track-{track}"] = vectors
    return per_track


class RocTests(unittest.TestCase):
    def test_perfect_separation_gives_zero_eer_and_clean_thresholds(self) -> None:
        genuine = np.array([0.9, 0.92, 0.95, 0.97])
        impostor = np.array([0.1, 0.2, 0.3, 0.4])
        roc = calib._roc(genuine, impostor)
        self.assertEqual(roc["eer"], 0.0)
        review = calib._review_threshold(roc)
        confirm, achieved = calib._confirm_threshold(roc, review, 0.95)
        self.assertGreaterEqual(review, 0.4)
        self.assertLessEqual(review, 0.9)
        self.assertGreaterEqual(confirm, review)
        self.assertGreaterEqual(achieved, 0.95)

    def test_confirm_never_below_review_and_precision_index_safe(self) -> None:
        genuine = np.array([0.6, 0.65, 0.7])
        impostor = np.array([0.55, 0.58, 0.62])
        roc = calib._roc(genuine, impostor)
        review = calib._review_threshold(roc)
        confirm, achieved = calib._confirm_threshold(roc, review, 0.99)
        self.assertGreaterEqual(confirm, review)
        self.assertLessEqual(confirm, 1.0)
        self.assertGreaterEqual(achieved, 0.0)


class CalibrateClassTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rng = random.Random(7)

    def test_separable_tracks_calibrate_with_review_below_confirm(self) -> None:
        result = calib.calibrate_class("person", separable_tracks(self.rng), make_args(), self.rng)
        self.assertEqual(result["status"], "calibrated")
        self.assertGreaterEqual(result["genuine_pairs"], calib.MIN_GENUINE_PAIRS)
        self.assertLessEqual(result["review_threshold"], result["confirm_threshold"])
        self.assertGreater(result["genuine"]["p50"], result["impostor"]["p50"])  # sanity: genuine closer
        self.assertGreater(result["retrieval"]["top1_accuracy"], 0.9)

    def test_insufficient_pairs_reports_status(self) -> None:
        per_track = {"track-1": [np.array([1.0, 0.0]), np.array([1.0, 0.0])]}
        result = calib.calibrate_class("person", per_track, make_args(), self.rng)
        self.assertEqual(result["status"], "insufficient_pairs")

    def test_margin_falls_back_when_retrieval_has_no_errors(self) -> None:
        result = calib.calibrate_class("person", separable_tracks(self.rng), make_args(), self.rng)
        self.assertEqual(result["min_score_margin"], calib.DEFAULT_MARGIN)
        self.assertEqual(result["retrieval"]["errors"], 0)

    def test_margin_from_gap_roc_when_errors_exist(self) -> None:
        # track-2's gallery representative (first sample) is unrelated to its
        # remaining crops, which overlap track-1: every track-2 query then
        # retrieves track-1 first (>= 10 errors), forcing the gap-ROC branch.
        rng = random.Random(11)
        dim = 8
        base = np.zeros(dim)
        base[0] = 1.0
        outlier = np.zeros(dim)
        outlier[3] = 1.0
        side = np.zeros(dim)
        side[1] = 1.0

        def noisy(direction: np.ndarray) -> np.ndarray:
            vector = direction + np.array([rng.gauss(0.0, 0.01) for _ in range(dim)])
            return vector / np.linalg.norm(vector)

        per_track = {
            "track-1": [noisy(base) for _ in range(12)],
            "track-2": [outlier] + [noisy(base + 0.02 * side) for _ in range(11)],
            "track-3": [noisy(np.eye(dim)[2]) for _ in range(6)],
        }
        result = calib.calibrate_class("person", per_track, make_args(), self.rng)
        self.assertEqual(result["status"], "calibrated")
        self.assertEqual(result["margin_source"], "gap-roc")
        self.assertGreaterEqual(result["retrieval"]["errors"], 10)
        self.assertGreaterEqual(result["min_score_margin"], 0.0)
        self.assertLessEqual(result["min_score_margin"], 0.5)


if __name__ == "__main__":
    unittest.main()
