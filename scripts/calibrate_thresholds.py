"""Calibrate ReID similarity thresholds from a real recording.

Labels come from tracker pseudo-labels inside one segment:
  genuine  = two accepted crops of the same track at different times
  impostor = crops from two different tracks of the same object class

From those pairs the script derives ROC/EER, a review threshold (Youden J),
a confirm threshold (target precision), and a top1/top2 score margin from a
retrieval-gap ROC, then writes a JSON report. With --apply it also updates
global `reid_threshold_policies` rows in PostgreSQL and prints the matching
REID_MIN_SIMILARITY / REID_MIN_MARGIN lines for .env.

Caveats are recorded in the report: single-camera short-term pairs are
optimistic for genuine scores, while track fragmentation can leak a genuine
pair into the impostor set (biases thresholds upward = conservative).
Re-run against multi-camera ground-truth identities when available.

Usage (from the repository root):
    python scripts/calibrate_thresholds.py --video data/recordings/cam01/segment.mp4 --apply

Montage recordings often contain person appearances only in one time window;
pass --start-second/--end-second to score only that window, and --sample-fps
higher (e.g. 4) so tracks survive long enough to yield genuine pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from services.worker.app.cli import build_embedding_service

MIN_GENUINE_PAIRS = 20
MIN_IMPOSTOR_PAIRS = 20
MIN_MARGIN_ERRORS = 10
DEFAULT_MARGIN = 0.05

CAVEATS = [
    "pseudo-labels: track id == identity, so track fragmentation inflates impostor scores (conservative bias)",
    "genuine pairs are short-term same-camera crops, which is optimistic versus cross-camera re-identification",
    "single source segment: thresholds should be re-validated when multi-camera ground-truth data exists",
]


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0:
        raise ValueError("zero-norm embedding cannot be calibrated")
    return (vector / norm).astype("float64")


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": round(float(values.min()), 4),
        "p05": round(float(np.percentile(values, 5)), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p95": round(float(np.percentile(values, 95)), 4),
        "max": round(float(values.max()), 4),
        "mean": round(float(values.mean()), 4),
    }


def _rate_below(sorted_values: np.ndarray, threshold: float) -> np.ndarray:
    """Fraction of values strictly below each threshold (vectorized search)."""

    counts = np.searchsorted(sorted_values, threshold, side="left")
    return counts / len(sorted_values)


def _roc(genuine: np.ndarray, impostor: np.ndarray) -> dict[str, float]:
    thresholds = np.unique(np.concatenate([genuine, impostor]))
    g_sorted = np.sort(genuine)
    i_sorted = np.sort(impostor)
    tpr = 1.0 - _rate_below(g_sorted, thresholds)
    fpr = 1.0 - _rate_below(i_sorted, thresholds)
    fnr = 1.0 - tpr

    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    precision = np.divide(
        tpr * len(genuine),
        tpr * len(genuine) + fpr * len(impostor),
        out=np.zeros_like(tpr),
        where=(tpr * len(genuine) + fpr * len(impostor)) > 0,
    )
    return {
        "eer": round(float((fpr[eer_index] + fnr[eer_index]) / 2), 4),
        "eer_threshold": round(float(thresholds[eer_index]), 4),
        "youden_index": int(np.argmax(tpr - fpr)),
        "thresholds": thresholds,
        "precision": precision,
    }


def _review_threshold(roc: dict) -> float:
    return float(roc["thresholds"][roc["youden_index"]])


def _confirm_threshold(roc: dict, review: float, target_precision: float) -> tuple[float, float]:
    thresholds = roc["thresholds"]
    precision = roc["precision"]
    candidates = np.flatnonzero(precision >= target_precision)
    confirm = float(thresholds[candidates[0]]) if candidates.size else 1.0
    confirm = max(review, min(confirm, 1.0))
    probe = min(int(np.searchsorted(thresholds, confirm, side="left")), len(thresholds) - 1)
    achieved = float(precision[probe]) if thresholds.size else 0.0
    return confirm, achieved


def _margin(samples: np.ndarray, track_ids: list[str]) -> tuple[float, str, dict]:
    """Calibrate MatchPolicy's top1/top2 margin from a retrieval simulation.

    The gallery holds one representative (first) crop per track, exactly like
    production where the index stores one embedding per track; queries are the
    remaining crops. A margin below which a top1/top2 gap counts as ambiguous
    is chosen by ROC over retrieval-error gaps versus correct gaps.
    """

    tracks = sorted(set(track_ids))
    if len(tracks) < 2:
        return DEFAULT_MARGIN, "default (fewer than 2 tracks)", {"queries": 0, "errors": 0}
    first_row: dict[str, int] = {}
    for row, track in enumerate(track_ids):
        first_row.setdefault(track, row)
    gallery = np.stack([samples[first_row[track]] for track in tracks])
    query_rows = [row for row, track in enumerate(track_ids) if row != first_row[track]]
    if not query_rows:
        return DEFAULT_MARGIN, "default", {"queries": 0, "errors": 0}

    queries = samples[query_rows]
    query_tracks = np.array([track_ids[row] for row in query_rows])
    scores = queries @ gallery.T
    order = np.argsort(-scores, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    gaps = scores[np.arange(len(queries)), top1] - scores[np.arange(len(queries)), top2]
    correct = np.array([tracks[idx] == track for idx, track in zip(top1, query_tracks)])
    stats = {
        "queries": int(len(queries)),
        "errors": int((~correct).sum()),
        "top1_accuracy": round(float(correct.mean()), 4),
    }
    if int((~correct).sum()) < MIN_MARGIN_ERRORS:
        return DEFAULT_MARGIN, "default (fewer than 10 retrieval errors in calibration set)", stats

    error_gaps = np.sort(gaps[~correct])
    correct_gaps = np.sort(gaps[correct])
    candidate_margins = np.unique(gaps)
    reject_errors = _rate_below(error_gaps, candidate_margins)  # errors with gap < margin
    reject_correct = _rate_below(correct_gaps, candidate_margins)
    margin = float(candidate_margins[int(np.argmax(reject_errors - reject_correct))])
    margin = round(max(0.0, min(margin, 0.5)), 4)
    return margin, "gap-roc", stats


def collect_samples(args) -> dict:
    from services.worker.app.domain.models import ObjectType
    from services.worker.app.infrastructure.media import (
        IoUTracker,
        OpenCVCropSelector,
        OpenCVSegmentReader,
        UltralyticsDetector,
    )

    reader = OpenCVSegmentReader(args.camera)
    detector = UltralyticsDetector(args.yolo, float(os.getenv("YOLO_CONFIDENCE", "0.35")))
    tracker = IoUTracker()
    selector = OpenCVCropSelector()
    start_wall = datetime.fromtimestamp(Path(args.video).stat().st_mtime, tz=timezone.utc)

    buckets: dict[tuple[str, str], list] = {}
    frames_seen = 0
    scanned = 0
    chunk: list = []

    def flush() -> None:
        nonlocal scanned
        if not chunk:
            return
        for frame, detections in zip(chunk, detector.detect_batch(chunk)):
            scanned += 1
            for observation in tracker.update(frame, detections):
                crop = selector.select(observation)
                if crop is not None:
                    buckets.setdefault((observation.object_type.value, observation.track_id), []).append(
                        (frame.captured_at, crop)
                    )
        if scanned % 100 < len(chunk):
            print(f"  scanned {scanned} frames, {sum(len(v) for v in buckets.values())} crops ...")
        chunk.clear()

    for frame in reader.frames(args.video, args.sample_fps):
        video_second = (frame.captured_at - start_wall).total_seconds()
        if args.start_second is not None and video_second < args.start_second:
            continue
        if args.end_second is not None and video_second >= args.end_second:
            continue
        frames_seen += 1
        chunk.append(frame)
        if len(chunk) >= args.predict_batch:
            flush()
    flush()

    # Thin each track evenly so one long-lived track cannot dominate the cost.
    selected: dict[tuple[str, str], list] = {}
    for key, crops in buckets.items():
        if len(crops) > args.max_per_track:
            step = len(crops) / args.max_per_track
            crops = [crops[min(len(crops) - 1, int(i * step))] for i in range(args.max_per_track)]
        selected[key] = crops
    return {"frames_seen": frames_seen, "selected": selected, "ObjectType": ObjectType}


def embed_samples(collected: dict, args) -> dict[str, dict]:
    from services.worker.app.domain.models import CropSample

    model_root = Path(os.getenv("MODEL_ROOT", str(ROOT / "models")))
    embedder = build_embedding_service(
        args.reid_backend,
        device=args.device,
        person_weights=os.getenv("PERSON_REID_WEIGHTS", str(model_root / "fastreid" / "msmt17_sbs_R50-IBN.pth")),
        person_config=os.getenv("PERSON_REID_CONFIG", str(model_root / "fastreid" / "configs" / "msmt17_sbs_R50-IBN.yml")),
        vehicle_weights=os.getenv("VEHICLE_REID_WEIGHTS", str(model_root / "fastreid" / "veri_sbs_R50-IBN.pth")),
        vehicle_config=os.getenv("VEHICLE_REID_CONFIG", str(model_root / "fastreid" / "configs" / "veri_sbs_R50-IBN.yml")),
    )
    object_type = collected["ObjectType"]
    classes: dict[str, dict] = {}
    total = sum(len(v) for v in collected["selected"].values())
    done = 0
    for (class_name, track_id), crops in collected["selected"].items():
        vectors: list[np.ndarray] = []
        for _observed_at, crop in crops:
            embedding = embedder.embed(CropSample(crop.track_id, object_type(class_name), crop.image, crop.quality))
            if embedding is None:
                continue
            vectors.append(_normalize(np.asarray(embedding.values, dtype="float64")))
            done += 1
            if done % 50 == 0:
                print(f"  embedded {done}/{total} crops ...")
        if vectors:
            classes.setdefault(class_name, {})[track_id] = vectors
    return classes


def _pair_scores(samples: np.ndarray, pairs: list[tuple[int, int]], cap: int, rng: random.Random) -> np.ndarray:
    if not pairs:
        return np.empty(0, dtype="float64")
    if len(pairs) > cap:
        pairs = [pairs[index] for index in rng.sample(range(len(pairs)), cap)]
    rows = np.array([left for left, _ in pairs], dtype=int)
    cols = np.array([right for _, right in pairs], dtype=int)
    return np.einsum("ij,ij->i", samples[rows], samples[cols])


def calibrate_class(class_name: str, per_track: dict[str, list[np.ndarray]], args, rng: random.Random) -> dict:
    track_ids: list[str] = []
    rows: list[np.ndarray] = []
    for track_id in sorted(per_track):
        for vector in per_track[track_id]:
            track_ids.append(track_id)
            rows.append(vector)
    samples = np.stack(rows)
    index_by_track: dict[str, list[int]] = {}
    for row, track_id in enumerate(track_ids):
        index_by_track.setdefault(track_id, []).append(row)

    genuine_pairs = [
        (left, right)
        for indices in index_by_track.values()
        for offset, left in enumerate(indices)
        for right in indices[offset + 1 :]
    ]
    tracks = sorted(index_by_track)
    impostor_pairs = [
        (left, right)
        for offset, first in enumerate(tracks)
        for second in tracks[offset + 1 :]
        for left in index_by_track[first]
        for right in index_by_track[second]
    ]
    genuine = _pair_scores(samples, genuine_pairs, args.max_genuine_pairs, rng)
    impostor = _pair_scores(samples, impostor_pairs, args.max_impostor_pairs, rng)

    result: dict = {
        "tracks": len(tracks),
        "samples": int(len(samples)),
        "genuine_pairs": int(len(genuine)),
        "impostor_pairs": int(len(impostor)),
    }
    if genuine.size:
        result["genuine"] = _percentiles(genuine)
    if impostor.size:
        result["impostor"] = _percentiles(impostor)
    if len(genuine) < MIN_GENUINE_PAIRS or len(impostor) < MIN_IMPOSTOR_PAIRS:
        result["status"] = "insufficient_pairs"
        return result

    roc = _roc(genuine, impostor)
    review = round(_review_threshold(roc), 4)
    confirm, achieved = _confirm_threshold(roc, review, args.target_precision)
    confirm = round(confirm, 4)
    margin, margin_source, retrieval = _margin(samples, track_ids)
    result.update({
        "status": "calibrated",
        "eer": roc["eer"],
        "eer_threshold": roc["eer_threshold"],
        "review_threshold": review,
        "confirm_threshold": confirm,
        "precision_at_confirm": round(achieved, 4),
        "min_score_margin": margin,
        "margin_source": margin_source,
        "retrieval": retrieval,
    })
    return result


def apply_to_postgres(report: dict) -> None:
    import psycopg

    from services.worker.app.infrastructure.postgres import _dsn

    updated = 0
    with psycopg.connect(_dsn()) as connection:
        with connection.cursor() as cursor:
            for class_name, values in report["classes"].items():
                if values.get("status") != "calibrated":
                    continue
                cursor.execute(
                    """
                    UPDATE reid_threshold_policies AS p
                    SET review_threshold = %s,
                        confirm_threshold = %s,
                        min_score_margin = %s
                    FROM reid_models AS m
                    WHERE p.model_id = m.model_id
                      AND m.object_class = %s::object_class
                      AND p.camera_from_id IS NULL
                      AND p.camera_to_id IS NULL
                    """,
                    (
                        values["review_threshold"],
                        values["confirm_threshold"],
                        values["min_score_margin"],
                        class_name,
                    ),
                )
                print(f"  {class_name}: {cursor.rowcount} policy row(s) updated")
                updated += cursor.rowcount
        connection.commit()
    print(f"applied {updated} threshold policy row(s)")


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Calibrate ReID thresholds from one real recording")
    parser.add_argument("--video", default="data/recordings/cam01/segment.mp4")
    parser.add_argument("--camera", default="cam01")
    parser.add_argument("--yolo", default=os.getenv("YOLO_WEIGHTS", str(ROOT / "models" / "yolo" / "yolo11m.pt")))
    parser.add_argument("--reid-backend", choices=["histogram", "fastreid"], default=os.getenv("REID_BACKEND", "fastreid"))
    parser.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--start-second", type=float, default=None, help="skip video time before this second")
    parser.add_argument("--end-second", type=float, default=None, help="skip video time at/after this second")
    parser.add_argument("--predict-batch", type=int, default=8, help="frames per YOLO call (CPU batching)")
    parser.add_argument("--max-per-track", type=int, default=12)
    parser.add_argument("--max-genuine-pairs", type=int, default=20000)
    parser.add_argument("--max-impostor-pairs", type=int, default=20000)
    parser.add_argument("--target-precision", type=float, default=0.95)
    parser.add_argument("--report", default="data/cache/threshold-calibration.json")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--apply", action="store_true", help="update reid_threshold_policies in PostgreSQL")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    print(f"scanning {args.video} at {args.sample_fps:g} fps (camera={args.camera}) ...")
    collected = collect_samples(args)
    tracks_seen = len({key[1] for key in collected["selected"]})
    print(f"frames={collected['frames_seen']} tracks={tracks_seen}; embedding crops ...")
    classes = embed_samples(collected, args)

    report: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "video": args.video,
        "camera": args.camera,
        "backend": args.reid_backend,
        "device": args.device,
        "sample_fps": args.sample_fps,
        "window_seconds": [args.start_second, args.end_second],
        "predict_batch": args.predict_batch,
        "frames_seen": collected["frames_seen"],
        "target_precision": args.target_precision,
        "caveats": CAVEATS,
        "classes": {},
    }
    for class_name, per_track in sorted(classes.items()):
        print(f"calibrating {class_name}: {len(per_track)} tracks ...")
        report["classes"][class_name] = calibrate_class(class_name, per_track, args, rng)

    primary = report["classes"].get("person") or next(
        (v for v in report["classes"].values() if v.get("status") == "calibrated"), None
    )
    if primary and primary.get("status") == "calibrated":
        report["suggested_env"] = {
            "REID_MIN_SIMILARITY": primary["review_threshold"],
            "REID_MIN_MARGIN": primary["min_score_margin"],
        }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"report: {report_path}")

    for class_name, values in report["classes"].items():
        if values.get("status") == "calibrated":
            print(
                f"  {class_name}: review={values['review_threshold']} "
                f"confirm={values['confirm_threshold']} margin={values['min_score_margin']} "
                f"({values['margin_source']}, eer={values['eer']}, n_g={values['genuine_pairs']}, "
                f"n_i={values['impostor_pairs']})"
            )
        else:
            print(f"  {class_name}: not calibrated ({values.get('status')})")
    if "suggested_env" in report:
        print("add to .env:")
        for key, value in report["suggested_env"].items():
            print(f"  {key}={value}")

    if args.apply:
        apply_to_postgres(report)


if __name__ == "__main__":
    main()
