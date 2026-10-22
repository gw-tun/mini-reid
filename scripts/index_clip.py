"""Index one recording for person search: video -> people -> tracks -> best crops
-> barcodes from each ReID model -> PostgreSQL, plus the crops as files.

    .venv/bin/python scripts/index_clip.py VIDEO_OR_FOLDER ... [--device mps] [--reid osnet fastreid]
                     [--camera cam01 --start 2026-10-20T08:00:00] [--fps 2] [--force] [--dsn ...]

A folder means every .mp4 under it, e.g. data/recordings/semantic-va (written by
scripts/fetch_search_clips.py). The models load once for all of them, and a file
already in the database is skipped unless --force.

Writes:
    data/crops/<camera>/<recording>/<track>/<ms>.jpg   each track's best crops (the evidence)
    cameras, recordings          one row each
    detections                   every person box, kept or rejected (payload says why)
    tracks, track_observations   one row per person track, one per confirmed box
    reid_models                  one row per model (key + weights SHA-256)
    reid_embeddings_2048 / _512  one barcode per best crop and model

Indexing the same file again updates rows instead of doubling them. The work
itself is services/worker/app/application/person_indexer.py. The tables are
drafts until agreed with VIP-8 (infra/postgres/init/002, 003).
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.common.database import redact, resolve_dsn  # noqa: E402
from services.worker.app.application.person_indexer import PersonIndexer  # noqa: E402
from services.worker.app.detection.yolo import DEFAULT_WEIGHTS, YoloDetector  # noqa: E402
from services.worker.app.domain.models import ObjectType  # noqa: E402
from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader, describe_segment  # noqa: E402

MODELS = {"osnet": ("services.worker.app.reid.osnet", "OsnetAinReID"),
          "fastreid": ("services.worker.app.reid.fastreid", "FastReIDPersonReID")}
DATA = ROOT / "data"

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("paths", nargs="+", type=pathlib.Path, help="video files and/or folders of them")
p.add_argument("--fps", type=float, default=2.0)
p.add_argument("--device", help="cpu | mps | cuda:0 (default: DEVICE env, else auto)")
p.add_argument("--reid", nargs="+", choices=list(MODELS), default=list(MODELS))
p.add_argument("--camera", help="for ONE file whose path doesn't name its camera")
p.add_argument("--start", help="for ONE file whose name doesn't give its recording start (ISO time)")
p.add_argument("--force", action="store_true", help="index files that are already in the database again")
p.add_argument("--dsn", help="database (default: DATABASE_URL, .env, then localhost:5434)")
args = p.parse_args()

videos = sorted({v.resolve() for path in args.paths
                 for v in (path.rglob("*.mp4") if path.is_dir() else [path])
                 if not v.name.endswith(".part.mp4")})
if not videos:
    sys.exit("no .mp4 files there")
if (args.camera or args.start) and len(videos) > 1:
    sys.exit("--camera and --start are for a single file")
start = dt.datetime.fromisoformat(args.start) if args.start else None
if start is not None and start.tzinfo is None:
    start = start.replace(tzinfo=dt.timezone.utc)
plan = []
for video in videos:
    named_camera, named_start = describe_segment(video)
    if (start or named_start) is None:
        sys.exit(f"{video.name}: no recording time in the file name; pass --start")
    plan.append((video, args.camera or named_camera, start or named_start,
                 str(video.relative_to(DATA)) if video.is_relative_to(DATA) else str(video)))

import psycopg  # noqa: E402  (after the argument checks, so --help works without it)

dsn = resolve_dsn(args.dsn)
try:  # fail before minutes of video work, not after
    with psycopg.connect(dsn, connect_timeout=5) as conn:
        done = {(c, u) for c, u in conn.execute(
            "SELECT c.camera_key, r.media_uri FROM recordings r JOIN cameras c ON c.camera_id = r.camera_id")}
except psycopg.OperationalError as exc:
    sys.exit(f"cannot reach {redact(dsn)}: {str(exc).strip().splitlines()[0]}\n"
             "start it with: docker compose up -d postgres; then: .venv/bin/python scripts/migrate.py")
todo = [x for x in plan if args.force or (x[1], x[3]) not in done]
print(f"{len(plan)} videos, {len(plan) - len(todo)} already indexed" + (" (--force: again)" if args.force else ""))
if not todo:
    sys.exit(0)

detector = YoloDetector(ROOT / DEFAULT_WEIGHTS, device=args.device, object_types={ObjectType.PERSON})
models = [getattr(importlib.import_module(MODELS[n][0]), MODELS[n][1])(device=args.device) for n in args.reid]
indexer = PersonIndexer(detector, models, DATA / "crops")
print(f"{detector.model_key} on {detector.device}, ReID {', '.join(m.model_key for m in models)}, {args.fps} fps\n")

totals = dict(frames=0, detections=0, kept=0, tracks=0, crops=0)
for n, (video, camera, started_at, media_uri) in enumerate(todo, start=1):
    shutil.rmtree(DATA / "crops" / camera / video.stem, ignore_errors=True)  # this recording's crops are regenerated
    reader = PyAVSegmentReader(camera_key=camera, started_at=started_at)
    with psycopg.connect(dsn) as conn:  # one transaction per file: all of it or nothing
        r = indexer.index(conn, reader.frames(str(video), args.fps), camera_key=camera, media_uri=media_uri,
                          recording_started_at=started_at, file_size=video.stat().st_size)
    crops = sum(t.crops for t in r.tracks)
    for key, value in (("frames", r.frames), ("detections", r.detections), ("kept", r.kept),
                       ("tracks", len(r.tracks)), ("crops", crops)):
        totals[key] += value
    print(f"[{n}/{len(todo)}] {camera}/{video.name} ({started_at:%Y-%m-%d %H:%M:%S}): {r.frames} frames,"
          f" {r.detections} person boxes ({r.kept} kept), {len(r.tracks)} tracks, {crops} best crops")

print(f"\ntotal: {totals['frames']} frames -> {totals['detections']} person boxes ({totals['kept']} kept for ReID)"
      f" -> {totals['tracks']} tracks, {totals['crops']} best crops, each with {len(models)} barcodes")
print(f"database {redact(dsn)}; look at them: http://localhost:3001")
