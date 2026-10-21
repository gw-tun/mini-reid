"""Read/seek endpoints for the playback overlay and cross-record track search."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import psycopg
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field, model_validator

from services.common.paths import resolve_media_path
from services.worker.app.domain.models import Embedding
from services.worker.app.infrastructure.postgres import _dsn, PostgresVectorIndex

router = APIRouter(prefix="/internal", tags=["internal-records"])

_CROP_DIR = Path(os.getenv("CROP_CACHE_DIR", "data/cache/crops"))
_QUEUE_DIR = Path(os.getenv("JOB_QUEUE_DIR", "data/cache/job-queue"))
_PROBE_CACHE: dict[str, tuple[int, int, float, int]] = {}

_QUEUE_FOLDERS = (("pending", "queued"), ("processing", "processing"), ("failed", "failed"))
_QUEUE_PRIORITY = {"failed": 1, "queued": 2, "processing": 3}

_VECTOR_SPACE_BY_MODEL = {
    "msmt17_sbs_r50_ibn": "fastreid-msmt17-v1",
    "veri_sbs_r50_ibn": "fastreid-veri-v1",
}


class TrackSearchRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    camera_key: str = Field(min_length=1, max_length=64)
    segment: str | None = None
    track_db_id: int | None = None
    detection_id: int | None = None
    top_k: int = Field(default=50, ge=1, le=200)
    min_similarity: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_track_or_detection(self) -> "TrackSearchRequest":
        if self.track_db_id is None and self.detection_id is None:
            raise ValueError("track_db_id or detection_id is required")
        return self


def _probe_video(path: str) -> tuple[int, int, float, int] | None:
    """Return (width, height, fps, frame_count) with a tiny mtime-keyed cache."""

    try:
        stat = os.stat(path)
    except OSError:
        return None
    key = f"{path}:{stat.st_mtime}:{stat.st_size}"
    cached = _PROBE_CACHE.get(key)
    if cached is not None:
        return cached
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    probe = (width, height, fps, max(frames, 0))
    _PROBE_CACHE[key] = probe
    return probe


@router.get("/records")
def list_records(camera_key: str | None = None) -> dict[str, object]:
    query = """SELECT r.recording_id, c.camera_key, r.media_uri, r.started_at,
                      r.duration_ms, r.status,
                      (SELECT count(*) FROM detections d WHERE d.recording_id = r.recording_id),
                      (SELECT count(*) FROM tracks t WHERE t.recording_id = r.recording_id)
               FROM recordings r
               JOIN cameras c ON c.camera_id = r.camera_id"""
    params: list[object] = []
    if camera_key:
        query += " WHERE c.camera_key = %s"
        params.append(camera_key)
    query += " ORDER BY r.started_at DESC NULLS LAST LIMIT 200"
    with psycopg.connect(_dsn()) as connection:
        rows = connection.execute(query, params).fetchall()
    return {
        "records": [
            {
                "recording_id": int(row[0]),
                "camera_key": row[1],
                "segment": Path(row[2]).name,
                "media_uri": row[2],
                "started_at": row[3],
                "duration_seconds": (row[4] / 1000.0) if row[4] else None,
                "status": row[5],
                "detection_count": int(row[6]),
                "track_count": int(row[7]),
            }
            for row in rows
        ]
    }


def _queue_segment_states() -> dict[tuple[str, str], str]:
    """Map (camera_key, segment file) -> queue state for every pending job on disk."""

    states: dict[tuple[str, str], str] = {}
    if not _QUEUE_DIR.is_dir():
        return states
    for folder, state in _QUEUE_FOLDERS:
        directory = _QUEUE_DIR / folder
        if not directory.is_dir():
            continue
        priority = _QUEUE_PRIORITY.get(state, 0)
        for job_file in directory.glob("*.json"):
            try:
                payload = json.loads(job_file.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            event = payload.get("event")
            segment_path = event.get("segment_path") if isinstance(event, dict) else None
            if not segment_path:
                continue
            parts = Path(str(segment_path)).parts
            if len(parts) < 2:
                continue
            camera = parts[-2]
            key = (camera, parts[-1])
            if priority > _QUEUE_PRIORITY.get(states.get(key, ""), 0):
                states[key] = state
    return states


@router.get("/records/index-status")
def record_index_status(camera_key: str | None = None) -> dict[str, object]:
    """Per-segment indexing state: indexed / processing / queued / failed / unknown."""

    query = """SELECT c.camera_key, r.media_uri, r.status,
                      (SELECT count(*) FROM detections d WHERE d.recording_id = r.recording_id),
                      (SELECT count(*) FROM reid_embeddings_2048 e
                         JOIN detections d ON d.detection_id = e.detection_id
                        WHERE d.recording_id = r.recording_id)
               FROM recordings r
               JOIN cameras c ON c.camera_id = r.camera_id"""
    params: list[object] = []
    if camera_key:
        query += " WHERE c.camera_key = %s"
        params.append(camera_key)
    query += " ORDER BY r.started_at DESC NULLS LAST LIMIT 500"
    with psycopg.connect(_dsn()) as connection:
        rows = connection.execute(query, params).fetchall()

    queue = _queue_segment_states()
    segments: dict[str, dict[str, object]] = {}
    for camera, media_uri, status, detections, embeddings in rows:
        name = Path(media_uri).name
        if status == "processing":
            state = "processing"
        elif int(embeddings) > 0 or int(detections) > 0 or status == "complete":
            state = "indexed"
        else:
            state = queue.get((camera, name), "unknown")
        segments[name] = {
            "camera_key": camera,
            "status": state,
            "detections": int(detections),
            "embeddings": int(embeddings),
        }

    for (camera, name), state in queue.items():
        if camera_key and camera != camera_key:
            continue
        if name in segments:
            continue
        segments[name] = {
            "camera_key": camera,
            "status": state,
            "detections": 0,
            "embeddings": 0,
        }

    return {"segments": segments}


@router.get("/records/detections")
def list_segment_detections(camera: str, segment: str) -> dict[str, object]:
    with psycopg.connect(_dsn()) as connection:
        row = connection.execute(
            """SELECT r.recording_id, r.started_at, r.media_uri
               FROM recordings r
               JOIN cameras c ON c.camera_id = r.camera_id
               WHERE c.camera_key = %s AND r.media_uri LIKE %s
               ORDER BY r.recording_id DESC LIMIT 1""",
            (camera, f"%{segment}"),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail=f"recording not indexed: {camera}/{segment}")
        recording_id, started_at, media_uri = int(row[0]), row[1], row[2]
        detections = connection.execute(
            """SELECT d.detection_id, t.track_id, t.local_track_key, d.class_name,
                      d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                      d.detector_score, d.quality_score,
                      EXTRACT(EPOCH FROM (d.observed_at - r.started_at)) AS offset_s,
                      EXISTS (SELECT 1 FROM reid_embeddings_2048 e WHERE e.detection_id = d.detection_id)
               FROM detections d
               JOIN recordings r ON r.recording_id = d.recording_id
               LEFT JOIN track_observations o ON o.detection_id = d.detection_id
               LEFT JOIN tracks t ON t.track_id = o.track_id
               WHERE d.recording_id = %s
               ORDER BY offset_s""",
            (recording_id,),
        ).fetchall()

    resolved = resolve_media_path(media_uri)
    probe = _probe_video(resolved)
    width, height, fps = (probe[0], probe[1], probe[2]) if probe else (0, 0, 0.0)
    duration_s = probe[3] / fps if probe and fps and probe[3] else None
    return {
        "recording_id": recording_id,
        "camera_key": camera,
        "segment": Path(media_uri).name,
        "started_at": started_at,
        "video": {"width": width, "height": height, "fps": fps, "duration_seconds": duration_s},
        "detections": [
            {
                "detection_id": int(item[0]),
                "track_db_id": int(item[1]) if item[1] is not None else None,
                "track_label": item[2],
                "class_name": item[3],
                "bbox": {"x1": float(item[4]), "y1": float(item[5]), "x2": float(item[6]), "y2": float(item[7])},
                "detector_score": float(item[8]) if item[8] is not None else None,
                "quality_score": float(item[9]) if item[9] is not None else None,
                "offset_seconds": float(item[10]) if item[10] is not None else None,
                "has_embedding": bool(item[11]),
            }
            for item in detections
        ],
    }


@router.get("/detections/{detection_id}/crop")
def detection_crop(detection_id: int) -> Response:
    cache_file = _CROP_DIR / f"{detection_id}.jpg"
    if cache_file.is_file():
        return Response(content=cache_file.read_bytes(), media_type="image/jpeg")
    with psycopg.connect(_dsn()) as connection:
        row = connection.execute(
            """SELECT d.bbox_x1, d.bbox_y1, d.bbox_x2, d.bbox_y2,
                      d.observed_at, r.started_at, r.media_uri
               FROM detections d
               JOIN recordings r ON r.recording_id = d.recording_id
               WHERE d.detection_id = %s""",
            (detection_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="detection not found")
    x1, y1, x2, y2 = (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
    observed_at: datetime = row[4]
    started_at: datetime = row[5]
    resolved = resolve_media_path(row[6])
    probe = _probe_video(resolved)
    if probe is None:
        raise HTTPException(status_code=404, detail=f"media file missing: {row[6]}")
    width, height, fps, _frame_count = probe
    offset = max(0.0, (observed_at - started_at).total_seconds())

    import cv2

    capture = cv2.VideoCapture(resolved)
    if not capture.isOpened():
        capture.release()
        raise HTTPException(status_code=404, detail=f"cannot open video: {row[6]}")
    capture.set(cv2.CAP_PROP_POS_MSEC, offset * 1000.0)
    ok, frame = capture.read()
    if not ok and fps > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, min(round(offset * fps), max(probe[3] - 1, 0)))
        ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise HTTPException(status_code=404, detail=f"cannot seek video to {offset:.2f}s")

    frame_h, frame_w = frame.shape[:2]
    pad_x = (x2 - x1) * 0.05
    pad_y = (y2 - y1) * 0.05
    left = max(0, int(x1 - pad_x))
    top = max(0, int(y1 - pad_y))
    right = min(frame_w, int(math.ceil(x2 + pad_x)))
    bottom = min(frame_h, int(math.ceil(y2 + pad_y)))
    if right <= left or bottom <= top:
        raise HTTPException(status_code=422, detail="degenerate bounding box")
    ok, encoded = cv2.imencode(".jpg", frame[top:bottom, left:right], [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise HTTPException(status_code=500, detail="crop encoding failed")
    payload = encoded.tobytes()
    _CROP_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(payload)
    return Response(
        content=payload,
        media_type="image/jpeg",
        headers={"Cache-Control": "private, max-age=31536000, immutable"},
    )


@router.post("/record-search/track")
def track_search(request: TrackSearchRequest) -> dict[str, object]:
    with psycopg.connect(_dsn()) as connection:
        track_id = request.track_db_id
        if track_id is None:
            linked = connection.execute(
                """SELECT o.track_id FROM track_observations o
                   WHERE o.detection_id = %s LIMIT 1""",
                (request.detection_id,),
            ).fetchone()
            if linked is None:
                raise HTTPException(status_code=404, detail="detection has no track")
            track_id = int(linked[0])
        track_row = connection.execute(
            """SELECT t.local_track_key, t.class_name FROM tracks t WHERE t.track_id = %s""",
            (track_id,),
        ).fetchone()
        if track_row is None:
            raise HTTPException(status_code=404, detail="track not found")
        track_label, class_name = track_row[0], track_row[1]
        if request.segment is None:
            segment_row = connection.execute(
                """SELECT r.media_uri FROM tracks t
                   JOIN recordings r ON r.recording_id = t.recording_id
                   WHERE t.track_id = %s""",
                (track_id,),
            ).fetchone()
            if segment_row is None:
                raise HTTPException(status_code=404, detail="track has no recording")
            request = request.model_copy(update={"segment": Path(segment_row[0]).name})
        embeddings = connection.execute(
            """SELECT e.embedding::text, e.model_key, e.quality_score
               FROM reid_embeddings_2048 e
               WHERE e.track_id = %s AND e.embedding IS NOT NULL
               ORDER BY e.quality_score DESC NULLS LAST
               LIMIT 8""",
            (track_id,),
        ).fetchall()

    if not embeddings:
        raise HTTPException(status_code=404, detail="track has no stored embeddings")

    vector_space = _VECTOR_SPACE_BY_MODEL.get(embeddings[0][1])
    if vector_space is None:
        raise HTTPException(status_code=503, detail=f"unsupported model_key: {embeddings[0][1]}")
    dim = len(_parse_vector(embeddings[0][0]))
    sums = [0.0] * dim
    for text, _model_key, _quality in embeddings:
        for i, value in enumerate(_parse_vector(text)):
            sums[i] += value
    norm = math.sqrt(sum(value * value for value in sums)) or 1.0
    centroid = Embedding(vector_space, tuple(value / norm for value in sums), "track-centroid")

    min_similarity = request.min_similarity
    if min_similarity is None:
        min_similarity = float(os.getenv("REID_MIN_SIMILARITY", "0.628"))
    candidates = PostgresVectorIndex().search(
        centroid, top_k=request.top_k, exclude_track_db_id=track_id
    )
    matches = [item for item in candidates if item.score >= min_similarity and item.segment_path]

    started_by_uri: dict[str, datetime] = {}
    if matches:
        uris = sorted({item.segment_path for item in matches})
        with psycopg.connect(_dsn()) as connection:
            rows = connection.execute(
                "SELECT media_uri, started_at FROM recordings WHERE media_uri = ANY(%s)",
                (uris,),
            ).fetchall()
        started_by_uri = {row[0]: row[1] for row in rows if row[1] is not None}

    serialized = []
    for item in matches:
        start = started_by_uri.get(item.segment_path)
        offset = None
        if start is not None and item.observed_at is not None:
            offset = (item.observed_at - start).total_seconds()
        serialized.append(
            {
                "camera_key": item.camera_key,
                "segment": Path(item.segment_path).name,
                "track_db_id": item.track_db_id,
                "track_label": item.track_id,
                "class_name": item.object_type.value if item.object_type else class_name,
                "score": round(item.score, 4),
                "detection_id": item.detection_id,
                "observed_at": item.observed_at,
                "offset_seconds": round(offset, 3) if offset is not None else None,
            }
        )

    return {
        "query": {
            "camera_key": request.camera_key,
            "segment": request.segment,
            "track_db_id": track_id,
            "track_label": track_label,
            "class_name": class_name,
            "embedding_samples": len(embeddings),
        },
        "min_similarity": min_similarity,
        "matches": serialized,
    }


def _parse_vector(text: str) -> list[float]:
    return [float(part) for part in text.strip().strip("[]").split(",") if part]
