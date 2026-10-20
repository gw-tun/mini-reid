"""Person tracks and "find similar person" (VIP-9), read from PostgreSQL.

Reads what the indexer writes (draft tables 002/003). The ranking itself is
services/worker/app/infrastructure/pg_person_search.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from services.common.database import EMBEDDING_TABLES
from services.worker.app.infrastructure.pg_person_search import current_model, rank_similar

from ..dependencies import crops_dir, get_db
from ..schemas import (PersonCrop, PersonTrack, PersonTrackDetail, ReIDModelInfo, SimilarTrack,
                       SimilarTracksResponse)

router = APIRouter(prefix="/person", tags=["person"])
crops_router = APIRouter(tags=["evidence"])

_TRACK = """
    t.track_id, c.camera_key, t.local_track_key, t.started_at, t.ended_at,
    round(EXTRACT(EPOCH FROM t.ended_at - t.started_at)::numeric, 1)::float AS seconds,
    (SELECT count(*) FROM track_observations o WHERE o.track_id = t.track_id)::int AS observations,
    t.best_crop_uri, t.best_quality_score AS best_quality,
    ARRAY(SELECT model_key FROM reid_embeddings_2048 e WHERE e.track_id = t.track_id
          UNION SELECT model_key FROM reid_embeddings_512 e WHERE e.track_id = t.track_id) AS models
"""


def _track(row: dict) -> PersonTrack:
    return PersonTrack(**{k: row[k] for k in PersonTrack.model_fields})


@router.get("/models", response_model=list[ReIDModelInfo])
def models(conn=Depends(get_db)) -> list[ReIDModelInfo]:
    rows = conn.execute(
        "SELECT DISTINCT ON (model_key) model_key, backend, feature_dim, model_version"
        " FROM reid_models WHERE object_class = 'person' AND active"
        " ORDER BY model_key, created_at DESC").fetchall()
    out = []
    for r in rows:
        table = EMBEDDING_TABLES[r["feature_dim"]][0]
        tracks = conn.execute(f"SELECT count(DISTINCT track_id) AS n FROM {table}"
                              " WHERE model_key = %s AND model_version = %s",
                              (r["model_key"], r["model_version"])).fetchone()["n"]
        out.append(ReIDModelInfo(model_key=r["model_key"], backend=r["backend"],
                                 feature_dim=r["feature_dim"], tracks=tracks))
    return out


@router.get("/tracks", response_model=list[PersonTrack])
def tracks(camera_key: str | None = None, limit: int = Query(100, ge=1, le=500),
           conn=Depends(get_db)) -> list[PersonTrack]:
    rows = conn.execute(
        f"SELECT {_TRACK} FROM tracks t JOIN cameras c ON c.camera_id = t.camera_id"
        " WHERE t.class_name = 'person' AND (%(camera)s::text IS NULL OR c.camera_key = %(camera)s)"
        " ORDER BY t.started_at DESC, t.track_id LIMIT %(limit)s",
        {"camera": camera_key, "limit": limit}).fetchall()
    return [_track(r) for r in rows]


@router.get("/tracks/{track_id}", response_model=PersonTrackDetail)
def track(track_id: int, conn=Depends(get_db)) -> PersonTrackDetail:
    row = conn.execute(
        f"SELECT {_TRACK}, r.media_uri AS recording_uri, t.quality_summary"
        " FROM tracks t JOIN cameras c ON c.camera_id = t.camera_id"
        " LEFT JOIN recordings r ON r.recording_id = t.recording_id"
        " WHERE t.track_id = %s AND t.class_name = 'person'", (track_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"no person track {track_id}")
    crops = [PersonCrop(uri=c["uri"], quality=c["quality"], observed_at=c["observed_at"])
             for c in (row["quality_summary"] or {}).get("best_crops", [])]
    return PersonTrackDetail(**_track(row).model_dump(), recording_uri=row["recording_uri"], crops=crops)


@router.get("/tracks/{track_id}/similar", response_model=SimilarTracksResponse)
def similar(track_id: int, model_key: str, top_k: int = Query(10, ge=1, le=100),
            conn=Depends(get_db)) -> SimilarTracksResponse:
    model = current_model(conn, model_key)
    if model is None:
        raise HTTPException(status_code=404, detail=f"no person ReID model {model_key!r}")
    if conn.execute("SELECT 1 FROM tracks WHERE track_id = %s AND class_name = 'person'",
                    (track_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"no person track {track_id}")
    rows = rank_similar(conn, track_id, model_key, top_k=top_k)
    return SimilarTracksResponse(
        model_key=model_key, model_version=model["model_version"], query_track_id=track_id,
        results=[SimilarTrack(track=_track(r), similarity=round(r["similarity"], 4), same_time=r["same_time"])
                 for r in rows])


@crops_router.get("/crops/{path:path}")
def crop(path: str) -> FileResponse:
    """An evidence crop. Only .jpg files inside the crops folder are served."""
    root = crops_dir().resolve()
    file = (root / path).resolve()
    if not file.is_relative_to(root) or file.suffix.lower() != ".jpg" or not file.is_file():
        raise HTTPException(status_code=404, detail="no such crop")
    return FileResponse(file, media_type="image/jpeg", headers={"Cache-Control": "public, max-age=3600"})
