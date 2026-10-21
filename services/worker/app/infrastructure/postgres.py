"""PostgreSQL/pgvector persistence adapter for the worker backbone."""

from __future__ import annotations

import hashlib
import os
from datetime import timedelta
from typing import Any

import psycopg

from ..domain.models import Embedding, MatchCandidate, ObjectType, TrackObservation


def _dsn(value: str | None = None) -> str:
    return (value or os.getenv(
        "DATABASE_URL",
        "postgresql://mini_reid:change_me_for_local_dev@localhost:5434/mini_reid",
    )).replace("postgresql+psycopg://", "postgresql://", 1)


def _vector_literal(values: tuple[float, ...]) -> str:
    return "[" + ",".join(f"{value:.10f}" for value in values) + "]"


def _model_key(object_type: ObjectType) -> str:
    return "msmt17_sbs_r50_ibn" if object_type is ObjectType.PERSON else "veri_sbs_r50_ibn"


def _model_key_for_vector_space(vector_space: str) -> str | None:
    if vector_space == "fastreid-msmt17-v1":
        return "msmt17_sbs_r50_ibn"
    if vector_space == "fastreid-veri-v1":
        return "veri_sbs_r50_ibn"
    return None


class PostgresVectorIndex:
    """CRUD adapter for the 2048-D FastReID gallery."""

    def __init__(self, database_url: str | None = None, detector_version: str = "yolo11m") -> None:
        self._database_url = _dsn(database_url)
        self._detector_version = detector_version

    def upsert(self, event, observation: TrackObservation, embedding: Embedding) -> None:
        if len(embedding.values) != 2048:
            raise ValueError(
                "PostgresVectorIndex stores vector(2048); set REID_BACKEND=fastreid to embed with 2048-D models"
            )
        expected_space = "fastreid-msmt17-v1" if observation.object_type is ObjectType.PERSON else "fastreid-veri-v1"
        if embedding.vector_space != expected_space:
            raise ValueError(f"embedding vector space {embedding.vector_space!r} does not match {expected_space!r}")
        model_key = _model_key(observation.object_type)
        vector = _vector_literal(embedding.values)
        # Must match OpenCVSegmentReader's time base: frame 0 == completed_at - duration.
        started_at = event.completed_at - timedelta(seconds=event.duration_seconds)
        ingest_key = hashlib.sha256(
            f"{event.event_id}:{observation.track_id}:{observation.observed_at.isoformat()}".encode()
        ).hexdigest()
        sample_key = f"{event.event_id}:{observation.track_id}:{observation.observed_at.isoformat()}:{embedding.vector_space}"

        with psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """INSERT INTO cameras (camera_key, display_name, mediamtx_path)
                       VALUES (%s, %s, %s)
                       ON CONFLICT (camera_key) DO UPDATE SET updated_at = now()
                       RETURNING camera_id""",
                    (observation.camera_key, observation.camera_key, observation.camera_key),
                )
                camera_id = cursor.fetchone()[0]
                cursor.execute(
                    """INSERT INTO recordings
                       (camera_id, media_uri, started_at, ended_at, duration_ms, status)
                       VALUES (%s, %s, %s, %s, %s, 'processing')
                       ON CONFLICT (camera_id, media_uri) DO UPDATE SET
                         ended_at = EXCLUDED.ended_at,
                         duration_ms = EXCLUDED.duration_ms,
                         status = 'processing'
                       RETURNING recording_id""",
                    (camera_id, event.segment_path, started_at, event.completed_at, int(event.duration_seconds * 1000)),
                )
                recording_id = cursor.fetchone()[0]
                cursor.execute(
                    """INSERT INTO reid_models
                       (model_key, backend, object_class, model_version, weights_uri,
                        config_uri, feature_dim, input_height, input_width, distance_metric)
                       VALUES (%s, 'fastreid', %s, %s, %s, NULL, 2048, 256, 256, 'cosine')
                       ON CONFLICT (model_key, model_version) DO UPDATE SET active = TRUE
                       RETURNING model_id""",
                    (model_key, observation.object_type.value, embedding.model_version,
                     os.getenv("REID_WEIGHTS_URI", "models/fastreid")),
                )
                model_id = cursor.fetchone()[0]
                cursor.execute(
                    """SELECT track_id FROM tracks
                       WHERE camera_id = %s AND recording_id = %s
                         AND local_track_key = %s AND class_name = %s
                       ORDER BY started_at LIMIT 1""",
                    (camera_id, recording_id, observation.track_id, observation.object_type.value),
                )
                row = cursor.fetchone()
                if row:
                    track_id = row[0]
                else:
                    cursor.execute(
                        """INSERT INTO tracks
                           (camera_id, recording_id, source_kind, class_name, local_track_key, started_at)
                           VALUES (%s, %s, 'recording', %s, %s, %s)
                           RETURNING track_id""",
                        (camera_id, recording_id, observation.object_type.value, observation.track_id, observation.observed_at),
                    )
                    track_id = cursor.fetchone()[0]
                quality = max(0.0, min(1.0, float(observation.confidence)))
                cursor.execute(
                    """INSERT INTO detections
                       (camera_id, recording_id, source_kind, observed_at, class_name,
                        detector_key, detector_version, detector_score,
                        bbox_x1, bbox_y1, bbox_x2, bbox_y2, quality_score, ingest_key)
                       VALUES (%s, %s, 'recording', %s, %s, 'yolo', %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (ingest_key) DO UPDATE SET quality_score = EXCLUDED.quality_score
                       RETURNING detection_id""",
                    (camera_id, recording_id, observation.observed_at, observation.object_type.value,
                     self._detector_version, quality, *observation.bbox, quality, ingest_key),
                )
                detection_id = cursor.fetchone()[0]
                cursor.execute(
                    """INSERT INTO track_observations
                       (track_id, detection_id, observed_at, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                        detector_score, quality_score)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (track_id, observed_at) DO UPDATE SET
                         detection_id = EXCLUDED.detection_id,
                         quality_score = EXCLUDED.quality_score""",
                    (track_id, detection_id, observation.observed_at, *observation.bbox, quality, quality),
                )
                cursor.execute(
                    """INSERT INTO reid_embeddings_2048
                       (track_id, detection_id, camera_id, class_name, model_id, model_key,
                        model_version, observed_at, quality_score, embedding, sample_key)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector, %s)
                       ON CONFLICT (sample_key) DO UPDATE SET
                         embedding = EXCLUDED.embedding,
                         quality_score = EXCLUDED.quality_score""",
                    (track_id, detection_id, camera_id, observation.object_type.value, model_id,
                     model_key, embedding.model_version, observation.observed_at, quality, vector, sample_key),
                )

    def search(
        self,
        embedding: Embedding,
        *,
        camera_key: str | None = None,
        object_type: str | None = None,
        top_k: int = 20,
        exclude_track_db_id: int | None = None,
    ) -> tuple[MatchCandidate, ...]:
        if len(embedding.values) != 2048:
            raise ValueError("PostgresVectorIndex searches vector(2048) embeddings only")
        clauses = ["e.embedding IS NOT NULL"]
        params: list[Any] = [_vector_literal(embedding.values)]
        model_key = _model_key_for_vector_space(embedding.vector_space)
        if model_key is None:
            raise ValueError(f"unsupported PostgreSQL vector space: {embedding.vector_space}")
        clauses.append("e.model_key = %s")
        params.append(model_key)
        if camera_key:
            clauses.append("c.camera_key = %s")
            params.append(camera_key)
        if object_type:
            clauses.append("e.class_name = %s")
            params.append(object_type)
        if exclude_track_db_id is not None:
            clauses.append("t.track_id <> %s")
            params.append(exclude_track_db_id)
        params.append(_vector_literal(embedding.values))
        params.append(max(1, min(top_k, 500)))
        query = f"""SELECT c.camera_key, t.local_track_key, t.track_id,
                          1 - (e.embedding <=> %s::vector) AS score,
                          r.media_uri, e.observed_at, e.class_name, e.detection_id
                   FROM reid_embeddings_2048 e
                   JOIN cameras c ON c.camera_id = e.camera_id
                   JOIN tracks t ON t.track_id = e.track_id
                   JOIN recordings r ON r.recording_id = t.recording_id
                   WHERE {' AND '.join(clauses)}
                   ORDER BY e.embedding <=> %s::vector
                   LIMIT %s"""
        with psycopg.connect(self._database_url) as connection:
            rows = connection.execute(query, params).fetchall()
        return tuple(
            MatchCandidate(
                camera_key=row[0], track_id=row[1], track_db_id=int(row[2]), score=float(row[3]),
                segment_path=row[4], observed_at=row[5], object_type=ObjectType(row[6]),
                detection_id=int(row[7]) if row[7] is not None else None,
            )
            for row in rows
        )


class PostgresRecordingStore:
    """Marks a recording complete after all segment vectors have been written."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = _dsn(database_url)

    def save_index_result(self, event, frames_seen: int, embeddings_written: int) -> None:
        with psycopg.connect(self._database_url) as connection:
            connection.execute(
                """UPDATE recordings r SET status = 'complete'
                   FROM cameras c
                   WHERE r.camera_id = c.camera_id
                     AND c.camera_key = %s AND r.media_uri = %s""",
                (event.camera_key, event.segment_path),
            )


class PostgresSearchResultStore:
    """Persists a completed image search and its ranked candidates."""

    def __init__(self, database_url: str | None = None) -> None:
        self._database_url = _dsn(database_url)

    def save_search(self, object_type: str, vector_space: str, top_k: int, candidates: tuple[MatchCandidate, ...]) -> None:
        model_key = _model_key_for_vector_space(vector_space)
        if model_key is None:
            raise ValueError(f"unsupported PostgreSQL vector space: {vector_space}")
        with psycopg.connect(self._database_url) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT model_id FROM reid_models WHERE model_key = %s ORDER BY active DESC, model_id DESC LIMIT 1",
                    (model_key,),
                )
                model = cursor.fetchone()
                if model is None:
                    raise RuntimeError(f"model registry entry not found: {model_key}")
                cursor.execute(
                    """INSERT INTO search_queries (query_kind, class_name, model_id, top_k, status, completed_at)
                       VALUES ('record_search', %s, %s, %s, 'complete', now())
                       RETURNING query_id""",
                    (object_type, model[0], top_k),
                )
                query_id = cursor.fetchone()[0]
                for rank, candidate in enumerate(candidates, start=1):
                    if candidate.track_db_id is None or candidate.observed_at is None:
                        continue
                    cursor.execute(
                        "SELECT camera_id FROM cameras WHERE camera_key = %s",
                        (candidate.camera_key,),
                    )
                    camera = cursor.fetchone()
                    if camera is None:
                        continue
                    cursor.execute(
                        """INSERT INTO match_results
                           (query_id, candidate_track_id, camera_id, observed_at,
                            raw_similarity, rank_position)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (query_id, candidate.track_db_id, camera[0], candidate.observed_at, candidate.score, rank),
                    )
