"""Rank person tracks by ReID similarity to one track, in PostgreSQL.

Exact search: each track's barcode is the mean of its best crops' barcodes,
compared by cosine with every other person track of the SAME model AND
version -- a barcode is never compared with another model's, or with an older
checkpoint's. Fine for small galleries; HNSW comes in when galleries grow (M4).

One query for every caller: the API's /person/tracks/{id}/similar and the
recording search (application/recording_search.py).
"""

from __future__ import annotations

from psycopg.rows import tuple_row

from services.common.database import EMBEDDING_TABLES

TRACK_COLUMNS = """
    t.track_id, c.camera_key, t.local_track_key, t.started_at, t.ended_at, t.recording_id,
    r.media_uri, r.started_at AS recording_started_at,
    round(EXTRACT(EPOCH FROM t.ended_at - t.started_at)::numeric, 1)::float AS seconds,
    (SELECT count(*) FROM track_observations o WHERE o.track_id = t.track_id)::int AS observations,
    t.best_crop_uri, t.best_quality_score AS best_quality,
    ARRAY(SELECT model_key FROM reid_embeddings_2048 e WHERE e.track_id = t.track_id
          UNION SELECT model_key FROM reid_embeddings_512 e WHERE e.track_id = t.track_id) AS models
"""
"""What a caller gets per track; needs `tracks t`, `cameras c` and `recordings r` joined."""

TRACK_JOINS = ("tracks t JOIN cameras c ON c.camera_id = t.camera_id"
               " LEFT JOIN recordings r ON r.recording_id = t.recording_id")


def current_model(conn, model_key: str) -> dict | None:
    """The newest version of a person ReID model: {model_version, feature_dim}."""
    cur = conn.cursor(row_factory=tuple_row)  # whatever rows the caller's connection makes
    cur.execute("SELECT model_version, feature_dim FROM reid_models WHERE model_key = %s"
                " AND object_class = 'person' ORDER BY created_at DESC LIMIT 1", (model_key,))
    row = cur.fetchone()
    return None if row is None else {"model_version": row[0], "feature_dim": row[1]}


def rank_similar(conn, track_id: int, model_key: str, *, top_k: int = 10,
                 media_prefix: str | None = None) -> list[dict]:
    """The `top_k` person tracks most like `track_id`, best first, each with
    TRACK_COLUMNS plus `similarity` (cosine, one model's scale) and `same_time`
    (same camera, overlapping in time: certainly someone else).
    `media_prefix`: only tracks from recordings whose media_uri starts with it.
    Empty when the model is unknown or the track has no barcode from it."""
    model = current_model(conn, model_key)
    if model is None:
        return []
    table = EMBEDDING_TABLES[model["feature_dim"]][0]
    cur = conn.cursor(row_factory=tuple_row)  # whatever rows the caller's connection makes
    cur.execute(f"""
        WITH q AS (SELECT avg(embedding) AS v FROM {table}
                   WHERE track_id = %(track)s AND model_key = %(model)s AND model_version = %(version)s),
             src AS (SELECT camera_id, started_at, ended_at FROM tracks WHERE track_id = %(track)s),
             cand AS (SELECT e.track_id, avg(e.embedding) AS v FROM {table} e
                      JOIN tracks ct ON ct.track_id = e.track_id
                      LEFT JOIN recordings cr ON cr.recording_id = ct.recording_id
                      WHERE e.model_key = %(model)s AND e.model_version = %(version)s
                        AND e.class_name = 'person' AND e.track_id <> %(track)s
                        AND (%(prefix)s::text IS NULL OR cr.media_uri LIKE %(prefix)s::text || '%%')
                      GROUP BY e.track_id)
        SELECT {TRACK_COLUMNS}, 1 - (cand.v <=> q.v) AS similarity,
               COALESCE(t.camera_id = src.camera_id AND t.started_at <= src.ended_at
                        AND src.started_at <= t.ended_at, FALSE) AS same_time
        FROM cand CROSS JOIN q CROSS JOIN src
        JOIN {TRACK_JOINS} ON t.track_id = cand.track_id
        WHERE q.v IS NOT NULL
        ORDER BY similarity DESC LIMIT %(k)s""",
        {"track": track_id, "model": model_key, "version": model["model_version"],
         "prefix": media_prefix, "k": top_k})
    names = [d.name for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]
