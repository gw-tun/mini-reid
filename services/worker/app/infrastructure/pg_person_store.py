"""Person indexing results -> PostgreSQL (infra/postgres/init/002 and 003).

DRAFT, like those tables: until they are agreed with VIP-8 this is used by
scripts/index_clip.py, not by the worker's RecordIndexingService, whose ports
have no place for detections or tracks yet (docs/VIP-9-PERSON-PLAN.md §4).

Indexing the same recording again is safe. Every row has a natural key --
`ingest_key` for a detection, (camera, source, local track id, start) for a
track, `sample_key` for an embedding -- so a second run updates or skips rows
instead of doubling them.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Sequence

from services.common.database import EMBEDDING_TABLES

from ..reid.card import ReIDModelCard


def vector_literal(values: Iterable[float]) -> str:
    """pgvector's text form: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{v:.6g}" for v in values) + "]"


class PgPersonStore:
    def __init__(self, conn) -> None:
        """`conn`: an open psycopg connection; the caller commits."""
        self.conn = conn

    def camera_id(self, camera_key: str) -> int:
        return self.conn.execute(
            "INSERT INTO cameras (camera_key, display_name, mediamtx_path) VALUES (%s, %s, %s) "
            "ON CONFLICT (camera_key) DO UPDATE SET updated_at = now() RETURNING camera_id",
            (camera_key, camera_key, camera_key)).fetchone()[0]

    def recording_id(self, camera_id: int, media_uri: str, started_at: dt.datetime,
                     ended_at: dt.datetime, file_size_bytes: int | None) -> int:
        """The recording's row. `ended_at` only ever grows: indexing a stretch in
        the middle of a file must not shorten what an earlier run recorded."""
        return self.conn.execute(
            "INSERT INTO recordings (camera_id, media_uri, started_at, ended_at, duration_ms,"
            " media_format, status, file_size_bytes) VALUES (%s, %s, %s, %s, %s, 'mp4', 'complete', %s) "
            "ON CONFLICT (camera_id, media_uri) DO UPDATE SET"
            " ended_at = GREATEST(recordings.ended_at, EXCLUDED.ended_at),"
            " duration_ms = GREATEST(recordings.duration_ms, EXCLUDED.duration_ms), status = 'complete' "
            "RETURNING recording_id",
            (camera_id, media_uri, started_at, ended_at,
             int((ended_at - started_at).total_seconds() * 1000), file_size_bytes)).fetchone()[0]

    def model_id(self, card: ReIDModelCard) -> int:
        return self.conn.execute(
            "INSERT INTO reid_models (model_key, backend, object_class, model_version, weights_uri,"
            " feature_dim, input_height, input_width, distance_metric) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'cosine') "
            "ON CONFLICT (model_key, model_version) DO UPDATE SET active = TRUE RETURNING model_id",
            (card.model_key, card.backend, card.object_class.value, card.sha256, str(card.weights),
             card.feature_dim, card.input_height, card.input_width)).fetchone()[0]

    def add_detections(self, camera_id: int, recording_id: int, detector_key: str,
                       detector_version: str, rows: Sequence[dict]) -> dict[str, int]:
        """Every person box, kept or rejected -> {ingest_key: detection_id}."""
        ids = {}
        for r in rows:
            x1, y1, x2, y2 = r["bbox"]
            ids[r["ingest_key"]] = self.conn.execute(
                "INSERT INTO detections (camera_id, recording_id, source_kind, observed_at, class_name,"
                " detector_key, detector_version, detector_score, bbox_x1, bbox_y1, bbox_x2, bbox_y2,"
                " crop_uri, crop_width, crop_height, quality_score, payload, ingest_key) "
                "VALUES (%s, %s, 'recording', %s, 'person', %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (ingest_key) DO UPDATE SET crop_uri = COALESCE(EXCLUDED.crop_uri, detections.crop_uri) "
                "RETURNING detection_id",
                (camera_id, recording_id, r["observed_at"], detector_key, detector_version, r["score"],
                 x1, y1, x2, y2, r.get("crop_uri"), r.get("crop_width"), r.get("crop_height"),
                 r.get("quality"), json.dumps(r.get("payload") or {}), r["ingest_key"])).fetchone()[0]
        return ids

    def add_track(self, camera_id: int, recording_id: int, local_track_key: str,
                  started_at: dt.datetime, ended_at: dt.datetime, best_crops: Sequence[dict],
                  summary: dict) -> int:
        best = best_crops[0] if best_crops else {}
        return self.conn.execute(
            "INSERT INTO tracks (camera_id, recording_id, source_kind, class_name, local_track_key,"
            " started_at, ended_at, status, best_crop_uri, best_quality_score, quality_summary) "
            "VALUES (%s, %s, 'recording', 'person', %s, %s, %s, 'ended', %s, %s, %s) "
            "ON CONFLICT (camera_id, source_kind, local_track_key, started_at) DO UPDATE SET"
            " ended_at = EXCLUDED.ended_at, status = 'ended', best_crop_uri = EXCLUDED.best_crop_uri,"
            " best_quality_score = EXCLUDED.best_quality_score, quality_summary = EXCLUDED.quality_summary "
            "RETURNING track_id",
            (camera_id, recording_id, local_track_key, started_at, ended_at, best.get("uri"),
             best.get("quality"), json.dumps(summary | {"best_crops": list(best_crops)}))).fetchone()[0]

    def add_observations(self, track_id: int, rows: Sequence[dict]) -> None:
        for r in rows:
            x1, y1, x2, y2 = r["bbox"]
            self.conn.execute(
                "INSERT INTO track_observations (track_id, detection_id, observed_at, bbox_x1, bbox_y1,"
                " bbox_x2, bbox_y2, detector_score, quality_score) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (track_id, observed_at) DO NOTHING",
                (track_id, r.get("detection_id"), r["observed_at"], x1, y1, x2, y2, r["score"], r.get("quality")))

    def add_embeddings(self, card: ReIDModelCard, model_id: int, rows: Sequence[dict]) -> int:
        """One row per best crop and model -> how many were new."""
        table, kind = EMBEDDING_TABLES[card.feature_dim]
        new = 0
        for r in rows:
            cur = self.conn.execute(
                f"INSERT INTO {table} (track_id, detection_id, camera_id, class_name, model_id, model_key,"
                f" model_version, observed_at, quality_score, embedding, sample_key) "
                f"VALUES (%s, %s, %s, 'person', %s, %s, %s, %s, %s, %s::{kind}, %s) "
                f"ON CONFLICT (sample_key) DO NOTHING",
                (r["track_id"], r.get("detection_id"), r["camera_id"], model_id, card.model_key, card.sha256,
                 r["observed_at"], r["quality"], vector_literal(r["vector"]), r["sample_key"]))
            new += cur.rowcount
        return new
