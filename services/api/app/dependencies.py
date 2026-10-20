from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

from fastapi import HTTPException

from services.common.database import ROOT, resolve_dsn
from services.common.job_queue import FileJobQueue, JobQueue

_queue = FileJobQueue(
    os.getenv("JOB_QUEUE_DIR", str(Path("data") / "cache" / "job-queue"))
)


def get_job_queue() -> JobQueue:
    """Dependency seam for replacing the file queue with PostgreSQL/Redis."""

    return _queue


def get_db() -> Iterator:
    """One connection per request, rows as dicts. Enough for local development;
    a connection pool can replace it behind this same seam."""

    import psycopg
    from psycopg.rows import dict_row

    try:
        conn = psycopg.connect(resolve_dsn(), row_factory=dict_row, connect_timeout=5)
    except psycopg.OperationalError as exc:
        raise HTTPException(status_code=503, detail="database unavailable") from exc
    with conn:
        yield conn


def data_dir() -> Path:
    """mini-reid's data folder: $DATA_DIR (docker: /data), else data/ in the repository."""

    return Path(os.getenv("DATA_DIR", str(ROOT / "data")))


def crops_dir() -> Path:
    """Where the indexer writes evidence crops."""

    return data_dir() / "crops"


def downloads_dir() -> Path:
    """Where scripts/fetch_search_clips.py puts semantic-va search results."""

    return Path(os.getenv("SEMANTIC_VA_DOWNLOADS", str(data_dir() / "recordings" / "semantic-va")))
