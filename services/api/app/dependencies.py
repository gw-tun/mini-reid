from __future__ import annotations

import os
from pathlib import Path

from services.common.job_queue import FileJobQueue, JobQueue

_queue = FileJobQueue(
    os.getenv("JOB_QUEUE_DIR", str(Path("data") / "cache" / "job-queue"))
)


def get_job_queue() -> JobQueue:
    """Dependency seam for replacing the file queue with PostgreSQL/Redis."""

    return _queue
