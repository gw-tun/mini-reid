"""A durable file-backed queue used by the API and worker scaffold."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .events import RecordSegmentEvent


@dataclass(frozen=True)
class QueueReceipt:
    job_id: str
    duplicate: bool


@dataclass(frozen=True)
class ClaimedJob:
    job_id: str
    event: RecordSegmentEvent
    processing_path: Path


class JobQueue(Protocol):
    def enqueue(self, event: RecordSegmentEvent) -> QueueReceipt: ...

    def claim(self, limit: int = 1) -> list[ClaimedJob]: ...

    def acknowledge(self, job: ClaimedJob) -> None: ...

    def fail(self, job: ClaimedJob, error: str) -> None: ...


class FileJobQueue:
    """Cross-process queue backed by atomic file moves on the shared volume."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.pending = self.root / "pending"
        self.processing = self.root / "processing"
        self.failed = self.root / "failed"
        for directory in (self.pending, self.processing, self.failed):
            directory.mkdir(parents=True, exist_ok=True)

    def enqueue(self, event: RecordSegmentEvent) -> QueueReceipt:
        filename = f"{event.event_id}.json"
        destination = self.pending / filename
        if any((directory / filename).exists() for directory in (self.pending, self.processing, self.failed)):
            return QueueReceipt(job_id=event.event_id, duplicate=True)

        temporary = self.pending / f".{filename}.{os.getpid()}.tmp"
        payload = {"job_id": event.event_id, "event": event.model_dump(mode="json")}
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, destination)
        return QueueReceipt(job_id=event.event_id, duplicate=False)

    def claim(self, limit: int = 1) -> list[ClaimedJob]:
        claimed: list[ClaimedJob] = []
        for pending_path in sorted(self.pending.glob("*.json")):
            if len(claimed) >= limit:
                break
            processing_path = self.processing / pending_path.name
            try:
                os.replace(pending_path, processing_path)
            except FileNotFoundError:
                continue
            try:
                raw = json.loads(processing_path.read_text(encoding="utf-8"))
                event = RecordSegmentEvent.model_validate(raw["event"])
            except Exception as exc:
                self._move_to_failed(processing_path, f"invalid queue payload: {exc}")
                continue
            claimed.append(
                ClaimedJob(
                    job_id=str(raw["job_id"]),
                    event=event,
                    processing_path=processing_path,
                )
            )
        return claimed

    def acknowledge(self, job: ClaimedJob) -> None:
        job.processing_path.unlink(missing_ok=True)

    def fail(self, job: ClaimedJob, error: str) -> None:
        self._move_to_failed(job.processing_path, error)

    def _move_to_failed(self, processing_path: Path, error: str) -> None:
        failed_path = self.failed / processing_path.name
        try:
            raw = json.loads(processing_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {"job_id": processing_path.stem}
        raw["last_error"] = error
        raw["failed_at"] = time.time()
        temporary = self.failed / f".{failed_path.name}.{os.getpid()}.tmp"
        temporary.write_text(json.dumps(raw, sort_keys=True), encoding="utf-8")
        os.replace(temporary, failed_path)
        processing_path.unlink(missing_ok=True)
