"""The recording search engine inside the API process, built on first use.

Heavy -- YOLO and two ReID models, a few seconds to load and about a gigabyte
of memory -- so it is built when the first "Search similar" arrives, not at
startup: listing recordings and the person pages never import torch.

Searches run on ONE background thread, one at a time (the engine also holds a
lock, because MPS crashes on concurrent use); at most MAX_WAITING wait. Their
state lives here, in memory: a search outlives neither a restart nor the last
MAX_KEPT searches.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger("search")

MAX_WAITING = 3
MAX_KEPT = 50


class Busy(RuntimeError):
    pass


class SearchRunner:
    def __init__(self) -> None:
        self._engine = None
        self._build_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="recording-search")
        self._lock = threading.Lock()
        self.searches: dict[str, dict] = {}

    def engine(self):
        with self._build_lock:
            if self._engine is None:
                self._engine = self._build()
            return self._engine

    @staticmethod
    def _build():
        import psycopg

        from services.common.database import resolve_dsn
        from services.worker.app.application.person_indexer import PersonIndexer
        from services.worker.app.application.recording_search import RecordingSearch
        from services.worker.app.detection.yolo import YoloDetector
        from services.worker.app.domain.models import ObjectType
        from services.worker.app.reid.fastreid import FastReIDPersonReID
        from services.worker.app.reid.osnet import OsnetAinReID

        from .dependencies import crops_dir, data_dir, downloads_dir

        detector = YoloDetector(object_types={ObjectType.PERSON})  # device: $DEVICE, else auto
        models = [FastReIDPersonReID(), OsnetAinReID()]
        log.info("recording search: %s and %s on %s", *(m.model_key for m in models), detector.device)
        dsn = resolve_dsn()
        return RecordingSearch(PersonIndexer(detector, models, crops_dir()), lambda: psycopg.connect(dsn),
                               downloads_dir(), data_dir())

    def submit(self, track_id: int, model_key: str, top_k: int) -> str:
        with self._lock:
            if sum(s["status"] in ("queued", "running") for s in self.searches.values()) >= MAX_WAITING:
                raise Busy(f"{MAX_WAITING} searches are already waiting; try again shortly")
            search_id = uuid.uuid4().hex[:12]
            self.searches[search_id] = {"search_id": search_id, "status": "queued", "stage": None, "done": 0.0,
                                        "total": 0.0, "error": None, "model_key": model_key,
                                        "query_track_id": track_id, "results": []}
            for old in list(self.searches)[:-MAX_KEPT]:
                self.searches.pop(old)
        state = self.searches[search_id]

        def report(stage: str, done: float, total: float) -> None:
            state.update(stage=stage, done=done, total=total)

        def job() -> None:
            state["status"] = "running"
            try:
                state["results"] = self.engine().search(track_id, model_key, top_k, report)
                state["status"] = "complete"
            except Exception as exc:
                log.exception("search %s failed", search_id)
                state.update(status="failed", error=str(exc)[:500])

        self._pool.submit(job)
        return search_id


runner = SearchRunner()
