"""Search similar people across the downloaded recordings, processing them on demand.

What the mini-reid web app's "Search similar" button runs (through
services/api/app/api/recordings.py):

1. `people(path)`: process one recording if it hasn't been -- YOLO -> tracking
   -> best crops -> FastReID and OSNet barcodes -> pgvector -- and list the
   people in it.
2. `search(track_id, model_key)`: process every other recording that hasn't
   been, then rank all their people by similarity to the chosen one.

The recordings are what scripts/fetch_search_clips.py downloads:
<root>/<camera>/<YYYY-MM-DD_HH-MM-SS>.mp4, with manifest.jsonl and
searches.jsonl beside them. Processing is the same PersonIndexer that
scripts/index_clip.py uses, recorded the same way (media_uri relative to the
data folder), so a file processed by either is never processed again.
"""

from __future__ import annotations

import json
import pathlib
import threading
from collections.abc import Callable

from psycopg.rows import dict_row

from ..infrastructure.pg_person_search import TRACK_COLUMNS, TRACK_JOINS, rank_similar
from ..infrastructure.pyav_reader import PyAVSegmentReader, describe_segment

Progress = Callable[[str, float, float], None]


def _quiet(stage: str, done: float, total: float) -> None:
    pass


def _jsonl(path: pathlib.Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class RecordingLibrary:
    """The downloaded recordings and what is known about them. Needs no models."""

    def __init__(self, root: pathlib.Path, data_dir: pathlib.Path) -> None:
        self.root = pathlib.Path(root)
        self.data_dir = pathlib.Path(data_dir)

    def files(self) -> list[pathlib.Path]:
        if not self.root.is_dir():
            return []
        return sorted(p for p in self.root.rglob("*.mp4")
                      if not p.name.endswith(".part.mp4") and ".snippets" not in p.parts)

    def path_of(self, rel: str) -> pathlib.Path:
        """A recording by its path under the root. Refuses anything else."""
        root = self.root.resolve()
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or path.suffix.lower() != ".mp4" or not path.is_file():
            raise FileNotFoundError(f"no recording {rel!r}")
        return path

    def rel(self, path: pathlib.Path) -> str:
        return path.resolve().relative_to(self.root.resolve()).as_posix()

    def media_uri(self, path: pathlib.Path) -> str:
        """How the database names the file -- the way scripts/index_clip.py does."""
        path = path.resolve()
        data = self.data_dir.resolve()
        return str(path.relative_to(data)) if path.is_relative_to(data) else str(path)

    @property
    def media_prefix(self) -> str:
        return self.media_uri(self.root) + "/"

    def listing(self, conn) -> list[dict]:
        """The searches, newest first, each with its recordings in hit order, plus
        recordings no search names. Each recording says whether it has been
        processed and how many people with a usable crop it holds."""
        manifest = {m["file"]: m for m in _jsonl(self.root / "manifest.jsonl")}
        cur = conn.cursor(row_factory=dict_row)
        cur.execute(
            "SELECT r.media_uri, count(t.track_id) FILTER (WHERE t.best_crop_uri IS NOT NULL) AS people"
            " FROM recordings r LEFT JOIN tracks t ON t.recording_id = r.recording_id AND t.class_name = 'person'"
            " WHERE r.media_uri LIKE %s GROUP BY r.media_uri", (self.media_prefix + "%",))
        processed = {row["media_uri"]: row["people"] for row in cur.fetchall()}

        def card(path: pathlib.Path, hit: dict | None = None) -> dict:
            rel = self.rel(path)
            camera, started = describe_segment(path)
            m = manifest.get(rel, {})
            uri = self.media_uri(path)
            return {"path": rel, "camera": camera, "recorded_from": started,
                    "seconds": m.get("duration"), "via": m.get("via"), "width": m.get("width"),
                    "height": m.get("height"), "processed": uri in processed, "people": processed.get(uri),
                    "rank": (hit or {}).get("rank"), "score": (hit or {}).get("score"),
                    "semantic_va_clip_id": (hit or {}).get("clip_id")}

        on_disk = {self.rel(p): p for p in self.files()}
        groups, named = [], set()
        for search in reversed(_jsonl(self.root / "searches.jsonl")):
            seen, cards = set(), []
            for hit in search.get("hits", []):
                for rel in hit.get("files") or []:
                    if rel in on_disk and rel not in seen:
                        seen.add(rel)
                        cards.append(card(on_disk[rel], hit))
            named |= seen
            groups.append({"at": search.get("at"), "text": search.get("text"),
                           "predicate": search.get("predicate"), "recordings": cards})
        others = [card(p) for rel, p in on_disk.items() if rel not in named]
        if others:
            groups.append({"at": None, "text": None, "predicate": None, "recordings": others})
        return groups


class RecordingSearch(RecordingLibrary):
    def __init__(self, indexer, connect: Callable, root: pathlib.Path, data_dir: pathlib.Path,
                 fps: float = 2.0) -> None:
        """`indexer`: PersonIndexer. `connect`: a fresh psycopg connection."""
        super().__init__(root, data_dir)
        self.indexer, self.connect, self.fps = indexer, connect, fps
        self.lock = threading.Lock()
        """One job on the GPU at a time (MPS crashes on concurrent use)."""

    def _processed(self, conn) -> set[str]:
        cur = conn.cursor(row_factory=dict_row)
        cur.execute("SELECT media_uri FROM recordings WHERE media_uri LIKE %s", (self.media_prefix + "%",))
        return {row["media_uri"] for row in cur.fetchall()}

    def _process(self, conn, path: pathlib.Path) -> None:
        camera, started = describe_segment(path)
        if started is None:
            raise ValueError(f"{path.name}: no recording time in the name")
        reader = PyAVSegmentReader(camera_key=camera, started_at=started)
        self.indexer.index(conn, reader.frames(str(path), self.fps), camera_key=camera,
                           media_uri=self.media_uri(path), recording_started_at=started,
                           file_size=path.stat().st_size)
        conn.commit()  # each file is kept even if a later one fails

    def people(self, rel: str) -> dict:
        path = self.path_of(rel)
        with self.lock, self.connect() as conn:
            processed_now = self.media_uri(path) not in self._processed(conn)
            if processed_now:
                self._process(conn, path)
            cur = conn.cursor(row_factory=dict_row)
            cur.execute(f"SELECT {TRACK_COLUMNS} FROM {TRACK_JOINS} WHERE r.media_uri = %s"
                        " AND t.class_name = 'person' AND t.best_crop_uri IS NOT NULL ORDER BY t.started_at",
                        (self.media_uri(path),))
            people = cur.fetchall()
        camera, started = describe_segment(path)
        return {"path": self.rel(path), "camera": camera, "recorded_from": started,
                "processed_now": processed_now, "people": people}

    def search(self, track_id: int, model_key: str, top_k: int = 30,
               progress: Progress = _quiet) -> list[dict]:
        with self.lock, self.connect() as conn:
            done = self._processed(conn)
            todo = [p for p in self.files() if self.media_uri(p) not in done]
            for n, path in enumerate(todo):
                progress("processing", n, len(todo))
                self._process(conn, path)
            progress("ranking", len(todo), len(todo))
            rows = rank_similar(conn, track_id, model_key, top_k=top_k, media_prefix=self.media_prefix)
        for row in rows:  # where to play it: the file, and how far into it
            row["path"] = pathlib.Path(row["media_uri"]).relative_to(self.media_prefix.rstrip("/")).as_posix()
            row["offset"] = round((row["started_at"] - row["recording_started_at"]).total_seconds(), 2)
        return rows
