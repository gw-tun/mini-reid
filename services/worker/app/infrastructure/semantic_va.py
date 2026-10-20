"""A small, read-only client for semantic-va's HTTP API.

Used by scripts/fetch_search_clips.py to bring search-result video into
mini-reid. GET and the read-only POST /search only; nothing here changes
semantic-va. Standard library only.

Clocks (see archive/legacy-docs/SEMANTIC-VA-NOTES.md):
- `clip_start` / `clip_end`: seconds into the SOURCE recording.
- `capture_ts`: when the clip was RECORDED -- a naive local wall clock taken
  from the file name. Kept naive here, as semantic-va keeps it.
- `wall_ts`: when semantic-va INDEXED the clip. Never used for time.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import time
import urllib.error
import urllib.parse
import urllib.request

MEVA_CAMERA = re.compile(r"\.(G\d+)(?:__\d+)?\.")
"""MEVA names carry the camera: `...school.G336.r13` is G336. Activity clips cut
from those files add `__03`, and are the same camera's footage."""


class SemanticVaError(RuntimeError):
    """semantic-va didn't give what was asked (not running, bad request, no recording)."""


class SemanticVa:
    def __init__(self, base: str | None = None, timeout: float = 120) -> None:
        self.base = (base or os.environ.get("SEMANTIC_VA_API", "http://10.1.2.89:3000/api")).rstrip("/")
        self.timeout = timeout

    def _call(self, method: str, path: str, body: dict | None = None) -> bytes:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:300]
            raise SemanticVaError(f"semantic-va {method} {path}: HTTP {exc.code} {detail}") from exc
        except urllib.error.URLError as exc:
            raise SemanticVaError(f"semantic-va is not reachable at {self.base} ({exc.reason})") from exc

    def health(self) -> dict:
        return json.loads(self._call("GET", "/health"))

    def search(self, request: dict) -> list[dict]:
        """`POST /search` (read-only) -> the hits of its one rule, best first."""
        results = json.loads(self._call("POST", "/search", request))["results"]
        return results[0]["hits"] if results else []

    def clip(self, clip_id: int) -> dict:
        """`GET /clips/{id}`: `hit`, `tracks`, `has_media`, `source_url`, `storage_location`."""
        return json.loads(self._call("GET", f"/clips/{int(clip_id)}?neighbours=0"))

    def recording(self, clip_id: int, start: float, duration: float) -> bytes:
        """`GET /clips/{id}/recording`: an H.264 MP4 of `start`..`start+duration`
        seconds of the clip's source (at most 10 s, at most 960 px)."""
        query = urllib.parse.urlencode({"start": f"{start:.3f}", "duration": f"{duration:.3f}"})
        return self._call("GET", f"/clips/{int(clip_id)}/recording?{query}")

    def snippet(self, clip_id: int, seconds: float) -> tuple[bytes, float]:
        """`GET /clips/{id}/snippet`: the ORIGINAL video, copied rather than
        re-encoded (full resolution), from `frame_pts - seconds` (not before 0)
        to `frame_pts + seconds`, as MP4 -> (bytes, lead).

        A copy can only begin at a keyframe, so the file opens `lead` seconds
        BEFORE `frame_pts - seconds` (semantic-va's X-Lead-Seconds; up to 2 s on
        MEVA). Caveat: semantic-va keeps leads in memory, so a snippet it serves
        from its cache after a restart reports 0 -- times then run up to one
        keyframe interval late.
        """
        path = f"/clips/{int(clip_id)}/snippet?seconds={min(30.0, max(1.0, seconds)):.2f}"
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
                    if r.status == 204:
                        raise SemanticVaError(f"clip {clip_id}: no recording behind it")
                    return r.read(), float(r.headers.get("X-Lead-Seconds") or 0.0)
            except urllib.error.HTTPError as exc:
                if exc.code == 503 and attempt == 1:  # its cutting queue is full: wait as told, once
                    time.sleep(float(exc.headers.get("Retry-After") or 5))
                    continue
                raise SemanticVaError(f"semantic-va GET {path}: HTTP {exc.code}") from exc
            except urllib.error.URLError as exc:
                raise SemanticVaError(f"semantic-va is not reachable at {self.base} ({exc.reason})") from exc
        raise SemanticVaError(f"semantic-va GET {path}: busy")


def camera_of(stream_id: str, source_url: str | None = None) -> str:
    """MEVA's camera code when the names carry one, else the stream id (made path-safe)."""
    for name in (stream_id, pathlib.PurePath(source_url or "").name):
        m = MEVA_CAMERA.search(name + ".")
        if m:
            return m.group(1)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stream_id)[:80]


def file_started_at(hit: dict) -> dt.datetime | None:
    """Recording time of second 0 of the clip's source file (naive), or None
    when semantic-va has no recording time for it (a live stream)."""
    if not hit.get("capture_ts"):
        return None
    at = dt.datetime.fromisoformat(hit["capture_ts"]).replace(tzinfo=None)
    return at - dt.timedelta(seconds=float(hit["clip_start"]))
