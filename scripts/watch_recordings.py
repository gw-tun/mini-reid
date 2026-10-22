"""Host-side MediaMTX completion watcher.

The official MediaMTX image ships without curl, so the in-container
runOnRecordSegmentComplete hook cannot fire. This script polls the shared
recordings directory for finalized segments and POSTs the same webhook the
hook would have sent, keeping the record → embed pipeline fully automatic.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# MediaMTX recordPath pattern: /recordings/%path/%Y-%m-%d_%H-%M-%S (+ .mp4 for fmp4).
SEGMENT_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.mp4$")
SETTLE_SECONDS = 3.0


def segment_duration(path: Path) -> float | None:
    try:
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if frames <= 0 or fps <= 0:
            return None
        return max(frames / fps, 0.5)
    except ImportError:
        return None


def notify(api_url: str, camera: str, segment_path: str, duration: float) -> str:
    query = urllib.parse.urlencode(
        {
            "camera": camera,
            "segment_path": segment_path,
            "duration": f"{duration:.3f}",
        }
    )
    request = urllib.request.Request(
        f"{api_url}/internal/record-segments/complete?{query}",
        method="POST",
        data=b"",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return str(payload.get("status", "unknown"))


def watch(api_url: str, recordings_root: Path, poll_seconds: float, once: bool) -> int:
    # On startup treat existing files as already handled so a restart does not
    # re-index the whole archive; only segments finalized after start are new.
    seen: set[str] = {
        str(path.resolve())
        for camera_dir in recordings_root.iterdir()
        if camera_dir.is_dir()
        for path in camera_dir.glob("*.mp4")
        if SEGMENT_NAME.match(path.name)
    } if recordings_root.is_dir() else set()
    if seen:
        print(f"watching {recordings_root}; {len(seen)} existing segment(s) marked handled")
    else:
        print(f"watching {recordings_root}; no segments yet")
    while True:
        if recordings_root.is_dir():
            for camera_dir in sorted(recordings_root.iterdir()):
                if not camera_dir.is_dir():
                    continue
                for path in sorted(camera_dir.glob("*.mp4")):
                    key = str(path.resolve())
                    if key in seen or not SEGMENT_NAME.match(path.name):
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    if time.time() - stat.st_mtime < SETTLE_SECONDS:
                        continue  # still being written
                    duration = segment_duration(path)
                    if duration is None:
                        seen.add(key)
                        print(f"skip unreadable segment {path}")
                        continue
                    # Container-style path keeps event_id stable with the hook.
                    container_path = f"/recordings/{camera_dir.name}/{path.name}"
                    try:
                        status = notify(api_url, camera_dir.name, container_path, duration)
                    except (urllib.error.URLError, OSError, TimeoutError) as exc:
                        print(f"notify failed for {path.name}: {exc}")
                        continue
                    seen.add(key)
                    print(f"{status}: {camera_dir.name}/{path.name} ({duration:.1f}s)")
        if once:
            return 0
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="watch MediaMTX recordings and enqueue indexing")
    parser.add_argument("--api-url", default=os.getenv("MINI_REID_API", "http://127.0.0.1:8090"))
    parser.add_argument("--recordings", default=os.getenv("RECORDINGS_DIR", "data/recordings"))
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--once", action="store_true", help="scan once and exit")
    args = parser.parse_args()
    try:
        watch(args.api_url.rstrip("/"), Path(args.recordings), args.poll_seconds, args.once)
    except KeyboardInterrupt:
        print("watcher stopped")
        sys.exit(0)


if __name__ == "__main__":
    main()
