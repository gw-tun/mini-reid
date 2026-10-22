"""Read a recorded segment as sampled frames (implements the `SegmentReader` port).

Each frame carries the camera and the RECORDING time: when the camera saw it,
not when the worker processed it. Cross-camera timing (Live Follow windows,
temporal reranking) is meaningless on processing time.

Recording start comes from the file name:

    MediaMTX  /recordings/cam01/2026-10-19_08-30-00.mp4   (infra/mediamtx recordPath)
    MEVA      ...2018-03-13.15-50-03.15-55-03.school.G336.r13.avi

Frame time = recording start + the frame's position in the file. Names carry no
time zone, so `RECORDING_TZ` (default UTC, MediaMTX's container default) is
attached; set it to the cameras' zone if they record local time.
"""

from __future__ import annotations

import datetime as dt
import math
import os
import pathlib
import re
from collections.abc import Iterator
from zoneinfo import ZoneInfo

from ..domain.models import Frame

MEDIAMTX_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})")
MEVA_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})\.(\d{2})-(\d{2})-(\d{2})")
MEVA_CAMERA = re.compile(r"\.(G\d+)(?:__\d+)?\.")


def describe_segment(path: str | pathlib.Path,
                     tz: dt.tzinfo | None = None) -> tuple[str, dt.datetime | None]:
    """(camera key, recording start) read from a segment's path."""
    path = pathlib.Path(path)
    tz = tz or ZoneInfo(os.getenv("RECORDING_TZ", "UTC"))
    for pattern in (MEDIAMTX_NAME, MEVA_NAME):
        m = pattern.search(path.name)
        if m:
            start = dt.datetime.fromisoformat(
                f"{m.group(1)}T{m.group(2)}:{m.group(3)}:{m.group(4)}").replace(tzinfo=tz)
            camera = MEVA_CAMERA.search(path.name + ".")
            return (camera.group(1) if pattern is MEVA_NAME and camera else path.parent.name), start
    return path.parent.name, None


class PyAVSegmentReader:
    def __init__(self, camera_key: str | None = None, started_at: dt.datetime | None = None) -> None:
        """Overrides for files whose name carries neither (e.g. a test clip)."""
        self.camera_key = camera_key
        self.started_at = started_at

    def frames(self, segment_path: str, sample_fps: float,
               start_s: float = 0.0, end_s: float | None = None) -> Iterator[Frame]:
        """Sampled frames of the file, or of only `start_s`..`end_s` (seconds into it).

        A stretch seeks to the keyframe before `start_s` instead of decoding from
        the top, and samples on the SAME grid as a whole-file read (multiples of
        1/fps from the file's start), so two overlapping stretches yield the same
        frames at the same times.
        """
        import av

        camera, start = describe_segment(segment_path)
        camera = self.camera_key or camera
        start = self.started_at or start
        if start is None:
            raise ValueError(f"can't tell when {segment_path} was recorded: "
                             "name it like MediaMTX (2026-10-19_08-30-00.mp4) or pass started_at")
        step = 1.0 / sample_fps
        next_t = math.ceil(start_s * sample_fps - 1e-6) * step
        with av.open(str(segment_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            # The file's own clock starts here; "seconds into the file" count from it.
            origin = float(stream.start_time * stream.time_base) if stream.start_time is not None else None
            if start_s > 0:
                origin = origin or 0.0
                container.seek(int((origin + start_s) / stream.time_base), stream=stream, backward=True)
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                t = float(frame.pts * stream.time_base)
                origin = t if origin is None else origin
                t -= origin
                if end_s is not None and t > end_s + 1e-6:
                    break
                if t + 1e-6 < next_t:
                    continue  # between samples: decoded (unavoidable) but not converted
                while next_t <= t + 1e-6:
                    next_t += step
                yield Frame(camera, start + dt.timedelta(seconds=t),
                            frame.to_ndarray(format="rgb24"), str(segment_path))
