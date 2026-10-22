"""Live camera detection boxes for the Live View overlay (SSE stream)."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Protocol

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from services.worker.app.domain.models import Detection, Frame

router = APIRouter(prefix="/internal", tags=["internal-live"])

_CAMERA_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
_DEFAULT_RTSP_TEMPLATE = "rtsp://127.0.0.1:8554/{camera}"

# Prefer TCP for OpenCV's FFmpeg backend so partial UDP packets do not tear
# the live view apart on lossy links.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

_CONSTRUCT_LOCK = threading.Lock()
_INFER_LOCK = threading.Lock()
_detector = None


class _Capture(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, object]: ...
    def release(self) -> None: ...


def _shared_detector():
    """Lazy-load one Ultralytics model for every live-view subscriber."""

    global _detector
    with _CONSTRUCT_LOCK:
        if _detector is None:
            model_root = Path(os.getenv("MODEL_ROOT", "/models" if Path("/models").is_dir() else "models"))
            weights = os.getenv("YOLO_WEIGHTS", str(model_root / "yolo" / "yolo11m.pt"))
            if not Path(weights).is_file():
                raise RuntimeError(f"YOLO weights not found: {weights}")
            config_dir = Path(os.getenv("YOLO_CONFIG_DIR", "data/cache/ultralytics"))
            config_dir.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
            from services.worker.app.infrastructure.media import UltralyticsDetector

            _detector = UltralyticsDetector(weights, float(os.getenv("YOLO_CONFIDENCE", "0.35")))
        return _detector


def _normalize_boxes(detections: list[Detection], width: int, height: int) -> list[dict[str, object]]:
    """Convert pixel boxes to clamped frame fractions the overlay can style directly."""

    if width <= 0 or height <= 0:
        return []
    boxes: list[dict[str, object]] = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        boxes.append(
            {
                "label": detection.object_type.value,
                "score": round(float(detection.confidence), 3),
                "x1": round(min(max(x1 / width, 0.0), 1.0), 4),
                "y1": round(min(max(y1 / height, 0.0), 1.0), 4),
                "x2": round(min(max(x2 / width, 0.0), 1.0), 4),
                "y2": round(min(max(y2 / height, 0.0), 1.0), 4),
            }
        )
    return boxes


def _event(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


class _LatestFrame:
    """Hand-off of the freshest decoded frame from a pump thread to the detector.

    RTSP delivery is in-order and cannot be skipped, so a consumer that reads
    as slowly as YOLO inference would fall permanently behind live video. The
    pump keeps reading at stream rate and this holder keeps only the newest
    frame, bounding box/video misalignment to roughly one detect pass.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._image: object = None
        self._seq = 0
        self._failed = False
        self._closed = False

    def put(self, image: object) -> None:
        with self._condition:
            self._image = image
            self._seq += 1
            self._condition.notify_all()

    def fail(self) -> None:
        with self._condition:
            self._failed = True
            self._condition.notify_all()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def await_next(self, after_seq: int, timeout: float) -> tuple:
        """Return ("frame", seq, image), ("failed",), ("closed",), or ("timeout",)."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                if self._closed:
                    return ("closed",)
                if self._seq > after_seq:
                    return ("frame", self._seq, self._image)
                if self._failed:
                    return ("failed",)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return ("timeout",)
                self._condition.wait(remaining)


def _pump_frames(capture: _Capture, holder: _LatestFrame) -> None:
    """Decode continuously so stale frames are dropped instead of queued."""

    try:
        while True:
            ok, image = capture.read()
            if not ok:
                holder.fail()
                return
            holder.put(image)
    except Exception:
        holder.fail()


def _detection_events(
    camera: str,
    fps: float,
    *,
    open_capture: Callable[[], _Capture],
    detect: Callable[[Frame], list[Detection]],
    sleep: Callable[[float], None] = time.sleep,
) -> Iterator[str]:
    """Yield Server-Sent Events until the client disconnects or the stream dies.

    ``open_capture``/``detect``/``sleep`` are injectable so unit tests can run
    the loop without RTSP or the YOLO runtime.
    """

    capture = open_capture()
    holder = _LatestFrame()
    pump: threading.Thread | None = None
    try:
        if not capture.isOpened():
            yield _event({"type": "error", "camera": camera, "detail": "camera stream unavailable"})
            return
        pump = threading.Thread(
            target=_pump_frames,
            args=(capture, holder),
            daemon=True,
            name=f"live-frame-pump-{camera}",
        )
        pump.start()
        interval = 1.0 / fps
        seq = 0
        while True:
            started = time.monotonic()
            status: tuple = ("timeout",)
            while status[0] == "timeout":
                status = holder.await_next(seq, timeout=1.0)
            if status[0] == "failed":
                yield _event({"type": "error", "camera": camera, "detail": "frame read failed"})
                return
            if status[0] == "closed":
                return
            seq = int(status[1])
            image = status[2]
            height, width = image.shape[:2]
            frame = Frame(
                camera_key=camera,
                captured_at=datetime.now(timezone.utc),
                image=image,
                source_path=camera,
            )
            with _INFER_LOCK:
                detections = detect(frame)
            yield _event(
                {
                    "type": "detections",
                    "camera": camera,
                    "frame": {"width": width, "height": height},
                    "boxes": _normalize_boxes(detections, width, height),
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                sleep(remaining)
    finally:
        holder.close()
        if pump is not None:
            pump.join(timeout=2.0)
        capture.release()


@router.get("/live/detections")
def live_detections(camera: str = "cam01", fps: float = 2.0) -> StreamingResponse:
    """Stream YOLO boxes for one RTSP camera as SSE while the viewer is connected."""

    if not _CAMERA_RE.match(camera):
        raise HTTPException(status_code=422, detail="invalid camera name")
    if not 0.5 <= fps <= 5.0:
        raise HTTPException(status_code=422, detail="fps must be between 0.5 and 5")
    try:
        detector = _shared_detector()
    except (ImportError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"detection runtime unavailable: {exc}") from exc

    template = os.getenv("LIVE_RTSP_TEMPLATE", _DEFAULT_RTSP_TEMPLATE)
    url = template.format(camera=camera)

    def open_capture() -> _Capture:
        import cv2

        return cv2.VideoCapture(url)

    def detect(frame: Frame) -> list[Detection]:
        return detector.detect(frame)

    return StreamingResponse(
        _detection_events(camera, fps, open_capture=open_capture, detect=detect),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
