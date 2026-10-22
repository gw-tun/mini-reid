"""Optional media/model adapters used by the local MVP commands.

The application services depend on protocols. These adapters keep OpenCV and
Ultralytics at the infrastructure edge so the backbone remains testable without
GPU or model downloads.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ..application.backbone import CropQualityPolicy
from ..domain.models import (
    CropSample,
    Detection,
    Embedding,
    Frame,
    ObjectType,
    TrackObservation,
)


class OpenCVSegmentReader:
    def __init__(self, camera_key: str | None = None) -> None:
        self._camera_key = camera_key

    def frames(
        self, segment_path: str, sample_fps: float, started_at: datetime | None = None
    ):
        import cv2

        capture = cv2.VideoCapture(segment_path)
        if not capture.isOpened():
            raise FileNotFoundError(f"cannot open video: {segment_path}")
        source = Path(segment_path)
        # The event-derived start is the true wall-clock time of video frame 0;
        # mtime is only a legacy fallback for CLI/replay paths that lack one.
        if started_at is not None:
            start = started_at
        else:
            start = datetime.fromtimestamp(source.stat().st_mtime, tz=timezone.utc)
        native_fps = capture.get(cv2.CAP_PROP_FPS) or sample_fps
        step = max(1, round(native_fps / max(sample_fps, 0.1)))
        frame_number = 0
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                if frame_number % step == 0:
                    yield Frame(
                        camera_key=self._camera_key or source.stem,
                        captured_at=start + timedelta(seconds=frame_number / native_fps),
                        image=image,
                        source_path=segment_path,
                    )
                frame_number += 1
        finally:
            capture.release()


class UltralyticsDetector:
    """YOLO adapter; import is lazy so unit tests do not need Ultralytics."""

    def __init__(self, weights: str, confidence: float = 0.35) -> None:
        from ultralytics import YOLO

        self._model = YOLO(weights)
        self._confidence = confidence

    def detect(self, frame: Frame) -> list[Detection]:
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: list[Frame]) -> list[list[Detection]]:
        """Detect many frames in one Ultralytics call.

        CPU inference dominates indexing time; batching ~8 frames roughly
        halves the per-frame cost while producing the same boxes as predict().
        """

        if not frames:
            return []
        results = self._model.predict([frame.image for frame in frames], conf=self._confidence, verbose=False)
        batched: list[list[Detection]] = []
        for result in results:
            boxes = result.boxes
            detections: list[Detection] = []
            for box, confidence, class_id in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
                label = int(class_id)
                if label == 0:
                    object_type = ObjectType.PERSON
                elif label in {2, 3, 5, 7}:
                    object_type = ObjectType.VEHICLE
                else:
                    continue
                detections.append(Detection(object_type, float(confidence), tuple(float(v) for v in box)))
            batched.append(detections)
        return batched


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union else 0.0


class IoUTracker:
    """Small single-process tracker for the first CPU MVP."""

    def __init__(self, match_iou: float = 0.3) -> None:
        self._match_iou = match_iou
        self._next_id = 1
        self._tracks: dict[str, list[tuple[str, ObjectType, tuple[float, float, float, float]]]] = {}

    def update(self, frame: Frame, detections: list[Detection]) -> list[TrackObservation]:
        previous = self._tracks.setdefault(frame.camera_key, [])
        current: list[tuple[str, ObjectType, tuple[float, float, float, float]]] = []
        observations: list[TrackObservation] = []
        used_previous: set[str] = set()
        for detection in detections:
            best = max(
                (item for item in previous if item[0] not in used_previous and item[1] is detection.object_type),
                key=lambda item: _iou(item[2], detection.bbox),
                default=None,
            )
            track_id = best[0] if best and _iou(best[2], detection.bbox) >= self._match_iou else f"track-{self._next_id}"
            if best is not None and track_id == best[0]:
                used_previous.add(best[0])
            else:
                self._next_id += 1
            current.append((track_id, detection.object_type, detection.bbox))
            observations.append(
                TrackObservation(
                    camera_key=frame.camera_key,
                    track_id=track_id,
                    object_type=detection.object_type,
                    observed_at=frame.captured_at,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    image=frame.image,
                )
            )
        self._tracks[frame.camera_key] = current
        return observations


class OpenCVCropSelector:
    def __init__(self, policy: CropQualityPolicy | None = None) -> None:
        self._policy = policy or CropQualityPolicy()

    def select(self, observation: TrackObservation) -> CropSample | None:
        try:
            import numpy as np
            if not self._policy.accepts(observation):
                return None
            image = observation.image
            if not isinstance(image, np.ndarray):
                return None
            height, width = image.shape[:2]
            x1, y1, x2, y2 = observation.bbox
            left, top = max(0, int(x1)), max(0, int(y1))
            right, bottom = min(width, int(x2)), min(height, int(y2))
            if right <= left or bottom <= top:
                return None
            return CropSample(observation.track_id, observation.object_type, image[top:bottom, left:right].copy(), self._policy.score(observation))
        except ImportError:
            return None


class ColorHistogramReID:
    """Deterministic CPU baseline; replace with FastReID behind the same port."""

    def embed(self, crop: CropSample) -> Embedding | None:
        try:
            import cv2
            import numpy as np
            image = crop.image
            if not isinstance(image, np.ndarray) or image.size == 0:
                return None
            image = cv2.resize(image, (64, 128))
            channels = cv2.split(image)
            values: list[float] = []
            for channel in channels:
                histogram = cv2.calcHist([channel], [0], None, [32], [0, 256]).flatten()
                values.extend((histogram / (np.linalg.norm(histogram) + 1e-12)).tolist())
            vector = tuple(float(value) for value in values)
            return Embedding(f"hist-{crop.object_type.value}-v1", vector, "histogram-1")
        except ImportError:
            return None


class FastReIDModel:
    """FastReID DefaultPredictor adapter for the downloaded model zoo weights."""

    def __init__(self, config_path: str, weights_path: str, vector_space: str, device: str = "cpu", model_version: str = "v0.1.1") -> None:
        from . import fastreid_compat  # noqa: F401  installs torch._six / collections.abc shims

        try:
            from fastreid.config import get_cfg
            from fastreid.engine import DefaultPredictor
        except ImportError as exc:
            raise RuntimeError(
                "FastReID backend requested but fast-reid is not installed; "
                "install it with: python -m pip install -e .\\packages\\fastreid "
                "(or use REID_BACKEND=histogram)"
            ) from exc
        cfg = get_cfg()
        cfg.merge_from_file(config_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.DEVICE = device
        self._cfg = cfg
        self._predictor = DefaultPredictor(cfg)
        self._vector_space = vector_space
        self._weights_path = weights_path
        self._model_version = model_version

    def embed(self, crop: CropSample) -> Embedding | None:
        import cv2
        import numpy as np
        import torch

        image = crop.image
        if not isinstance(image, np.ndarray) or image.size == 0:
            return None
        # FastReID expects an RGB float32 tensor of shape (1, 3, H, W) at
        # INPUT.SIZE_TEST; the model applies its own pixel mean/std normalization.
        size = tuple(int(value) for value in self._cfg.INPUT.SIZE_TEST[::-1])
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, size, interpolation=cv2.INTER_CUBIC)
        tensor = torch.as_tensor(resized.astype("float32").transpose(2, 0, 1))[None]
        output = self._predictor(tensor)
        if hasattr(output, "detach"):
            values = output.detach().cpu().flatten().numpy()
        else:
            values = np.asarray(output).reshape(-1)
        if values.size == 0:
            return None
        return Embedding(self._vector_space, tuple(float(value) for value in values), self._model_version)
