"""Local MVP commands: index a video, search by image, and inspect detections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv()

from services.common.events import RecordSegmentEvent

from .application.backbone import EmbeddingService, ModelRegistry, ModelSpec
from .application.record_indexer import RecordIndexingService
from .application.record_search import RecordSearchService
from .domain.models import Frame, ObjectType
from .infrastructure.media import (
    ColorHistogramReID,
    FastReIDModel,
    IoUTracker,
    OpenCVCropSelector,
    OpenCVSegmentReader,
    UltralyticsDetector,
)
from .infrastructure.postgres import PostgresRecordingStore, PostgresSearchResultStore, PostgresVectorIndex


def build_embedding_service(
    backend: str = "histogram",
    *,
    person_weights: str = "models/fastreid/msmt17_sbs_R50-IBN.pth",
    person_config: str = "models/fastreid/configs/msmt17_sbs_R50-IBN.yml",
    vehicle_weights: str = "models/fastreid/veri_sbs_R50-IBN.pth",
    vehicle_config: str = "models/fastreid/configs/veri_sbs_R50-IBN.yml",
    device: str = "cpu",
) -> EmbeddingService:
    if backend == "fastreid":
        models = {
            ObjectType.PERSON: FastReIDModel(person_config, person_weights, "fastreid-msmt17-v1", device),
            ObjectType.VEHICLE: FastReIDModel(vehicle_config, vehicle_weights, "fastreid-veri-v1", device),
        }
        specs = {
            ObjectType.PERSON: ModelSpec("fastreid-msmt17", ObjectType.PERSON, "fastreid-msmt17-v1", 2048),
            ObjectType.VEHICLE: ModelSpec("fastreid-veri", ObjectType.VEHICLE, "fastreid-veri-v1", 2048),
        }
        return EmbeddingService(ModelRegistry(specs), models)
    model = ColorHistogramReID()
    return EmbeddingService(
        ModelRegistry({
            ObjectType.PERSON: ModelSpec("hist-person", ObjectType.PERSON, "hist-person-v1", 96),
            ObjectType.VEHICLE: ModelSpec("hist-vehicle", ObjectType.VEHICLE, "hist-vehicle-v1", 96),
        }),
        {ObjectType.PERSON: model, ObjectType.VEHICLE: model},
    )


def index_video(args: argparse.Namespace) -> None:
    import cv2

    if args.reid_backend != "fastreid":
        raise SystemExit("PostgreSQL gallery requires REID_BACKEND=fastreid (vector(2048))")
    capture = cv2.VideoCapture(args.video)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1
    capture.release()
    duration = max(0.01, frame_count / fps)
    event_id = hashlib.sha256(f"{args.camera}\n{args.video}".encode()).hexdigest()
    event = RecordSegmentEvent(
        event_id=event_id, camera_key=args.camera, segment_path=args.video,
        duration_seconds=duration, completed_at=datetime.now(timezone.utc),
    )
    indexer = RecordIndexingService(
        OpenCVSegmentReader(args.camera), UltralyticsDetector(args.yolo), IoUTracker(),
        OpenCVCropSelector(), build_embedding_service(args.reid_backend, device=args.device),
        PostgresVectorIndex(), PostgresRecordingStore(), args.sample_fps,
    )
    print(
        f"indexing video={args.video} camera={args.camera} "
        f"backend={args.reid_backend} sample_fps={args.sample_fps} ..."
    )
    result = indexer.index_segment(event)
    print(result)
    print(f"wrote {result.embeddings_written} embeddings to PostgreSQL")


def search_image(args: argparse.Namespace) -> None:
    import cv2

    if args.reid_backend != "fastreid":
        raise SystemExit("PostgreSQL gallery requires REID_BACKEND=fastreid (vector(2048))")
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)
    object_type = ObjectType(args.object_type)
    results = RecordSearchService(
        build_embedding_service(args.reid_backend, device=args.device),
        PostgresVectorIndex(),
        args.min_similarity,
        result_store=PostgresSearchResultStore(),
    ).search(image, object_type, args.top_k)
    print(json.dumps([candidate.__dict__ for candidate in results], default=str, indent=2))


def annotate_video(args: argparse.Namespace) -> None:
    """Replay a recording with YOLO boxes and local track IDs burned in."""
    import cv2

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise FileNotFoundError(args.video)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    fps = capture.get(cv2.CAP_PROP_FPS) or 25
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    detector = UltralyticsDetector(args.yolo)
    tracker = IoUTracker()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame = Frame(args.camera, datetime.now(timezone.utc), image, args.video)
            observations = tracker.update(frame, detector.detect(frame))
            for observation in observations:
                x1, y1, x2, y2 = (int(value) for value in observation.bbox)
                color = (0, 200, 0) if observation.object_type is ObjectType.PERSON else (255, 160, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{observation.object_type.value}:{observation.track_id} {observation.confidence:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            writer.write(image)
    finally:
        capture.release()
        writer.release()
    print(args.output)


def monitor_live(args: argparse.Namespace) -> None:
    """Show a live RTSP/MediaMTX stream with YOLO boxes and local track IDs."""
    import cv2

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"cannot open live source: {args.source}")
    detector = UltralyticsDetector(args.yolo)
    tracker = IoUTracker()
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            frame = Frame(args.camera, datetime.now(timezone.utc), image, args.source)
            for observation in tracker.update(frame, detector.detect(frame)):
                x1, y1, x2, y2 = (int(value) for value in observation.bbox)
                color = (0, 200, 0) if observation.object_type is ObjectType.PERSON else (255, 160, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{observation.object_type.value}:{observation.track_id}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.imshow(f"mini-reid {args.camera}", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-reid local Record Search MVP")
    sub = parser.add_subparsers(required=True)
    index = sub.add_parser("index-video")
    index.add_argument("--video", required=True)
    index.add_argument("--camera", required=True)
    index.add_argument("--yolo", default=os.getenv("YOLO_WEIGHTS", "models/yolo/yolo11m.pt"))
    index.add_argument("--sample-fps", type=float, default=float(os.getenv("SAMPLE_FPS", "2")))
    index.add_argument("--reid-backend", choices=["histogram", "fastreid"], default=os.getenv("REID_BACKEND", "fastreid"))
    index.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    index.set_defaults(func=index_video)
    search = sub.add_parser("search-image")
    search.add_argument("--image", required=True)
    search.add_argument("--object-type", choices=[item.value for item in ObjectType], default="person")
    search.add_argument("--top-k", type=int, default=20)
    search.add_argument("--min-similarity", type=float, default=float(os.getenv("REID_MIN_SIMILARITY", "0.65")))
    search.add_argument("--reid-backend", choices=["histogram", "fastreid"], default=os.getenv("REID_BACKEND", "fastreid"))
    search.add_argument("--device", default=os.getenv("DEVICE", "cpu"))
    search.set_defaults(func=search_image)
    annotate = sub.add_parser("annotate-video")
    annotate.add_argument("--video", required=True)
    annotate.add_argument("--camera", required=True)
    annotate.add_argument("--output", required=True)
    annotate.add_argument("--yolo", default="models/yolo/yolo11m.pt")
    annotate.set_defaults(func=annotate_video)
    live = sub.add_parser("monitor-live")
    live.add_argument("--source", required=True, help="RTSP URL, MediaMTX path, or webcam index")
    live.add_argument("--camera", required=True)
    live.add_argument("--yolo", default="models/yolo/yolo11m.pt")
    live.set_defaults(func=monitor_live)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
