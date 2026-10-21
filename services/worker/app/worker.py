"""Entry point for the shared AI worker."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from services.common.job_queue import FileJobQueue
from services.common.paths import resolve_media_path

load_dotenv()

from .application.live_camera import LiveCameraService
from .application.live_follow import LiveFollowService
from .application.record_indexer import RecordIndexingService
from .domain.models import CameraTransition, Frame, FollowSession
from .infrastructure.adapters import (
    EmptyFollowEventSink,
    StaticTransitionGraph,
)


class WorkerRuntime:
    def __init__(
        self,
        queue: FileJobQueue,
        indexer: RecordIndexingService,
        live_camera: LiveCameraService,
    ) -> None:
        self._queue = queue
        self._indexer = indexer
        self._live_camera = live_camera

    def run_once(self, batch_size: int = 4, *, announce_empty: bool = False) -> int:
        jobs = self._queue.claim(limit=batch_size)
        if not jobs:
            if announce_empty:
                print("no pending jobs")
            return 0
        for job in jobs:
            try:
                # MediaMTX webhooks carry container paths (/recordings/...);
                # map them onto the host tree before opening the file so the
                # stored media_uri and the reader agree on one location.
                event = job.event.model_copy(
                    update={"segment_path": resolve_media_path(job.event.segment_path)}
                )
                result = self._indexer.index_segment(event)
                if result.frames_seen == 0:
                    self._queue.fail(job, "indexer produced no frames; verify video path and model runtime")
                    print(f"failed job={job.job_id}: indexer produced no frames")
                    continue
                self._queue.acknowledge(job)
                print(
                    f"indexed job={job.job_id} camera={job.event.camera_key} "
                    f"frames={result.frames_seen} embeddings={result.embeddings_written}"
                )
            except Exception as exc:  # keep one bad segment from stopping the worker
                self._queue.fail(job, str(exc))
                print(f"failed job={job.job_id}: {exc}")
        return len(jobs)

    def process_live_frame(self, frame: Frame, session: FollowSession):
        """Entry point for a stream adapter once RTSP/live decoding is added."""

        return self._live_camera.process_frame(frame, session)

    def run_forever(self, poll_seconds: float = 1.0) -> None:
        queue_dir = getattr(self._queue, "root", "job-queue")
        print(f"worker started; polling {queue_dir} every {poll_seconds:g}s (Ctrl+C to stop)")
        last_idle_announce = 0.0
        try:
            while True:
                processed = self.run_once()
                if processed == 0:
                    now = time.monotonic()
                    if now - last_idle_announce >= 30.0:
                        print("no pending jobs")
                        last_idle_announce = now
                    time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("worker stopped")


def build_runtime() -> WorkerRuntime:
    queue = FileJobQueue(os.getenv("JOB_QUEUE_DIR", "data/cache/job-queue"))
    model_root = Path(os.getenv("MODEL_ROOT", "/models" if Path("/models").is_dir() else "models"))
    weights = Path(os.getenv("YOLO_WEIGHTS", str(model_root / "yolo" / "yolo11m.pt")))
    if not weights.is_file():
        raise RuntimeError(
            f"YOLO weights not found: {weights}. Run "
            "powershell.exe -NoProfile -ExecutionPolicy Bypass "
            "-File .\\scripts\\download_models.ps1"
        )

    # Ultralytics writes its settings on import. Keep that file inside the
    # project data directory so a read-only user profile or container can run.
    config_dir = Path(os.getenv("YOLO_CONFIG_DIR", "data/cache/ultralytics"))
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))

    # Every vector is persisted in PostgreSQL/pgvector (vector(2048)), so the
    # worker only supports the 2048-D FastReID gallery.
    reid_backend = os.getenv("REID_BACKEND", "fastreid")
    if reid_backend != "fastreid":
        raise RuntimeError(
            "vectors are stored in PostgreSQL vector(2048); set REID_BACKEND=fastreid"
        )

    try:
        from .cli import build_embedding_service
        from .infrastructure.media import IoUTracker, OpenCVCropSelector, OpenCVSegmentReader, UltralyticsDetector

        detector = UltralyticsDetector(str(weights), float(os.getenv("YOLO_CONFIDENCE", "0.35")))
        tracker = IoUTracker()
        crop_selector = OpenCVCropSelector()
        reader = OpenCVSegmentReader()
        reid = build_embedding_service(
            reid_backend,
            device=os.getenv("DEVICE", "cpu"),
            person_weights=os.getenv("PERSON_REID_WEIGHTS", str(model_root / "fastreid" / "msmt17_sbs_R50-IBN.pth")),
            person_config=os.getenv("PERSON_REID_CONFIG", str(model_root / "fastreid" / "configs" / "msmt17_sbs_R50-IBN.yml")),
            vehicle_weights=os.getenv("VEHICLE_REID_WEIGHTS", str(model_root / "fastreid" / "veri_sbs_R50-IBN.pth")),
            vehicle_config=os.getenv("VEHICLE_REID_CONFIG", str(model_root / "fastreid" / "configs" / "veri_sbs_R50-IBN.yml")),
        )
        from .infrastructure.postgres import PostgresRecordingStore, PostgresVectorIndex

        vector_index = PostgresVectorIndex()
        store = PostgresRecordingStore()
    except ImportError as exc:
        raise RuntimeError(
            f"model runtime unavailable ({exc}). Install dependencies with: "
            "python -m pip install -r services/worker/requirements.txt"
        ) from exc
    indexer = RecordIndexingService(
        reader=reader,
        detector=detector,
        tracker=tracker,
        crop_selector=crop_selector,
        reid=reid,
        index=vector_index,
        store=store,
        sample_fps=float(os.getenv("SAMPLE_FPS", "2")),
    )
    graph = StaticTransitionGraph(
        (
            CameraTransition("cam01", "cam02", 10, 45),
            CameraTransition("cam01", "cam03", 15, 60),
        )
    )
    follow = LiveFollowService(
        graph=graph,
        reid=reid,
        index=vector_index,
        sink=EmptyFollowEventSink(),
        similarity_threshold=float(os.getenv("REID_MIN_SIMILARITY", "0.65")),
    )
    return WorkerRuntime(queue, indexer, LiveCameraService(detector, tracker, crop_selector, follow))


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-reid shared indexing worker")
    parser.add_argument("--once", action="store_true", help="process one queue batch and exit")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    try:
        runtime = build_runtime()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    if args.once:
        runtime.run_once(announce_empty=True)
    else:
        runtime.run_forever(args.poll_seconds)


if __name__ == "__main__":
    main()
