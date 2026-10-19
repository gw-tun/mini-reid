"""Entry point for the shared AI worker."""

from __future__ import annotations

import argparse
import os
import time

from services.common.job_queue import FileJobQueue

from .application.live_camera import LiveCameraService
from .application.live_follow import LiveFollowService
from .application.record_indexer import RecordIndexingService
from .domain.models import CameraTransition, Frame, FollowSession
from .infrastructure.adapters import (
    EmptyCropSelector,
    EmptyDetector,
    EmptyReIDModel,
    EmptyRecordingStore,
    EmptySegmentReader,
    EmptyTracker,
    EmptyVectorIndex,
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

    def run_once(self, batch_size: int = 4) -> int:
        jobs = self._queue.claim(limit=batch_size)
        if not jobs:
            print("no pending jobs")
            return 0
        for job in jobs:
            try:
                result = self._indexer.index_segment(job.event)
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
        try:
            while True:
                processed = self.run_once()
                if processed == 0:
                    time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("worker stopped")


def build_runtime() -> WorkerRuntime:
    queue = FileJobQueue(os.getenv("JOB_QUEUE_DIR", "data/cache/job-queue"))
    detector = EmptyDetector()
    tracker = EmptyTracker()
    crop_selector = EmptyCropSelector()
    reid = EmptyReIDModel()
    vector_index = EmptyVectorIndex()
    indexer = RecordIndexingService(
        reader=EmptySegmentReader(),
        detector=detector,
        tracker=tracker,
        crop_selector=crop_selector,
        reid=reid,
        index=vector_index,
        store=EmptyRecordingStore(),
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
    )
    return WorkerRuntime(queue, indexer, LiveCameraService(detector, tracker, crop_selector, follow))


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-reid shared indexing worker")
    parser.add_argument("--once", action="store_true", help="process one queue batch and exit")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args()
    runtime = build_runtime()
    if args.once:
        runtime.run_once()
    else:
        runtime.run_forever(args.poll_seconds)


if __name__ == "__main__":
    main()
