# Shared Python AI worker

The worker owns the shared YOLO detector, local tracker, person/vehicle FastReID
pipelines, segment indexing jobs, and route-aware Live Follow scheduling. Models
are loaded once per worker. Per-camera queues are bounded, and active-follow work
has priority over background indexing.
