# Services architecture

The services use dependency inversion so the workflow can be tested before real
YOLO, FastReID, PostgreSQL, or vector-search adapters are installed.

```text
MediaMTX segment complete
        ↓ HTTP webhook
services/api
        ↓ services/common FileJobQueue
services/worker
        ↓
RecordIndexingService / LiveFollowService
        ↓ ports
YOLO · tracker · crop selector · FastReID · vector index · event sink
```

## Record Search

`POST /internal/record-segments/complete` accepts a JSON body or MediaMTX query
parameters. It validates the event, derives an idempotent event ID when needed,
and writes a small JSON job into the shared `data/cache/job-queue` directory.
The API returns `202` quickly; it never runs inference inside the webhook.

Manual webhook test:

```powershell
curl.exe -X POST "http://localhost:8000/internal/record-segments/complete?camera=cam01&segment_path=/recordings/cam01/test.mp4&duration=300"
```

`infra/mediamtx/mediamtx.yml` calls this endpoint from
`runOnRecordSegmentComplete`. The MediaMTX container therefore needs a `curl`
binary (or the hook can be replaced with a small webhook relay image).

The worker atomically moves pending jobs to `processing`, runs
`RecordIndexingService`, acknowledges successful jobs, and moves failed jobs to
`failed` with the error text. Replace `FileJobQueue` with a PostgreSQL queue when
the database schema is ready.

## Live Follow

`LiveFollowService` receives a selected target and a directed camera transition
graph. It returns bounded downstream camera windows and evaluates candidate track
observations through the same ReID and vector-index ports. A T-junction can return
multiple windows; the graph schedules candidates while similarity and temporal
continuity decide the match.

`LiveCameraService` is the per-camera seam: frame → detector → local tracker →
quality crop → `LiveFollowService`. It does not create a model process per camera;
the runtime wires shared detector/ReID adapters into each camera loop.

The current adapters are intentionally empty. Implement `Detector`, `Tracker`,
`ReIDModel`, `VectorIndex`, and `FollowEventSink` separately so model changes do
not alter the application services.

## SOLID boundaries

- **Single Responsibility:** routes enqueue events; `RecordIndexingService`
  orchestrates a segment; adapters own decoding, detection, ReID, storage, or
  event delivery.
- **Open/Closed:** replace `EmptyReIDModel` with FastReID MSMT17/VeRi adapters
  without changing the use cases.
- **Liskov Substitution:** every adapter implements the small protocol used by
  the application service.
- **Interface Segregation:** detection, tracking, embedding, indexing, and event
  delivery are separate ports.
- **Dependency Inversion:** use cases depend on protocols; the concrete file
  queue and empty adapters are wired only in `worker.py` and API dependencies.
