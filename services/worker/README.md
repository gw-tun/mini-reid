# Shared Python AI worker

The worker owns the shared YOLO detector, local tracker, person/vehicle FastReID
pipelines, segment indexing jobs, and route-aware Live Follow scheduling. Models
are loaded once per worker. Per-camera queues are bounded, and active-follow work
has priority over background indexing.

## Local Record Search MVP

The local command path supports video indexing and image queries. It samples video
frames, runs YOLO, assigns local track IDs, crops detections, embeds the crops, and
writes searchable evidence (camera, video path, track, and time) to the
PostgreSQL/pgvector gallery. PostgreSQL must be running with migrations applied
(`python .\scripts\migrate.py`); there is no file-based vector index.

```powershell
python -m services.worker.app.cli index-video --video .\data\recordings\cam01\segment.mp4 --camera cam01 --yolo .\models\yolo\yolo11m.pt
python -m services.worker.app.cli search-image --image .\data\query\person.jpg --object-type person
python -m services.worker.app.cli annotate-video --video .\data\recordings\cam01\segment.mp4 --camera cam01 --output .\data\exports\cam01-boxes.mp4 --yolo .\models\yolo\yolo11m.pt
python -m services.worker.app.cli monitor-live --source rtsp://localhost:8554/cam01 --camera cam01 --yolo .\models\yolo\yolo11m.pt
```

The database gallery is constrained to `vector(2048)`, so `index-video` and
`search-image` require the FastReID backend (`--reid-backend fastreid`, the default
from `.env`) and refuse `histogram`. The color-histogram embedder survives only as a
deterministic test double in unit tests.

To use the downloaded MSMT17/VeRi SBS R50-IBN assets, install the vendored FastReID
v0.1.1 runtime and choose the backend explicitly:

```powershell
python -m pip install -e .\packages\fastreid
python -m services.worker.app.cli index-video --video .\data\recordings\cam01\segment.mp4 --camera cam01 --yolo .\models\yolo\yolo11m.pt --reid-backend fastreid
python -m services.worker.app.cli search-image --image .\data\query\person.jpg --object-type person --reid-backend fastreid
```

The `fastreid` package is vendored under `packages/fastreid` because the PyPI
`fastreid` 1.4.0 release uses a different config schema and pins `torch==1.13.1`,
which conflicts with the torch used by the YOLO detector. FastReID v0.1.1 imports
`torch._six` and `collections.Mapping`, which were removed in modern torch and
Python; the worker installs a small shim
(`services/worker/app/infrastructure/fastreid_compat.py`) before importing it.

The FastReID model spaces are kept separate (`fastreid-msmt17-v1` for people and
`fastreid-veri-v1` for vehicles), so vectors from different model families are not
compared accidentally.

Install the worker dependencies before starting the queue worker. The worker checks
for `models/yolo/yolo11m.pt` (or `YOLO_WEIGHTS`) at startup and reports a setup
error when the detector runtime or model file is missing:

```powershell
python -m pip install -r services/worker/requirements.txt
python -m services.worker.app.worker --once
```

If `python` resolves to a different installation, use the venv interpreter
explicitly: `.\.venv\Scripts\python.exe -m pip install -r services/worker/requirements.txt`.

`--once` processes one batch and exits. The long-running command sleeps while the
queue is empty; `no pending jobs` is then an expected idle message. Ultralytics
settings are written under `data/cache/ultralytics` so the worker also works with
a read-only user profile or container.

For PostgreSQL/pgvector persistence, apply migrations first; the worker, CLI, and
API all read and write `DATABASE_URL`:

```powershell
python .\scripts\migrate.py
$env:REID_BACKEND = "fastreid"
python -m services.worker.app.worker
```

`REID_BACKEND=fastreid` is required: PostgreSQL `vector(2048)` is the only vector
store, and the worker exits with a configuration error for any other backend.
