# mini-reid — YOLO + FastReID CCTV VMS

`mini-reid` is a CPU-first multi-camera CCTV ReID MVP. It accepts hard-coded RTSP
camera paths through MediaMTX, records short segments, detects people and vehicles
with YOLO, creates class-specific FastReID embeddings, and supports Record Search
and route-aware Live Follow.

The Next.js application in `frontend/` is preserved as the product UI. The rest of
the repository follows the finalized service-oriented layout:

```text
mini-reid/
├── frontend/                 Next.js UI (preserved)
├── services/
│   ├── api/                  FastAPI HTTP and database boundary
│   └── worker/               shared YOLO, tracker, ReID, indexing, follow workers
├── packages/contracts/       API and event contract snapshots
├── infra/
│   ├── mediamtx/             hard-coded RTSP paths and recording settings
│   └── postgres/init/        pgvector extensions and database bootstrap
├── models/                   official YOLO/FastReID weights and configs (ignored)
├── data/                     recordings, crops, cache, exports, and logs (ignored)
├── scripts/                  migrations, backfills, benchmarks, validation
├── docs/                     finalized project and data/performance specifications
├── archive/                  previous learning scaffold and documents
├── docker-compose.yml        local PostgreSQL + MediaMTX infrastructure
└── index.html                interactive project briefing
```

See [services/README.md](services/README.md) for the webhook, queue, Record Search,
and Live Follow extension points.

## Final operating model

Record Search indexes each completed MediaMTX segment asynchronously. A midnight
maintenance job is reserved for retries, gap detection, backfill, and re-embedding.
Live Follow keeps a low-rate sentinel warm on every camera, then schedules higher-rate
ReID bursts only on predicted downstream cameras and bounded transition windows.
At a T-junction, candidate branches can run in parallel; shared workers and bounded
queues prevent one camera from creating a separate model process.

FastReID MSMT17 SBS R50-IBN is the initial person model and FastReID VeRi SBS
R50-IBN is the initial vehicle model. OSNet-AIN is a later comparison backend with
its own vector space. ReID scores rank candidates; crop quality, time continuity,
camera topology, and operator review determine a match.

## Prerequisites

- Node.js 20+ and npm for `frontend/`
- Python 3.11 for the service containers and local tools
- Docker Desktop with Compose
- CPU is the default development device; GPU acceleration is optional

Copy `.env.example` to `.env` before starting infrastructure:

```powershell
Copy-Item .env.example .env
```

Download the official YOLO11m and FastReID model/config assets:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

The command is safe to re-run: existing files are skipped and SHA-256 values are
printed. Use `-Force` to replace a file or `-VerifyOnly` to check the expected files.
See [scripts/README.md](scripts/README.md) for the complete model registry and sources.

## Start infrastructure

```powershell
docker compose up -d postgres mediamtx
docker compose ps
```

Create the versioned PostgreSQL schema and seed the ReID model registry:

```powershell
python .\scripts\migrate.py
```

This starts PostgreSQL with pgvector and MediaMTX. Replace the example camera URLs
in `infra/mediamtx/mediamtx.yml` with the approved hard-coded RTSP sources before
connecting real cameras.

## Run services locally

```powershell
# API
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r services/api/requirements.txt
pip install -r services/worker/requirements.txt
uvicorn services.api.app.main:app --reload --host 0.0.0.0 --port 8000

# Worker (another terminal using the same environment)
python -m services.worker.app.worker

# Frontend (another terminal)
Set-Location frontend
npm run dev
```

The API health endpoint is `http://localhost:8000/health`; the existing frontend
runs on its configured Next.js development port.

## Validation

```powershell
docker compose config
python -m compileall services
python -m unittest discover -s services/worker/tests -p "test_*.py"
Set-Location frontend
npm run lint
npm run build
```

Run a Record Search smoke test against PostgreSQL (the database must be up and
migrated; the CLI reads `REID_BACKEND=fastreid` and `DATABASE_URL` from `.env`.
PostgreSQL `vector(2048)` is the only vector store):

```powershell
docker compose up -d postgres
python .\scripts\migrate.py
python -m services.worker.app.cli index-video --video .\data\recordings\cam01\segment.mp4 --camera cam01
python -m services.worker.app.cli search-image --image .\data\query\person.jpg --object-type person
```

The equivalent HTTP flow is the webhook `POST /internal/record-segments/complete`
followed by the worker (`python -m services.worker.app.worker`) and
`POST /internal/record-search`.

Run the CPU-only path with one recorded segment and one camera first. Validate
crop quality, exact vector search, evidence links, and queue behavior before enabling
HNSW or multi-camera Live Follow.

## Documentation

- [Project specification](docs/PROJECT_SPEC.md)
- [Database and performance specification](docs/DATA_AND_PERFORMANCE.md)
- [Documentation map](docs/README.md)
- [Interactive briefing](index.html)
