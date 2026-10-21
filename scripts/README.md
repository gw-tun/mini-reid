# Project scripts

This directory contains one-off operational tools for the CCTV ReID MVP. Long-running
YOLO, tracking, ReID, and indexing loops belong in `services/worker`; scripts prepare,
validate, migrate, backfill, or benchmark those services.

## Download the official model assets

From the repository root, run this in PowerShell:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

The script downloads these files into the paths expected by the worker:

```text
models/yolo/yolo11m.pt
models/fastreid/msmt17_sbs_R50-IBN.pth
models/fastreid/veri_sbs_R50-IBN.pth
models/fastreid/configs/msmt17_sbs_R50-IBN.yml
models/fastreid/configs/veri_sbs_R50-IBN.yml
models/fastreid/Base-Strongerbaseline.yml
models/fastreid/Base-bagtricks.yml
```

The two `Base-*.yml` files are the FastReID config parents referenced by the
model configs (`_BASE_: "../Base-Strongerbaseline.yml"`). The `fastreid` package
itself is vendored under `packages/fastreid` and installed with
`python -m pip install -e .\packages\fastreid`.

It uses official release/repository URLs, skips files that already exist, and keeps
an incomplete download as `<file>.download` so a later run can resume it. Use
`-Force` to replace an existing file:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_models.ps1 -Force
```

Verify that all expected files exist and print their SHA-256 values without downloading:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_models.ps1 -VerifyOnly
```

The weights are ignored by Git. Record the printed SHA-256 values with the model
version used by a deployment. Check the applicable Ultralytics and FastReID licenses
before production use.

## Calibrate ReID thresholds from real footage

The seeded similarity thresholds are placeholders; derive them from a real
recording with tracker pseudo-labels (same track = genuine, different tracks =
impostor):

```powershell
python .\scripts\calibrate_thresholds.py --video data\recordings\cam01\segment.mp4 --apply
```

The script scans the video (YOLO + ReID embeddings), computes score
distributions, ROC/EER, and a top1/top2 margin, writes a JSON report to
`data/cache/threshold-calibration.json`, and with `--apply` updates the global
`reid_threshold_policies` rows in PostgreSQL. Suggested `REID_MIN_SIMILARITY` /
`REID_MIN_MARGIN` lines for `.env` are printed (the API, `search-image` CLI, and
live-follow worker read them).

Useful flags for montage recordings where people appear only in part of the
video:

- `--start-second/--end-second` — score only that time window
- `--sample-fps 4` — denser sampling so tracks survive long enough to form genuine pairs
- `--predict-batch 8` — batched YOLO calls (roughly halves CPU detection time)

The report records its caveats: track pseudo-labels are noisy (fragmentation
inflates impostor scores, a conservative bias), so re-run calibration when
multi-camera ground-truth identities are available.

## Planned operational scripts

Apply the PostgreSQL schema after starting the PG17 container:

```powershell
python .\scripts\migrate.py
```

The runner reads `DATABASE_URL` when set, otherwise it connects to the local
Compose mapping at `localhost:5434`. It records applied filenames in
`schema_migrations`, so re-running the command is safe.

- `migrate.py` — apply database migrations in order
- `calibrate_thresholds.py` — derive ReID thresholds from a real recording
- `backfill_recordings.py` — enqueue missing or failed segment-indexing jobs
- `validate_models.py` — check files, configs, dimensions, and model manifests
- `benchmark_reid.py` — measure embedding latency, throughput, and queue age
- `check_camera_config.py` — validate camera keys and transition-graph edges
