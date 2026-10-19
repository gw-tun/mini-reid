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
```

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

## Planned operational scripts

- `migrate.py` — apply database migrations in order
- `backfill_recordings.py` — enqueue missing or failed segment-indexing jobs
- `validate_models.py` — check files, configs, dimensions, and model manifests
- `benchmark_reid.py` — measure embedding latency, throughput, and queue age
- `check_camera_config.py` — validate camera keys and transition-graph edges
