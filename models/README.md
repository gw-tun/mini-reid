# Model assets

Use `scripts/download_models.ps1` to fetch the official MVP assets:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\download_models.ps1
```

The expected registry is:

```text
models/
├── yolo/
│   └── yolo11m.pt
└── fastreid/
    ├── msmt17_sbs_R50-IBN.pth
    ├── veri_sbs_R50-IBN.pth
    └── configs/
        ├── msmt17_sbs_R50-IBN.yml
        └── veri_sbs_R50-IBN.yml
```

Weights are intentionally ignored by Git. Keep the model source, version, SHA-256,
preprocessing, and vector-space identifier in the deployment/model registry. Do not
mix embeddings from different model families in one vector space.
