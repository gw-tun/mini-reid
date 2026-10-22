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
├── osnet/
│   └── osnet_ain_x1_0_msmt17.pth   (VIP-9 comparison model, see below)
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

## OSNet-AIN (VIP-9)

`osnet/osnet_ain_x1_0_msmt17.pth` (17 MB) is the torchreid model-zoo checkpoint
"MSMT17 (combineall=True)", from
https://drive.google.com/file/d/1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal/view
(listed at https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO).
SHA-256 `8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f` --
`services/worker/app/reid/osnet.py` refuses any other file. Research-use
weights: see THIRD_PARTY_LICENSES.md.

## FastReID MSMT17 (person)

`fastreid/msmt17_sbs_R50-IBN.pth` (294 MB) from FastReID's v0.1.1 release
(`scripts/download_models.sh --only person`), SHA-256
`4e10cfb5dc667df0ff448838df6f9b56c9d10a95b5349bb9afebaf241a16de4a`. It is an old-format
(non-zip) checkpoint that also holds the optimizer and scheduler state; only
`model` is loaded (`weights_only=True`). `services/worker/app/reid/fastreid.py`
refuses any other file.

