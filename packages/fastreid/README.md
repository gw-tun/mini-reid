# Vendored FastReID v0.1.1

This directory vendors the `fastreid` package from
[JDAI-CV/fast-reid v0.1.1](https://github.com/JDAI-CV/fast-reid/tree/v0.1.1).

It is vendored because:

- The model weights and configs in `models/fastreid/` are the official v0.1.1
  release assets, so the v0.1.1 code is the matching implementation.
- The PyPI `fastreid` 1.4.0 release uses a different config schema
  (`INPUT.REA.MEAN` -> `INPUT.REA.VALUE`, `INPUT.FLIP` -> `INPUT.FLIP.ENABLED`,
  ...) and pins `torch==1.13.1`, which conflicts with the CPU/GPU torch used by
  the YOLO detector.

Install it into the worker environment with:

```powershell
python -m pip install -e .\packages\fastreid
```

The package is pure Python for inference. The optional `rank_cylib` Cython
extension is not built; the Python fallback is used for evaluation only and is
not needed by the ReID embedding path.

`fastreid.data.build` and `fastreid.engine.defaults` still import
`torch._six` and `collections.Mapping`, which were removed in modern torch and
Python. The worker installs a small compatibility shim
(`services/worker/app/infrastructure/fastreid_compat.py`) before importing
`fastreid`.
