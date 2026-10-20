"""Which device models run on: cpu, cuda:N, or mps (Apple GPU).

`DEVICE` in the environment decides; "auto" (or unset) picks CUDA, then MPS,
then CPU. CPU must always work (spec FR-12): torch is imported only here.
"""

from __future__ import annotations

import os


def pick_device(requested: str | None = None) -> str:
    choice = (requested or os.getenv("DEVICE") or "auto").strip().lower()
    if choice != "auto":
        return choice
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
