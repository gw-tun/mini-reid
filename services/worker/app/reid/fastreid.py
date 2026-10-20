"""FastReID MSMT17 SBS R50-IBN person ReID (the spec's first person model):
a crop becomes 2048 numbers.

The network is fastreid_sbs.py (FastReID v0.1.1, inference only, identical
arithmetic). Preprocessing follows FastReID's test pipeline exactly:

- RGB, resized to 384x128 (height x width) with BICUBIC interpolation
  (`T.Resize(SIZE_TEST, interpolation=3)`), aspect ratio not kept;
- pixels stay on the 0-255 scale (FastReID's own ToTensor does not divide by
  255); the network subtracts the checkpoint's `pixel_mean` and divides by its
  `pixel_std` (ImageNet's, times 255);
- the output is the BNNeck feature, L2-normalised here (FastReID normalises
  before comparing, too). Unlike OSNet's, these features can be negative.

Weights: models/fastreid/msmt17_sbs_R50-IBN.pth (scripts/download_models.sh
--only person). The file also holds the training state (optimizer, scheduler),
which is never loaded. Licences: THIRD_PARTY_LICENSES.md.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

from ..domain.models import ObjectType
from .base import TorchReIDModel
from .card import ReIDModelCard

FASTREID_MSMT17_SBS_R50_IBN = ReIDModelCard(
    model_key="msmt17_sbs_r50_ibn",  # = the HNSW index's WHERE in 003_reid_embeddings.sql
    backend="fastreid",
    object_class=ObjectType.PERSON,
    weights=pathlib.Path("models/fastreid/msmt17_sbs_R50-IBN.pth"),
    sha256="4e10cfb5dc667df0ff448838df6f9b56c9d10a95b5349bb9afebaf241a16de4a",
    feature_dim=2048,
    input_height=384,
    input_width=128,
    mean=(123.675, 116.28, 103.53),  # on the 0-255 scale; checked against the checkpoint
    std=(58.395, 57.12, 57.375),
)


def preprocess(image: np.ndarray, card: ReIDModelCard = FASTREID_MSMT17_SBS_R50_IBN) -> np.ndarray:
    """One RGB crop (height, width, 3) -> float32 (3, 384, 128) on the 0-255 scale."""
    pil = Image.fromarray(image).convert("RGB")
    pil = pil.resize((card.input_width, card.input_height), Image.Resampling.BICUBIC)
    return np.asarray(pil, dtype=np.float32).transpose(2, 0, 1)


class FastReIDPersonReID(TorchReIDModel):
    card = FASTREID_MSMT17_SBS_R50_IBN

    def build(self, checkpoint):
        import torch

        from .fastreid_sbs import SBSPersonNet

        state = checkpoint["model"]
        # The identity classifier (1041 MSMT17 training people) is for training only.
        state = {k: v for k, v in state.items() if not k.startswith("heads.classifier.")}
        for name, expected in (("pixel_mean", self.card.mean), ("pixel_std", self.card.std)):
            if not torch.allclose(state[name].flatten(), torch.tensor(expected), atol=1e-3):
                raise ValueError(f"checkpoint {name} {state[name].flatten().tolist()} differs from the card")
        net = SBSPersonNet()
        net.load_state_dict(state, strict=True)
        return net

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess(image, self.card)
