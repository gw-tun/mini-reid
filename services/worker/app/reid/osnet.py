"""OSNet-AIN person ReID (VIP-9's comparison model): a crop becomes 512 numbers.

Same network, weights and preprocessing as mini-person-reid's
`backend/app/model.py`, where it was tried on MEVA; the vectors are identical.

- PREPROCESSING MATCHES TRAINING, OR THE VECTORS ARE QUIETLY WRONG: RGB (crops
  are RGB already), stretched to 256x128 with bilinear resizing, scaled to 0-1,
  normalised with ImageNet's mean and std.
- Every feature is >= 0 (the network ends in ReLU), so every cosine is >= 0 and
  0.5 does not mean "unrelated". Thresholds come from labelled data.

Weights: models/osnet/ (models/README.md). Licences: THIRD_PARTY_LICENSES.md.
Loading, batching and checks: base.py.
"""

from __future__ import annotations

import pathlib

import numpy as np
from PIL import Image

from ..domain.models import ObjectType
from .base import TorchReIDModel
from .card import ReIDModelCard

OSNET_AIN_X1_0_MSMT17 = ReIDModelCard(
    model_key="osnet_ain_x1_0_msmt17",
    backend="torchreid",
    object_class=ObjectType.PERSON,
    weights=pathlib.Path("models/osnet/osnet_ain_x1_0_msmt17.pth"),
    sha256="8a07e8da38946f7cee37f4561617bf8b6d2fe8f3a4027852893ea092e46d919f",
    feature_dim=512,
    input_height=256,
    input_width=128,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)


def preprocess(image: np.ndarray, card: ReIDModelCard = OSNET_AIN_X1_0_MSMT17) -> np.ndarray:
    """One RGB crop (height, width, 3) -> float32 (3, 256, 128), as in training."""
    pil = Image.fromarray(image).convert("RGB")
    pil = pil.resize((card.input_width, card.input_height), Image.Resampling.BILINEAR)
    x = np.asarray(pil, dtype=np.float32) / 255.0
    x = (x - np.array(card.mean, np.float32)) / np.array(card.std, np.float32)
    return x.transpose(2, 0, 1)  # height, width, channel -> channel, height, width


class OsnetAinReID(TorchReIDModel):
    card = OSNET_AIN_X1_0_MSMT17

    def build(self, checkpoint):
        from .osnet_ain import osnet_ain_x1_0

        state = checkpoint.get("state_dict", checkpoint)
        # Saved from a DataParallel wrapper, so every key starts with "module.".
        state = {k.removeprefix("module."): v for k, v in state.items()}
        net = osnet_ain_x1_0(num_classes=state["classifier.weight"].shape[0])
        # strict: a wrong architecture fails here instead of making quiet nonsense.
        net.load_state_dict(state, strict=True)
        return net

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return preprocess(image, self.card)
