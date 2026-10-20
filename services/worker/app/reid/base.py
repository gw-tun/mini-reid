"""What every PyTorch ReID adapter does the same way (implements the `ReIDModel` port).

A subclass only says how to build its network from the weights file and how to
prepare one crop exactly as the model saw crops in training. This class:

- refuses a weights file whose SHA-256 differs from the model card: a different
  checkpoint makes barcodes of the same length that can't be compared with
  stored ones;
- loads with weights_only=True: unpickling the file runs no code from it;
- embeds in batches and makes every barcode length 1, so a dot product is a
  cosine;
- never describes another class (a person model returns None for a car);
- imports torch only when a model is built.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..device import pick_device
from ..domain.models import CropSample, Embedding
from .card import ReIDModelCard

ROOT = pathlib.Path(__file__).resolve().parents[4]
"""The repository: a card's relative weights path is under here, wherever the
process was started from."""


class TorchReIDModel:
    card: ReIDModelCard

    def __init__(self, weights: str | pathlib.Path | None = None, *, device: str | None = None,
                 verify: bool = True, batch_size: int = 32) -> None:
        import torch  # heavy: only when a model is built

        path = pathlib.Path(weights) if weights else ROOT / self.card.weights
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing: see models/README.md")
        if verify:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != self.card.sha256:
                raise ValueError(
                    f"{path.name} has SHA-256 {digest[:12]}..., the card expects {self.card.sha256[:12]}...: "
                    "a different checkpoint, whose barcodes can't be compared with stored ones")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        net = self.build(checkpoint)
        self.model_key = self.card.model_key
        self.model_version = self.card.sha256
        self.device = pick_device(device)
        self.net = net.eval().to(self.device)
        self.batch_size = batch_size

    def build(self, checkpoint: dict[str, Any]):
        """The network with the checkpoint's weights loaded (strict)."""
        raise NotImplementedError

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """One RGB crop (height, width, 3) -> float32 (3, height, width), as in training."""
        raise NotImplementedError

    def embed(self, crop: CropSample) -> Embedding | None:
        if crop.object_type is not self.card.object_class:
            return None
        return self.embed_batch([crop])[0]

    def embed_batch(self, crops: Sequence[CropSample]) -> list[Embedding]:
        """Crops of the card's class in batches: much faster than one at a time on a GPU."""
        return [Embedding(self.model_key, tuple(v.tolist()), self.model_version)
                for v in self.vectors([c.image for c in crops])]

    def vectors(self, images: Sequence[np.ndarray]) -> np.ndarray:
        """RGB crops -> (N, feature_dim) float32, every row of length 1."""
        import torch

        out = []
        for start in range(0, len(images), self.batch_size):
            batch = np.stack([self.preprocess(i) for i in images[start:start + self.batch_size]])
            with torch.inference_mode():
                out.append(self.net(torch.from_numpy(batch).to(self.device)).float().cpu().numpy())
        if not out:
            return np.zeros((0, self.card.feature_dim), np.float32)
        x = np.concatenate(out)
        return (x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)).astype(np.float32)
