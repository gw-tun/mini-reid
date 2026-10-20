"""What makes a ReID barcode comparable with another: its model card.

Two barcodes can be compared only when the same model, the same weights and the
same preprocessing made them. A card records exactly that -- it is one row of
the `reid_models` table (infra/postgres/init/003_reid_embeddings.sql).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

from ..domain.models import ObjectType


@dataclass(frozen=True)
class ReIDModelCard:
    model_key: str                  # the vector space; also the HNSW index's WHERE in 003
    backend: str                    # "torchreid", "fastreid"
    object_class: ObjectType
    weights: pathlib.Path
    sha256: str                     # of the weights file: stored as model_version
    feature_dim: int
    input_height: int
    input_width: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
