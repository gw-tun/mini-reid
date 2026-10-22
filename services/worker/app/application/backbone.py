"""Framework-independent ReID backbone logic.

These classes deliberately do not import OpenCV, Torch, FastReID, or a database.
They define the rules that Record Search and Live Follow will share.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from ..domain.models import (
    CropSample,
    Embedding,
    MatchCandidate,
    ObjectType,
    TrackObservation,
)
from ..domain.ports import ReIDModel, VectorIndex


class EmbeddingContractError(ValueError):
    """Raised when a model returns a vector outside its registered contract."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    object_type: ObjectType
    vector_space: str
    dimension: int


class ModelRegistry:
    """Maps object classes to one explicit model/vector space."""

    def __init__(self, specs: Mapping[ObjectType, ModelSpec] | None = None) -> None:
        self._specs: dict[ObjectType, ModelSpec] = dict(specs or {})

    def register(self, spec: ModelSpec) -> None:
        if spec.dimension <= 0:
            raise ValueError("model dimension must be positive")
        self._specs[spec.object_type] = spec

    def resolve(self, object_type: ObjectType) -> ModelSpec:
        try:
            return self._specs[object_type]
        except KeyError as exc:
            raise KeyError(f"no ReID model registered for {object_type.value}") from exc


class EmbeddingService:
    """Validates, normalizes, and tags model output before indexing."""

    def __init__(self, registry: ModelRegistry, models: Mapping[ObjectType, ReIDModel]) -> None:
        self._registry = registry
        self._models = dict(models)

    def embed(self, crop: CropSample) -> Embedding:
        spec = self._registry.resolve(crop.object_type)
        try:
            model = self._models[crop.object_type]
        except KeyError as exc:
            raise KeyError(f"no model adapter for {crop.object_type.value}") from exc
        raw = model.embed(crop)
        if raw is None:
            raise EmbeddingContractError(f"model returned no embedding for {spec.key}")
        if raw.vector_space != spec.vector_space:
            raise EmbeddingContractError(
                f"model space {raw.vector_space!r} does not match {spec.vector_space!r}"
            )
        if len(raw.values) != spec.dimension:
            raise EmbeddingContractError(
                f"expected {spec.dimension} values, received {len(raw.values)}"
            )
        normalized = _l2_normalize(raw.values)
        return Embedding(
            vector_space=spec.vector_space,
            values=normalized,
            model_version=raw.model_version,
        )


def _l2_normalize(values: tuple[float, ...]) -> tuple[float, ...]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise EmbeddingContractError("zero-length embedding cannot be normalized")
    return tuple(value / norm for value in values)


@dataclass(frozen=True)
class CropQualityPolicy:
    min_confidence: float = 0.5
    min_person_width: float = 40
    min_person_height: float = 80
    min_vehicle_width: float = 60
    min_vehicle_height: float = 40

    def accepts(self, observation: TrackObservation) -> bool:
        x1, y1, x2, y2 = observation.bbox
        width = x2 - x1
        height = y2 - y1
        if observation.confidence < self.min_confidence or width <= 0 or height <= 0:
            return False
        if observation.object_type is ObjectType.PERSON:
            return width >= self.min_person_width and height >= self.min_person_height
        return width >= self.min_vehicle_width and height >= self.min_vehicle_height

    def score(self, observation: TrackObservation) -> float:
        x1, y1, x2, y2 = observation.bbox
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        return area * max(0.0, min(1.0, observation.confidence))


class RepresentativeCropSelector:
    """Keeps the best accepted crop for each local track."""

    def __init__(self, policy: CropQualityPolicy | None = None) -> None:
        self._policy = policy or CropQualityPolicy()
        self._best_scores: dict[tuple[str, str], float] = {}

    def select(self, observation: TrackObservation) -> CropSample | None:
        if not self._policy.accepts(observation):
            return None
        key = (observation.camera_key, observation.track_id)
        score = self._policy.score(observation)
        if score <= self._best_scores.get(key, -1.0):
            return None
        self._best_scores[key] = score
        return CropSample(
            track_id=observation.track_id,
            object_type=observation.object_type,
            image=observation.image,
            quality=score,
        )


@dataclass(frozen=True)
class MatchPolicy:
    min_similarity: float = 0.65
    min_margin: float = 0.05

    def accept(self, candidates: tuple[MatchCandidate, ...] | list[MatchCandidate]) -> MatchCandidate | None:
        ranked = sorted(candidates, key=lambda candidate: candidate.score, reverse=True)
        if not ranked or ranked[0].score < self.min_similarity:
            return None
        if len(ranked) > 1 and ranked[0].score - ranked[1].score < self.min_margin:
            return None
        return ranked[0]


@dataclass(frozen=True)
class IndexedVector:
    event_id: str
    camera_key: str
    track_id: str
    object_type: ObjectType
    observed_at: datetime
    segment_path: str
    embedding: Embedding


class InMemoryVectorIndex:
    """Exact cosine index used for backbone tests and the first MVP iteration."""

    def __init__(self) -> None:
        self._vectors: dict[tuple[str, str, str], IndexedVector] = {}

    def upsert(self, event, observation: TrackObservation, embedding: Embedding) -> None:
        key = (event.event_id, observation.track_id, embedding.vector_space)
        self._vectors[key] = IndexedVector(
            event_id=event.event_id,
            camera_key=observation.camera_key,
            track_id=observation.track_id,
            object_type=observation.object_type,
            observed_at=observation.observed_at,
            segment_path=event.segment_path,
            embedding=embedding,
        )

    def search(
        self,
        embedding: Embedding,
        *,
        camera_key: str | None = None,
        object_type: str | None = None,
        top_k: int = 20,
        exclude_track_db_id: int | None = None,
    ) -> tuple[MatchCandidate, ...]:
        results: list[MatchCandidate] = []
        for item in self._vectors.values():
            if item.embedding.vector_space != embedding.vector_space:
                continue
            if camera_key is not None and item.camera_key != camera_key:
                continue
            if object_type is not None and item.object_type.value != object_type:
                continue
            results.append(
                MatchCandidate(
                    camera_key=item.camera_key,
                    track_id=item.track_id,
                    score=_cosine(embedding.values, item.embedding.values),
                    segment_path=item.segment_path,
                    observed_at=item.observed_at,
                    object_type=item.object_type,
                )
            )
        results.sort(key=lambda candidate: candidate.score, reverse=True)
        return tuple(results[:top_k])


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        raise EmbeddingContractError("cannot compare embeddings with different dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
