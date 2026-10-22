from __future__ import annotations

from collections.abc import Sequence

from ..domain.models import CropSample, MatchCandidate, ObjectType
from ..domain.ports import ReIDModel, SearchResultStore, VectorIndex


class RecordSearchService:
    """Embeds one query image and returns ranked video/time evidence."""

    def __init__(self, reid: ReIDModel, index: VectorIndex, min_similarity: float = 0.65, result_store: SearchResultStore | None = None) -> None:
        self._reid = reid
        self._index = index
        self._min_similarity = min_similarity
        self._result_store = result_store

    def search(self, image, object_type: ObjectType = ObjectType.PERSON, top_k: int = 20) -> Sequence[MatchCandidate]:
        embedding = self._reid.embed(CropSample("query", object_type, image, 1.0))
        if embedding is None:
            return ()
        results = tuple(
            candidate for candidate in self._index.search(
                embedding, object_type=object_type.value, top_k=top_k
            ) if candidate.score >= self._min_similarity
        )
        if self._result_store is not None:
            self._result_store.save_search(object_type.value, embedding.vector_space, top_k, results)
        return results
