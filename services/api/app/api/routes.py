from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import ValidationError

from services.common.events import RecordSegmentEvent, RecordSegmentPayload
from services.common.job_queue import JobQueue

from ..dependencies import get_job_queue
from ..schemas import AcceptedJobResponse

router = APIRouter(prefix="/internal", tags=["internal-events"])


@router.post(
    "/record-segments/complete",
    response_model=AcceptedJobResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def record_segment_complete(
    request: Request,
    queue: JobQueue = Depends(get_job_queue),
) -> AcceptedJobResponse:
    """Accept a MediaMTX completion hook and enqueue indexing work quickly."""

    try:
        raw: dict[str, object] = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raw = dict(request.query_params)
    if not raw:
        raise HTTPException(status_code=400, detail="empty record segment event")

    # MediaMTX examples commonly use camera/duration query names; normalize them
    # before validating the shared event contract.
    normalized = dict(raw)
    normalized.setdefault("camera_key", normalized.get("camera"))
    normalized.setdefault("duration_seconds", normalized.get("duration"))
    try:
        payload = RecordSegmentPayload.model_validate(normalized)
        event = RecordSegmentEvent.from_payload(payload)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    receipt = queue.enqueue(event)
    return AcceptedJobResponse(
        status="duplicate" if receipt.duplicate else "accepted",
        job_id=receipt.job_id,
        duplicate=receipt.duplicate,
    )


@router.post("/record-search")
async def record_search(request: Request) -> dict[str, object]:
    """Search indexed videos with one query image path.

    The API container and worker share ``/data``. The upload/UI layer can place a
    query image there and pass its mounted path in the request body.
    """
    try:
        raw = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        raw = dict(request.query_params)
    image_path = raw.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise HTTPException(status_code=422, detail="image_path is required")
    object_type = str(raw.get("object_type", "person"))
    try:
        top_k = int(raw.get("top_k", 20))
        min_similarity = float(raw.get("min_similarity", os.getenv("REID_MIN_SIMILARITY", "0.65")))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="top_k and min_similarity must be numeric") from exc
    if top_k <= 0 or top_k > 500 or not 0 <= min_similarity <= 1:
        raise HTTPException(status_code=422, detail="top_k must be 1..500 and min_similarity must be 0..1")
    if object_type not in {"person", "vehicle"}:
        raise HTTPException(status_code=422, detail="object_type must be person or vehicle")

    try:
        import cv2
        from services.worker.app.application.record_search import RecordSearchService
        from services.worker.app.domain.models import ObjectType
        from services.worker.app.infrastructure.postgres import PostgresSearchResultStore, PostgresVectorIndex

        from ..embedder_cache import embedder_from_env

        image = cv2.imread(str(Path(image_path)))
        if image is None:
            raise FileNotFoundError(image_path)
        embedder = embedder_from_env()
        index = PostgresVectorIndex()
        result_store = PostgresSearchResultStore()
        matches = RecordSearchService(embedder, index, min_similarity, result_store=result_store).search(
            image, ObjectType(object_type), top_k
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"query image not found: {exc}") from exc
    except (ImportError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=f"search runtime unavailable: {exc}") from exc

    return {
        "query": {"image_path": image_path, "object_type": object_type},
        "matches": [
            {
                "camera_key": item.camera_key,
                "track_id": item.track_id,
                "score": item.score,
                "segment_path": item.segment_path,
                "observed_at": item.observed_at,
            }
            for item in matches
        ],
    }
