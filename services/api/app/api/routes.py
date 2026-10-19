from __future__ import annotations

import json

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
