"""The downloaded semantic-va search results, and "Search similar" on them.

    GET  /recordings                     the recordings, grouped by the search that found them
    GET  /recordings/video/{path}        one recording, for the browser's player
    POST /recordings/people              process one recording (first time) -> the people in it
    POST /recordings/searches            look for one of them in every recording (async)
    GET  /recordings/searches/{id}       progress, then the ranked results

The work is services/worker/app/application/recording_search.py; the
recordings come from scripts/fetch_search_clips.py.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from services.worker.app.application.recording_search import RecordingLibrary
from services.worker.app.infrastructure.pg_person_search import current_model

from ..dependencies import data_dir, downloads_dir, get_db
from ..schemas import (PersonTrack, RecordingGroup, RecordingMatch, RecordingPeople, RecordingPeopleRequest,
                       RecordingSearchRequest, RecordingSearchStatus)
from ..search_runner import Busy, runner

router = APIRouter(prefix="/recordings", tags=["recordings"])


def _library() -> RecordingLibrary:
    return RecordingLibrary(downloads_dir(), data_dir())


def _track(row: dict) -> PersonTrack:
    return PersonTrack(**{k: row[k] for k in PersonTrack.model_fields})


@router.get("", response_model=list[RecordingGroup])
def recordings(conn=Depends(get_db)) -> list[RecordingGroup]:
    return [RecordingGroup(**g) for g in _library().listing(conn)]


@router.get("/video/{path:path}")
def video(path: str) -> FileResponse:
    """An .mp4 inside the downloads folder and nothing else. Range requests work,
    so the player can seek."""
    try:
        file = _library().path_of(path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="no such recording") from exc
    return FileResponse(file, media_type="video/mp4")


@router.post("/people", response_model=RecordingPeople)
def people(request: RecordingPeopleRequest) -> RecordingPeople:
    """Run YOLO, tracking and ReID on the recording if not done yet (a few
    seconds; the very first call also loads the models), then list its people."""
    try:
        found = runner.engine().people(request.path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RecordingPeople(path=found["path"], camera=found["camera"], recorded_from=found["recorded_from"],
                           processed_now=found["processed_now"], people=[_track(p) for p in found["people"]])


@router.post("/searches", status_code=202)
def start_search(request: RecordingSearchRequest, conn=Depends(get_db)) -> dict:
    if current_model(conn, request.model_key) is None:
        raise HTTPException(status_code=404, detail=f"no person ReID model {request.model_key!r}")
    if conn.execute("SELECT 1 FROM tracks WHERE track_id = %s AND class_name = 'person'",
                    (request.track_id,)).fetchone() is None:
        raise HTTPException(status_code=404, detail=f"no person track {request.track_id}")
    try:
        return {"search_id": runner.submit(request.track_id, request.model_key, request.top_k)}
    except Busy as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc


@router.get("/searches/{search_id}", response_model=RecordingSearchStatus)
def search(search_id: str) -> RecordingSearchStatus:
    state = runner.searches.get(search_id)
    if state is None:
        raise HTTPException(status_code=404, detail="no such search (it may predate a restart)")
    results = [RecordingMatch(rank=n, track=_track(r), similarity=round(r["similarity"], 4),
                              same_time=r["same_time"], path=r["path"], offset=r["offset"])
               for n, r in enumerate(state["results"], start=1)]
    return RecordingSearchStatus(**{k: state[k] for k in ("search_id", "status", "stage", "done", "total",
                                                          "error", "model_key", "query_track_id")},
                                 results=results)
