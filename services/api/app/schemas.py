"""API response models kept separate from route functions."""

from datetime import datetime

from pydantic import BaseModel, Field


class AcceptedJobResponse(BaseModel):
    status: str
    job_id: str
    duplicate: bool


class ReIDModelInfo(BaseModel):
    model_key: str
    backend: str
    feature_dim: int
    tracks: int
    """How many person tracks have barcodes from this model."""


class PersonCrop(BaseModel):
    uri: str
    """Under /crops/ (the web app reaches it as /api/crops/<uri>)."""
    quality: float
    observed_at: datetime


class PersonTrack(BaseModel):
    track_id: int
    camera_key: str
    local_track_key: str
    started_at: datetime
    ended_at: datetime | None
    seconds: float | None
    observations: int
    best_crop_uri: str | None
    best_quality: float | None
    models: list[str]
    """ReID models that have barcodes for this track."""


class PersonTrackDetail(PersonTrack):
    recording_uri: str | None
    crops: list[PersonCrop]


class SimilarTrack(BaseModel):
    track: PersonTrack
    similarity: float
    """Cosine of the two tracks' mean barcodes. Scales differ between models."""
    same_time: bool
    """Same camera, overlapping in time: certainly a different person."""


class SimilarTracksResponse(BaseModel):
    model_key: str
    model_version: str
    query_track_id: int
    results: list[SimilarTrack]


# -- Recordings and "Search similar" (api/recordings.py) -----------------------


class Recording(BaseModel):
    path: str
    """Under the downloads folder, e.g. G421/2018-03-13_17-05-10.mp4."""
    camera: str
    recorded_from: datetime | None
    seconds: float | None
    via: str | None
    """How it was downloaded: share, snippet (both full resolution) or api (960 px)."""
    width: int | None
    height: int | None
    processed: bool
    """YOLO, tracking and ReID have run on it."""
    people: int | None
    """People with a usable crop, once processed."""
    rank: int | None
    score: float | None
    semantic_va_clip_id: int | None


class RecordingGroup(BaseModel):
    at: str | None
    text: str | None
    """The semantic-va search that found these; None for files no search names."""
    predicate: str | None
    recordings: list[Recording]


class RecordingPeopleRequest(BaseModel):
    path: str


class RecordingPeople(BaseModel):
    path: str
    camera: str
    recorded_from: datetime | None
    processed_now: bool
    people: list[PersonTrack]


class RecordingSearchRequest(BaseModel):
    track_id: int
    model_key: str = "msmt17_sbs_r50_ibn"
    top_k: int = Field(30, ge=1, le=100)


class RecordingMatch(BaseModel):
    rank: int
    track: PersonTrack
    similarity: float
    """Cosine of mean barcodes, one model. Not a probability; scales differ between models."""
    same_time: bool
    """Same camera, overlapping the chosen person in time: certainly someone else."""
    path: str
    offset: float
    """Seconds into `path` where this person first appears: where to start playing."""


class RecordingSearchStatus(BaseModel):
    search_id: str
    status: str
    """queued, running, complete or failed."""
    stage: str | None
    done: float
    total: float
    error: str | None
    model_key: str
    query_track_id: int
    results: list[RecordingMatch]
