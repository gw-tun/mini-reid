"""API response models kept separate from route functions."""

from pydantic import BaseModel


class AcceptedJobResponse(BaseModel):
    status: str
    job_id: str
    duplicate: bool
