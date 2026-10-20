from fastapi import FastAPI

from .api.person import crops_router, router as person_router
from .api.recordings import router as recordings_router
from .api.routes import router as internal_router

app = FastAPI(title="mini-reid CCTV VMS API", version="0.1.0")
app.include_router(internal_router)
app.include_router(person_router)
app.include_router(crops_router)
app.include_router(recordings_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}
