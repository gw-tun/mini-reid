from fastapi import FastAPI

from .api.routes import router as internal_router

app = FastAPI(title="mini-reid CCTV VMS API", version="0.1.0")
app.include_router(internal_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}
