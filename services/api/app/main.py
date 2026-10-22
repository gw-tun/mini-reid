import os
import threading
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()

from .api.live import router as live_router
from .api.records import router as records_router
from .api.routes import router as internal_router
from .embedder_cache import embedder_from_env


def _warm_embedder() -> None:
    """Best-effort background load so the first search is not a cold 25s+ call."""

    try:
        embedder_from_env()
        print("embedder cache warmed")
    except Exception as exc:  # warmup must never block startup; lazy path retries
        print(f"embedder warmup failed (first search will retry): {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if os.getenv("API_WARMUP_EMBEDDER", "false").lower() in {"1", "true", "yes"}:
        threading.Thread(target=_warm_embedder, daemon=True, name="embedder-warmup").start()
    yield


app = FastAPI(title="mini-reid CCTV VMS API", version="0.1.0", lifespan=lifespan)
app.include_router(internal_router)
app.include_router(records_router)
app.include_router(live_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "api"}
