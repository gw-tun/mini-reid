"""Process-wide cache for heavy ReID embedding services.

FastReID model loading takes tens of seconds on CPU; building an
``EmbeddingService`` inside each search request made responses unacceptably
slow. The API hands every concurrent request the same built instance, keyed by
its full configuration, so configuration changes still get a fresh build while
repeated searches reuse the loaded models.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from services.worker.app.application.backbone import EmbeddingService
from services.worker.app.cli import build_embedding_service

_lock = threading.Lock()
_cache: dict[tuple[str, ...], EmbeddingService] = {}


def get_embedder(
    backend: str,
    *,
    device: str,
    person_weights: str,
    person_config: str,
    vehicle_weights: str,
    vehicle_config: str,
) -> EmbeddingService:
    """Return a process-wide embedder for this exact configuration.

    Builds on first use; failures propagate and are not cached so a missing
    weights file can be fixed without restarting the API.
    """

    key = (backend, device, person_weights, person_config, vehicle_weights, vehicle_config)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            return cached
        service = build_embedding_service(
            backend,
            person_weights=person_weights,
            person_config=person_config,
            vehicle_weights=vehicle_weights,
            vehicle_config=vehicle_config,
            device=device,
        )
        _cache[key] = service
        return service


def embedder_from_env() -> EmbeddingService:
    """Build (or reuse) the embedder described by the process environment."""

    model_root = Path(os.getenv("MODEL_ROOT", "/models" if Path("/models").is_dir() else "models"))
    return get_embedder(
        os.getenv("REID_BACKEND", "histogram"),
        device=os.getenv("DEVICE", "cpu"),
        person_weights=os.getenv("PERSON_REID_WEIGHTS", str(model_root / "fastreid" / "msmt17_sbs_R50-IBN.pth")),
        person_config=os.getenv("PERSON_REID_CONFIG", str(model_root / "fastreid" / "configs" / "msmt17_sbs_R50-IBN.yml")),
        vehicle_weights=os.getenv("VEHICLE_REID_WEIGHTS", str(model_root / "fastreid" / "veri_sbs_R50-IBN.pth")),
        vehicle_config=os.getenv("VEHICLE_REID_CONFIG", str(model_root / "fastreid" / "configs" / "veri_sbs_R50-IBN.yml")),
    )


def clear() -> None:
    """Drop every cached embedder (used by tests)."""

    with _lock:
        _cache.clear()
