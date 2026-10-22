"""Resolve media paths across container and host layouts.

MediaMTX fires webhooks with container paths (``/recordings/cam01/...``) while
the local worker and API run on the host where the same tree lives under
``data/recordings/...``. Both sides resolve through this helper so a single
event is indexed exactly once against the same file.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_media_path(path: str, *, recordings_root: str | Path | None = None) -> str:
    """Return the first existing filesystem location for a media path.

    Tries the path as given, then maps container-style prefixes onto the
    host recordings root (``data/recordings`` by default). When nothing
    exists yet the original path is returned unchanged so callers can still
    raise a precise open error.
    """

    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    root = Path(recordings_root) if recordings_root is not None else Path(
        os.getenv("RECORDINGS_DIR", "data/recordings")
    )
    name = candidate.name
    parent = candidate.parent.name
    if parent:
        mapped = root / parent / name
        if mapped.is_file():
            return str(mapped)
    direct = root / name
    if direct.is_file():
        return str(direct)
    return path
