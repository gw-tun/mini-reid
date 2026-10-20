"""Stand-ins for the models, so pipeline tests run in a second without weights."""

from __future__ import annotations

import contextlib

import numpy as np

from services.worker.app.domain.models import Detection, Embedding, ObjectType

RED = (220, 30, 30)


def write_video(path, people, seconds=3.0, fps=10, size=(320, 240)):
    """`people`: colour -> (from_s, to_s, x_start). A 40x130 box walks right."""
    import av

    w, h = size
    with av.open(str(path), "w") as out:
        stream = out.add_stream("mpeg4", rate=fps)
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.bit_rate = 2_000_000
        for n in range(int(seconds * fps)):
            t = n / fps
            img = np.full((h, w, 3), 15, np.uint8)
            for colour, (start, end, x) in people.items():
                if start <= t < end:
                    left = int(x + 20 * (t - start))
                    img[60:190, left:left + 40] = colour
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


class ColourDetector:
    """Finds each coloured rectangle and calls it a person. Stands in for YOLO."""

    def __init__(self, colours=(RED,)):
        self.colours = colours

    def detect(self, frame):
        found = []
        for colour in self.colours:
            mask = np.abs(frame.image.astype(int) - colour).sum(axis=2) < 90
            if mask.sum() < 50:
                continue
            rows, cols = np.where(mask)
            found.append(Detection(ObjectType.PERSON, 0.9,
                                   (float(cols.min()), float(rows.min()),
                                    float(cols.max() + 1), float(rows.max() + 1)), "person"))
        return found


class CountingReID:
    """Records what it was asked to embed."""

    def __init__(self):
        self.calls = []

    def embed(self, crop):
        self.calls.append(crop)
        return Embedding("fake-space", (1.0,), "v0")


@contextlib.contextmanager
def throwaway_database():
    """A new, migrated database in the local Postgres -> its DSN; dropped afterwards.
    Skips the test when Postgres isn't running. The developer's data is never touched."""
    import uuid

    import psycopg
    import pytest

    from scripts.migrate import main as migrate
    from services.common.database import resolve_dsn

    base = resolve_dsn()
    try:
        admin = psycopg.connect(base, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("local Postgres is not running")
    name = f"mini_reid_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f"CREATE DATABASE {name}")
    try:
        dsn = base.rsplit("/", 1)[0] + "/" + name
        assert migrate(["--dsn", dsn]) == 0
        yield dsn
    finally:
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.close()
