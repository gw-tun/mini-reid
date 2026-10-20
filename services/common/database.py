"""Which PostgreSQL database to use, decided the same way everywhere.

Precedence: an explicit DSN, else $DATABASE_URL, else DATABASE_URL in the
repository's .env, else the docker-compose defaults (localhost:5434). The
password in those defaults is the committed local-development one; a real
password belongs in .env (git-ignored) or the environment, never in code.
"""

from __future__ import annotations

import os
import pathlib
from urllib.parse import urlsplit, urlunsplit

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_DSN = "postgresql://mini_reid:change_me_for_local_dev@localhost:5434/mini_reid"

# Barcode table and pgvector type per vector size (infra/postgres/init/003).
# Code picks from here; a table name is never built from user input.
EMBEDDING_TABLES = {2048: ("reid_embeddings_2048", "halfvec"), 512: ("reid_embeddings_512", "vector")}


def resolve_dsn(dsn: str | None = None) -> str:
    if not dsn:
        dsn = os.environ.get("DATABASE_URL")
    if not dsn and (ROOT / ".env").is_file():
        for line in (ROOT / ".env").read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "DATABASE_URL":
                dsn = value.strip()
    # "postgresql+psycopg://" is SQLAlchemy's spelling; psycopg wants "postgresql://".
    return (dsn or DEFAULT_DSN).replace("postgresql+psycopg://", "postgresql://", 1)


def redact(dsn: str) -> str:
    """The DSN with the password hidden, for printing and logs."""
    parts = urlsplit(dsn)
    if not parts.password:
        return dsn
    user, _, hostport = parts.netloc.rpartition("@")
    return urlunsplit((parts.scheme, f"{user.split(':', 1)[0]}:***@{hostport}",
                       parts.path, parts.query, parts.fragment))
