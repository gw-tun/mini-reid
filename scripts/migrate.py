"""Apply versioned PostgreSQL migrations in lexical order."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[1]
MIGRATIONS = ROOT / "infra" / "postgres" / "migrations"


def database_url() -> str:
    value = os.getenv(
        "DATABASE_URL",
        "postgresql://mini_reid:change_me_for_local_dev@localhost:5434/mini_reid",
    )
    return value.replace("postgresql+psycopg://", "postgresql://", 1)


def apply_migrations() -> list[str]:
    applied: list[str] = []
    with psycopg.connect(database_url(), autocommit=True) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        existing = {
            row[0] for row in connection.execute("SELECT version FROM schema_migrations")
        }
        for migration in sorted(MIGRATIONS.glob("*.sql")):
            if migration.name in existing:
                continue
            try:
                connection.execute(migration.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"migration failed: {migration.name}: {exc}") from exc
            connection.execute(
                "INSERT INTO schema_migrations (version) VALUES (%s)", (migration.name,)
            )
            applied.append(migration.name)
    return applied


if __name__ == "__main__":
    completed = apply_migrations()
    print("applied: " + (", ".join(completed) if completed else "nothing"))
