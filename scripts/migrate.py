"""Apply infra/postgres/init/*.sql to a database that already exists.

    .venv/bin/python scripts/migrate.py --dry-run     # what would run; changes nothing
    .venv/bin/python scripts/migrate.py               # run it

Docker runs infra/postgres/init only on an EMPTY data directory, so a new
migration never reaches a database that already holds rows. This replays the
same files, in file-name order, against a live one -- the way semantic-va's
`tools/migrate.py` does. There is deliberately no second copy of the SQL:
every file there is written to be safe to run twice.

Each applied file is recorded in `schema_migrations` with the SHA-256 of its
contents. A file is run once; one whose contents changed after it was applied
is reported, not silently re-run -- editing an applied migration is how two
databases that both say "up to date" end up different. Add a new file instead.

Each file runs in its own transaction together with its record: it is applied
and recorded completely, or not at all.

Which database: --dsn, else $DATABASE_URL, else DATABASE_URL in .env, else the
docker-compose defaults (localhost:5434). Only point it at a local database.
"""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.common.database import redact, resolve_dsn  # noqa: E402  (also used by the tests)

MIGRATIONS = ROOT / "infra" / "postgres" / "init"

_TRACKING = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    name       TEXT PRIMARY KEY,
    sha256     TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""


def plan(files: list[pathlib.Path], applied: dict[str, str], force: str | None = None
         ) -> tuple[list[tuple[pathlib.Path, str]], list[str]]:
    """(files to run with their SHA-256, names of applied files that changed since)."""
    pending, changed = [], []
    for path in files:
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.name == force or path.name not in applied:
            pending.append((path, sha))
        elif applied[path.name] != sha:
            changed.append(path.name)
    return pending, changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dsn", help="postgresql://user:password@host:port/database")
    parser.add_argument("--dry-run", action="store_true", help="say what would run, change nothing")
    parser.add_argument("--force", metavar="NAME", help="re-apply one file although it is recorded")
    parser.add_argument("--dir", type=pathlib.Path, default=MIGRATIONS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    import psycopg

    sys.stdout.reconfigure(line_buffering=True)  # keep printed lines in order with errors
    dsn = resolve_dsn(args.dsn)
    print(f"database  {redact(dsn)}")
    try:
        conn = psycopg.connect(dsn, autocommit=True, connect_timeout=5)
    except psycopg.OperationalError as exc:
        print(f"cannot connect: {str(exc).strip().splitlines()[0]}\n"
              "is the database running?  docker compose up -d postgres", file=sys.stderr)
        return 2

    with conn:
        # A dry run must not write anything, so it doesn't create the record table either.
        exists = conn.execute("SELECT to_regclass('schema_migrations') IS NOT NULL").fetchone()[0]
        if not exists and not args.dry_run:
            conn.execute(_TRACKING)
        applied = dict(conn.execute("SELECT name, sha256 FROM schema_migrations").fetchall()) if exists else {}

        files = sorted(args.dir.glob("*.sql"))
        pending, changed = plan(files, applied, args.force)
        for name in changed:
            print(f"WARNING   {name} changed after it was applied; not re-running it. Add a new "
                  f"file instead (or --force {name} on a local database).")
        if not pending:
            print(f"up to date: {len(applied)} file(s) already applied")
            return 1 if changed else 0

        for path, sha in pending:
            if args.dry_run:
                print(f"would run {path.name}")
                continue
            print(f"running   {path.name}")
            try:
                # psycopg runs several statements in one execute() only without
                # parameters -- true of every migration, which is why none has any.
                with conn.transaction():
                    conn.execute(path.read_text(encoding="utf-8"))
                    conn.execute(
                        "INSERT INTO schema_migrations (name, sha256) VALUES (%s, %s) "
                        "ON CONFLICT (name) DO UPDATE SET sha256 = EXCLUDED.sha256, applied_at = now()",
                        (path.name, sha))
            except psycopg.Error as exc:
                print(f"FAILED    {path.name}: {str(exc).strip()}\n"
                      "          nothing from this file was kept; later files were not run",
                      file=sys.stderr)
                return 2

    verb = "would be run" if args.dry_run else "applied"
    print(f"{len(pending)} file(s) {verb}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
