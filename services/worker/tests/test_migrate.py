"""scripts/migrate.py's decision: which migration files to run. No database needed."""

import hashlib

from scripts.migrate import plan, redact, resolve_dsn


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def files(tmp_path):
    a, b = tmp_path / "001_a.sql", tmp_path / "002_b.sql"
    a.write_text("CREATE TABLE IF NOT EXISTS a (id int);")
    b.write_text("CREATE TABLE IF NOT EXISTS b (id int);")
    return [a, b]


def test_new_files_run_and_applied_ones_are_skipped(tmp_path):
    a, b = files(tmp_path)
    pending, changed = plan([a, b], {a.name: sha(a)})
    assert [p.name for p, _ in pending] == ["002_b.sql"] and changed == []


def test_a_file_edited_after_it_was_applied_is_reported_not_rerun(tmp_path):
    a, b = files(tmp_path)
    applied = {a.name: sha(a), b.name: sha(b)}
    a.write_text("CREATE TABLE IF NOT EXISTS a (id bigint);")
    assert plan([a, b], applied) == ([], ["001_a.sql"])


def test_force_reruns_one_file(tmp_path):
    a, b = files(tmp_path)
    pending, _ = plan([a, b], {a.name: sha(a), b.name: sha(b)}, force="001_a.sql")
    assert [p.name for p, _ in pending] == ["001_a.sql"]


def test_dsn_and_password_hiding(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:secret@localhost:5434/d")
    dsn = resolve_dsn(None)
    assert dsn == "postgresql://u:secret@localhost:5434/d"   # psycopg's spelling
    assert "secret" not in redact(dsn) and "u:***@localhost" in redact(dsn)
    assert resolve_dsn("postgresql://x@h/d") == "postgresql://x@h/d"  # --dsn wins
