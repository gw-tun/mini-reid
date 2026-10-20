"""The person API on a throwaway database with known barcodes.

A temporary database is created in the local Postgres (docker compose up -d
postgres), migrated with scripts/migrate.py, filled, and dropped afterwards;
the developer's own data is never touched. Skipped when Postgres isn't running."""

import datetime as dt
import os
import uuid

import pytest

from services.common.database import resolve_dsn

psycopg = pytest.importorskip("psycopg")
T0 = dt.datetime(2026, 10, 20, 8, 0, tzinfo=dt.timezone.utc)


def vec(*head, dim=512):
    return "[" + ",".join(str(v) for v in (list(head) + [0] * (dim - len(head)))) + "]"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    base = resolve_dsn()
    try:
        admin = psycopg.connect(base, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        pytest.skip("local Postgres is not running")
    name = f"mini_reid_test_{uuid.uuid4().hex[:8]}"
    admin.execute(f"CREATE DATABASE {name}")
    dsn = base.rsplit("/", 1)[0] + "/" + name
    try:
        from scripts.migrate import main as migrate
        assert migrate(["--dsn", dsn]) == 0
        with psycopg.connect(dsn) as conn:
            fill(conn)
        data = tmp_path_factory.mktemp("data")
        (data / "crops" / "cam1").mkdir(parents=True)
        (data / "crops" / "cam1" / "a.jpg").write_bytes(b"\xff\xd8 fake jpeg")
        (data / "secret.jpg").write_bytes(b"outside the crops folder")
        old = {k: os.environ.get(k) for k in ("DATABASE_URL", "DATA_DIR")}
        os.environ.update(DATABASE_URL=dsn, DATA_DIR=str(data))
        from fastapi.testclient import TestClient

        from services.api.app.main import app
        yield TestClient(app)
    finally:
        for k, v in locals().get("old", {}).items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
        admin.execute(f"DROP DATABASE IF EXISTS {name} WITH (FORCE)")
        admin.close()


def fill(conn):
    """Camera 1: tracks A (0-5 s) and B (2-6 s, overlaps A). Camera 2: track C (60-65 s).
    Model osnet v2 (current): A.B = 0.8, A.C = 0.6. Old version v1 ranks C first -- it must be ignored."""
    q = conn.execute
    q("INSERT INTO cameras (camera_key, display_name, mediamtx_path) VALUES ('cam1','cam1','cam1'), ('cam2','cam2','cam2')")
    tracks = {"A": (1, 0, 5), "B": (1, 2, 6), "C": (2, 60, 65)}
    for key, (cam, s, e) in tracks.items():
        q("INSERT INTO tracks (camera_id, source_kind, class_name, local_track_key, started_at, ended_at, best_crop_uri,"
          " quality_summary) VALUES (%s, 'recording', 'person', %s, %s, %s, 'cam1/a.jpg', %s)",
          (cam, key, T0 + dt.timedelta(seconds=s), T0 + dt.timedelta(seconds=e),
           '{"best_crops": [{"uri": "cam1/a.jpg", "quality": 0.9, "observed_at": "2026-10-20T08:00:01+00:00"}]}'))
    for version, created in (("v1", T0), ("v2", T0 + dt.timedelta(days=1))):
        q("INSERT INTO reid_models (model_key, backend, object_class, model_version, weights_uri, feature_dim,"
          " input_height, input_width, distance_metric, created_at) VALUES"
          " ('osnet', 'torchreid', 'person', %s, 'w', 512, 256, 128, 'cosine', %s)", (version, created))
    vectors = {"v2": {"A": vec(1, 0), "B": vec(0.8, 0.6), "C": vec(0.6, 0.8)},
               "v1": {"A": vec(1, 0), "B": vec(0, 1), "C": vec(1, 0)}}
    for model_id, version in ((1, "v1"), (2, "v2")):
        for track_id, key in enumerate("ABC", start=1):
            q("INSERT INTO reid_embeddings_512 (track_id, camera_id, class_name, model_id, model_key, model_version,"
              " observed_at, quality_score, embedding, sample_key) VALUES (%s, %s, 'person', %s, 'osnet', %s, %s, 0.9,"
              " %s::vector, %s)", (track_id, tracks[key][0], model_id, version, T0, vectors[version][key],
                                   f"{track_id}:{version}"))


def test_lists_person_tracks_with_their_models(client):
    tracks = client.get("/person/tracks").json()
    assert {t["local_track_key"] for t in tracks} == {"A", "B", "C"}
    assert all(t["models"] == ["osnet"] for t in tracks)
    assert client.get("/person/models").json() == [
        {"model_key": "osnet", "backend": "torchreid", "feature_dim": 512, "tracks": 3}]


def test_similar_ranks_by_cosine_with_the_current_model_version_only(client):
    body = client.get("/person/tracks/1/similar", params={"model_key": "osnet"}).json()
    assert body["model_version"] == "v2"
    ranked = [(r["track"]["local_track_key"], r["similarity"], r["same_time"]) for r in body["results"]]
    assert ranked == [("B", 0.8, True), ("C", 0.6, False)]  # B overlaps A on the same camera


def test_unknown_track_or_model_is_404(client):
    assert client.get("/person/tracks/999").status_code == 404
    assert client.get("/person/tracks/1/similar", params={"model_key": "nope"}).status_code == 404


def test_detail_lists_the_best_crops(client):
    detail = client.get("/person/tracks/1").json()
    assert [c["uri"] for c in detail["crops"]] == ["cam1/a.jpg"]


def test_serves_crops_and_nothing_outside_the_crops_folder(client):
    assert client.get("/crops/cam1/a.jpg").status_code == 200
    for path in ["../secret.jpg", "..%2Fsecret.jpg", "cam1/../../secret.jpg", "%2e%2e/secret.jpg"]:
        assert client.get("/crops/" + path).status_code == 404, path
