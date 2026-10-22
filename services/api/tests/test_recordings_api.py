"""The recordings page's API, as the web app uses it: list -> people -> search
-> poll -> results, plus the video endpoint. Real YOLO and OSNet-AIN on CPU,
the two local test clips as cameras G1 and G2, a throwaway database.
Skipped without the weights, the clips or a running local Postgres."""

import importlib.util
import json
import os
import pathlib
import shutil
import sys
import time

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "services" / "worker" / "tests"))  # the shared fakes

from fakes import throwaway_database  # noqa: E402

CLIPS = {"G1": ROOT / "data/recordings/test-waiting/2018-03-13_15-50-00.mp4",
         "G2": ROOT / "data/recordings/test-stairs/2018-03-13_15-50-00.mp4"}
NAME = "2018-03-13_15-50-00.mp4"

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ultralytics") is None or importlib.util.find_spec("torch") is None
    or not (ROOT / "models/yolo/yolo11m.pt").is_file()
    or not (ROOT / "models/osnet/osnet_ain_x1_0_msmt17.pth").is_file()
    or not all(p.is_file() for p in CLIPS.values()),
    reason="needs YOLO, OSNet-AIN weights and both test clips")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import psycopg
    from fastapi.testclient import TestClient

    from services.api.app.main import app
    from services.api.app.search_runner import runner
    from services.worker.app.application.person_indexer import PersonIndexer
    from services.worker.app.application.recording_search import RecordingSearch
    from services.worker.app.detection.yolo import YoloDetector
    from services.worker.app.domain.models import ObjectType
    from services.worker.app.reid.osnet import OsnetAinReID

    data = tmp_path_factory.mktemp("data")
    root = data / "recordings" / "semantic-va"
    for camera, clip in CLIPS.items():
        (root / camera).mkdir(parents=True)
        shutil.copy(clip, root / camera / NAME)
    (root / "searches.jsonl").write_text(json.dumps(
        {"at": "2026-10-20T15:00:00", "text": "people waiting", "predicate": "has(person)",
         "hits": [{"rank": 1, "clip_id": 101, "score": 0.4, "files": [f"G1/{NAME}", f"G2/{NAME}"]}]}) + "\n")
    with throwaway_database() as dsn:
        old = {k: os.environ.get(k) for k in ("DATABASE_URL", "DATA_DIR")}
        os.environ.update(DATABASE_URL=dsn, DATA_DIR=str(data))
        indexer = PersonIndexer(YoloDetector(device="cpu", object_types={ObjectType.PERSON}),
                                [OsnetAinReID(device="cpu")], data / "crops")
        runner._engine = RecordingSearch(indexer, lambda: psycopg.connect(dsn), root, data)
        try:
            yield TestClient(app)
        finally:
            runner._engine = None
            for k, v in old.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_from_the_recording_list_to_ranked_similar_people(client):
    groups = client.get("/recordings").json()
    assert [g["text"] for g in groups] == ["people waiting"]
    assert [r["path"] for r in groups[0]["recordings"]] == [f"G1/{NAME}", f"G2/{NAME}"]
    assert not any(r["processed"] for r in groups[0]["recordings"])

    found = client.post("/recordings/people", json={"path": f"G1/{NAME}"}).json()
    assert found["processed_now"] and found["camera"] == "G1" and len(found["people"]) >= 3
    query = found["people"][0]["track_id"]

    started = client.post("/recordings/searches", json={"track_id": query, "model_key": "osnet_ain_x1_0_msmt17",
                                                        "top_k": 10})
    assert started.status_code == 202
    search_id = started.json()["search_id"]
    deadline = time.time() + 300
    while (body := client.get(f"/recordings/searches/{search_id}").json())["status"] in ("queued", "running"):
        assert time.time() < deadline
        time.sleep(0.5)
    assert body["status"] == "complete", body["error"]
    results = body["results"]
    assert results and [r["rank"] for r in results] == list(range(1, len(results) + 1))
    assert {r["path"] for r in results} <= {f"G1/{NAME}", f"G2/{NAME}"}
    assert client.get(f"/crops/{results[0]['track']['best_crop_uri']}").status_code == 200

    after = client.get("/recordings").json()[0]["recordings"]
    assert all(r["processed"] for r in after)  # the search processed G2 too


def test_video_plays_with_seeking_and_nothing_outside_the_folder(client):
    whole = client.get(f"/recordings/video/G1/{NAME}")
    assert whole.status_code == 200 and whole.headers["content-type"] == "video/mp4"
    part = client.get(f"/recordings/video/G1/{NAME}", headers={"Range": "bytes=0-99"})
    assert part.status_code == 206 and len(part.content) == 100
    for bad in ["../../crops", f"G1/../../{NAME}", "..%2F..%2Fmanifest.jsonl", "G1/nothing.mp4"]:
        assert client.get(f"/recordings/video/{bad}").status_code == 404, bad
    assert client.post("/recordings/people", json={"path": "../../etc/hosts"}).status_code == 404


def test_unknown_model_person_or_search_is_404(client):
    assert client.post("/recordings/searches", json={"track_id": 1, "model_key": "nope"}).status_code == 404
    assert client.post("/recordings/searches", json={"track_id": 999999,
                                                     "model_key": "osnet_ain_x1_0_msmt17"}).status_code == 404
    assert client.get("/recordings/searches/nothing").status_code == 404
