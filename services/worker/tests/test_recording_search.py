"""Search similar people across downloaded recordings, processing on demand.

A temporary downloads folder holds the two local test clips as cameras G1
(waiting room) and G2 (stairwell), with a searches.jsonl naming only G1.
Real YOLO and OSNet-AIN on CPU, a throwaway PostgreSQL database. Skipped
without the weights, the clips or a running local Postgres."""

import importlib.util
import json
import pathlib
import shutil

import pytest

from fakes import throwaway_database

ROOT = pathlib.Path(__file__).resolve().parents[3]
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
def library(tmp_path_factory):
    import psycopg

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
         "hits": [{"rank": 1, "clip_id": 101, "score": 0.4, "files": [f"G1/{NAME}"]}]}) + "\n")
    indexer = PersonIndexer(YoloDetector(device="cpu", object_types={ObjectType.PERSON}),
                            [OsnetAinReID(device="cpu")], data / "crops")
    with throwaway_database() as dsn:
        yield RecordingSearch(indexer, lambda: psycopg.connect(dsn), root, data), dsn


def cards(library) -> dict:
    engine, dsn = library
    import psycopg
    with psycopg.connect(dsn) as conn:
        groups = engine.listing(conn)
    return {c["path"]: c | {"group": g["text"]} for g in groups for c in g["recordings"]}


def test_lists_recordings_by_search_and_refuses_paths_outside_the_folder(library):
    engine, _ = library
    listed = cards(library)
    assert listed[f"G1/{NAME}"]["group"] == "people waiting" and listed[f"G1/{NAME}"]["rank"] == 1
    assert listed[f"G2/{NAME}"]["group"] is None  # on disk, but no search named it
    assert not any(c["processed"] for c in listed.values())
    for bad in ["../../crops/x.mp4", f"G1/../../{NAME}", "/etc/hosts", "G1/missing.mp4"]:
        with pytest.raises(FileNotFoundError):
            engine.path_of(bad)


def test_people_processes_the_recording_once_and_lists_who_is_in_it(library):
    engine, _ = library
    first = engine.people(f"G1/{NAME}")
    assert first["processed_now"] and first["camera"] == "G1" and len(first["people"]) >= 3
    assert all(p["best_crop_uri"] for p in first["people"])
    again = engine.people(f"G1/{NAME}")
    assert not again["processed_now"]
    assert [p["track_id"] for p in again["people"]] == [p["track_id"] for p in first["people"]]
    assert cards(library)[f"G1/{NAME}"]["processed"] and cards(library)[f"G1/{NAME}"]["people"] >= 3


def test_search_processes_the_rest_then_ranks_everyone_against_the_chosen_person(library):
    engine, _ = library
    query = engine.people(f"G1/{NAME}")["people"][0]
    stages = []
    results = engine.search(query["track_id"], "osnet_ain_x1_0_msmt17", top_k=50,
                            progress=lambda stage, done, total: stages.append((stage, done, total)))
    assert stages[0] == ("processing", 0, 1) and stages[-1][0] == "ranking"  # only G2 was left
    assert results and query["track_id"] not in [r["track_id"] for r in results]
    assert [r["similarity"] for r in results] == sorted((r["similarity"] for r in results), reverse=True)
    assert {r["path"] for r in results} == {f"G1/{NAME}", f"G2/{NAME}"}
    assert all(0 <= r["offset"] <= 10 for r in results)  # where to start playing, inside the file
    for r in results:  # a person can't be in two places at once
        if r["same_time"]:
            assert r["path"] == f"G1/{NAME}"

    stages.clear()  # a second search processes nothing
    engine.search(query["track_id"], "osnet_ain_x1_0_msmt17", progress=lambda *a: stages.append(a))
    assert stages == [("ranking", 0, 0)]
