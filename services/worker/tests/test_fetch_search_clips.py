"""scripts/fetch_search_clips.py against a fake semantic-va: a little HTTP
server answering /health, /search, /clips/{id}, /clips/{id}/recording and
/clips/{id}/snippet from a local video, as if that video sat on semantic-va's
share. The snippet is cut the way semantic-va cuts it (src/core/snippets.py):
packets copied from the keyframe before the start, with the lead reported."""

import datetime as dt
import http.server
import json
import pathlib
import shutil
import threading
import urllib.parse

import numpy as np
import pytest

from fakes import RED, write_video
from scripts.fetch_search_clips import main, pieces, snippet_seconds

ROOT = pathlib.Path(__file__).resolve().parents[3]
CLIP = ROOT / "data" / "recordings" / "test-stairs" / "2018-03-13_15-50-00.mp4"
STREAM = "school.2018-03-13.15-50-00.15-55-00.school.G336.r13"
SOURCE = "meva/school/2018-03-13.15-50-00.15-55-00.school.G336.r13.mp4"


def test_pieces_sit_on_a_ten_second_clock_grid():
    start = dt.datetime(2018, 3, 13, 15, 50, 3)
    # 186-188 s into the file, 4 s either side: 15:53:05 .. 15:53:15
    assert pieces(start, 186.0, 188.0, 4.0) == [(dt.datetime(2018, 3, 13, 15, 53, 0), 177.0, 10.0),
                                               (dt.datetime(2018, 3, 13, 15, 53, 10), 187.0, 10.0)]
    # Near the start of the file the first piece starts with the file
    assert pieces(start, 1.0, 3.0, 4.0) == [(start, 0.0, 7.0)]


def test_a_snippet_is_asked_for_just_enough_seconds():
    assert snippet_seconds(25.0, 20.0, 30.0) == 5.25
    assert snippet_seconds(25.0, 0.0, 30.0) == 25.25
    assert snippet_seconds(100.0, 20.0, 30.0) == 30.0   # semantic-va's maximum
    assert snippet_seconds(5.0, 5.0, 5.1) == 1.0        # and its minimum


def copy_snippet(source: pathlib.Path, frame_pts: float, seconds: float, out: pathlib.Path) -> float:
    """semantic-va's SnippetService._cut, condensed -> the lead."""
    import av

    begin, end = max(0.0, frame_pts - seconds), frame_pts + seconds
    with av.open(str(source)) as c:
        s = c.streams.video[0]
        tb, origin = float(s.time_base), s.start_time or 0
        c.seek(int(begin / tb) + origin, stream=s, any_frame=False, backward=True)
        with av.open(str(out), "w", format="mp4") as o:
            ostream = o.add_stream_from_template(s)
            first, lead = None, 0.0
            for packet in c.demux(s):
                if packet.pts is None:
                    continue
                at = float((packet.pts - origin) * tb)
                if at > end:
                    break
                if first is None:
                    first, lead = packet.pts, max(0.0, begin - at)
                packet.pts -= first
                packet.dts = packet.dts - first if packet.dts is not None else None
                packet.stream = ostream
                o.mux(packet)
    return lead


class FakeSemanticVa(http.server.BaseHTTPRequestHandler):
    """Hits 1 and 2 are overlapping windows of one recording; 3 is a live source."""
    video = CLIP
    hits = [{"id": 1, "stream_id": STREAM, "clip_start": 2.0, "clip_end": 4.0, "capture_ts": "2018-03-13T15:50:02",
             "score": 0.41},
            {"id": 2, "stream_id": STREAM, "clip_start": 3.0, "clip_end": 5.0, "capture_ts": "2018-03-13T15:50:03",
             "score": 0.39},
            {"id": 3, "stream_id": "live-cam-7", "clip_start": 0.0, "clip_end": 2.0, "capture_ts": None, "score": 0.2}]
    searches: list = []
    leads: list = []

    def log_message(self, *args):
        pass

    def reply(self, body: bytes, kind="application/json", headers=()):
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        assert self.path == "/search"
        type(self).searches.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.reply(json.dumps({"results": [{"hits": self.hits}]}).encode())

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/health":
            return self.reply(b'{"status": "ok"}')
        clip_id = int(url.path.split("/")[2])
        hit = next(h for h in self.hits if h["id"] == clip_id)
        frame_pts = hit["clip_end"] - 0.05  # semantic-va's thumbnail moment, inside the window
        q = urllib.parse.parse_qs(url.query)
        out = self.server.tmp / f"cut-{clip_id}.mp4"
        if url.path.endswith("/recording"):
            from services.worker.app.infrastructure.video_cut import cut_video

            cut_video(self.video, float(q["start"][0]), float(q["duration"][0]), out)
            return self.reply(out.read_bytes(), "video/mp4")
        if url.path.endswith("/snippet"):
            lead = copy_snippet(self.video, frame_pts, float(q["seconds"][0]), out)
            type(self).leads.append(lead)
            return self.reply(out.read_bytes(), "video/mp4", [("X-Lead-Seconds", f"{lead:.3f}")])
        live = hit["capture_ts"] is None
        self.reply(json.dumps({"hit": hit, "tracks": [], "has_media": not live, "frame_pts": frame_pts,
                               "source_url": None if live else SOURCE}).encode())


def serve(handler, tmp_path):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.tmp = tmp_path
    threading.Thread(target=server.serve_forever, daemon=True).start()
    storage = tmp_path / "share"
    (storage / SOURCE).parent.mkdir(parents=True)
    shutil.copy(handler.video, storage / SOURCE)
    handler.searches, handler.leads = [], []
    return server, f"http://127.0.0.1:{server.server_address[1]}", storage


@pytest.fixture()
def semantic_va(tmp_path):
    if not CLIP.is_file():
        pytest.skip("needs the stairwell test clip")
    server, api, storage = serve(FakeSemanticVa, tmp_path)
    yield api, storage
    server.shutdown()


def lines(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("source", ["share", "snippet", "api"])
def test_downloads_each_piece_once_and_records_where_it_came_from(semantic_va, tmp_path, source):
    api, storage = semantic_va
    out = tmp_path / "out"
    args = ["people walking", "--top-k", "3", "--api", api, "--storage", str(storage), "--out", str(out),
            "--source", source]
    assert main(args) == 0
    assert FakeSemanticVa.searches[0] == {"rules": [{"id": "q", "text": "people walking"}], "top_k": 3,
                                          "predicate": "has(person)"}
    # Both hits (2-4 s and 3-5 s, 4 s either side) fall in the camera's 15:50:00 piece.
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*.mp4")) == ["G336/2018-03-13_15-50-00.mp4"]
    piece = lines(out / "manifest.jsonl")
    assert len(piece) == 1 and piece[0]["via"] == source and piece[0]["camera"] == "G336"
    assert piece[0]["source_url"] == SOURCE and piece[0]["frames"] > 100  # 10 s at 16 fps
    search = lines(out / "searches.jsonl")[0]
    assert [h.get("files") for h in search["hits"]] == [["G336/2018-03-13_15-50-00.mp4"]] * 2 + [None]
    assert "live" in search["hits"][2]["skipped"]

    # The same search again: nothing new is downloaded, the search is still logged.
    assert main(args) == 0
    assert len(lines(out / "manifest.jsonl")) == 1 and len(lines(out / "searches.jsonl")) == 2


def test_auto_takes_full_resolution_snippets_when_the_share_is_not_there(semantic_va, tmp_path):
    api, _ = semantic_va
    out = tmp_path / "out"
    assert main(["x", "--api", api, "--storage", str(tmp_path / "not-mounted"), "--out", str(out)]) == 0
    assert [p["via"] for p in lines(out / "manifest.jsonl")] == ["snippet"]
    assert not list((out / ".snippets").glob("*"))  # the snippet itself isn't kept


def test_a_piece_is_readable_by_the_indexer_with_its_camera_and_time(semantic_va, tmp_path):
    from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader

    api, storage = semantic_va
    out = tmp_path / "out"
    assert main(["--predicate", "has(person)", "--api", api, "--storage", str(storage), "--out", str(out)]) == 0
    frames = list(PyAVSegmentReader().frames(str(out / "G336" / "2018-03-13_15-50-00.mp4"), 2))
    assert frames[0].camera_key == "G336"
    assert frames[0].captured_at.replace(tzinfo=None) == dt.datetime(2018, 3, 13, 15, 50, 0)
    assert len(frames) == 20


class LongFake(FakeSemanticVa):
    """A 40 s recording with one hit at 24-26 s, far enough in for a snippet to
    start after second 0 -- so it really opens early, at a keyframe."""
    hits = [{"id": 7, "stream_id": STREAM, "clip_start": 24.0, "clip_end": 26.0,
             "capture_ts": "2018-03-13T15:50:24", "score": 0.5}]


def box_x(image: np.ndarray) -> float:
    """Where the red box is: the mean column of its pixels. It moves 20 px a second."""
    red = (image[..., 0] > 150) & (image[..., 1] < 100)
    return float(np.nonzero(red)[1].mean())


def test_snippet_pieces_land_at_the_same_moments_as_pieces_cut_from_the_file(tmp_path):
    """The snippet opens at a keyframe before its start and says so; pieces cut
    from it must hold the same frames, at the same times, as pieces cut from
    the source itself. A box moving 2 px per frame makes any slip visible."""
    from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader

    video = tmp_path / "long.mp4"
    write_video(video, {RED: (15.0, 40.0, 0)}, seconds=40.0, fps=10, size=(320, 240))
    LongFake.video = video
    server, api, storage = serve(LongFake, tmp_path)
    try:
        for source in ("share", "snippet"):
            assert main(["x", "--pad", "1", "--api", api, "--storage", str(storage),
                         "--out", str(tmp_path / source), "--source", source]) == 0
    finally:
        server.shutdown()
    assert LongFake.leads and LongFake.leads[0] > 0  # the copy really did open early
    name = "G336/2018-03-13_15-50-20.mp4"  # 23-27 s falls in the 20-30 s piece
    a = list(PyAVSegmentReader().frames(str(tmp_path / "share" / name), 2))
    b = list(PyAVSegmentReader().frames(str(tmp_path / "snippet" / name), 2))
    assert [f.captured_at for f in a] == [f.captured_at for f in b] and len(a) == 20
    for fa, fb in zip(a, b):
        assert abs(box_x(fa.image) - box_x(fb.image)) < 1.0, fa.captured_at


def test_says_what_to_do_when_semantic_va_is_not_running(tmp_path, capsys):
    assert main(["x", "--api", "http://127.0.0.1:9", "--out", str(tmp_path)]) == 2
    assert "VPN" in capsys.readouterr().err


def test_refetch_replaces_pieces_already_here(semantic_va, tmp_path):
    api, storage = semantic_va
    out = tmp_path / "out"
    base = ["x", "--api", api, "--storage", str(storage), "--out", str(out)]
    assert main(base + ["--source", "api"]) == 0  # the 960 px kind
    assert main(base + ["--source", "snippet"]) == 0  # skipped: already here
    assert [p["via"] for p in lines(out / "manifest.jsonl")] == ["api"]
    assert main(base + ["--source", "snippet", "--refetch"]) == 0  # replaced, once for both hits
    assert [p["via"] for p in lines(out / "manifest.jsonl")] == ["api", "snippet"]
    assert len(list(out.rglob("*.mp4"))) == 1
