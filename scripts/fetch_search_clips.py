"""Download semantic-va search results into mini-reid, to run person ReID here.

    .venv/bin/python scripts/fetch_search_clips.py "people walking" [--top-k 30]
                     [--predicate "has(person)"] [--stream STREAM_ID ...] [--pad 4]
                     [--source auto|share|snippet|api] [--refetch] [--api URL]

Needs semantic-va's API reachable (VPN on); see --api. It only reads from semantic-va.

Writes, under data/recordings/semantic-va/ (git-ignored):

    <camera>/<YYYY-MM-DD_HH-MM-SS>.mp4   10-second pieces of video around each hit
    manifest.jsonl                       one line per piece: where it came from
    searches.jsonl                       one line per search: its hits and their pieces

Pieces sit on a fixed 10 s clock grid per camera (...:00, ...:10, ...). semantic-va's
2 s windows overlap, and a later search may find the same moment: both reuse the
piece already on disk instead of downloading it again. A file's name is its
recording time and its folder the camera (MEVA's code, e.g. G336), which is how
index_clip.py reads them.

Where the video comes from (--source; auto tries them in this order):
    share     the source file on the mounted share: full resolution
    snippet   semantic-va's /clips/{id}/snippet: the original video COPIED, so also
              full resolution, over HTTP; fetched once per hit and cut into pieces
              here, placed in time by `frame_pts` and its X-Lead-Seconds
    api       semantic-va's /clips/{id}/recording: re-encoded at most 960 px, where
              small people fail the 96 px crop gate -- the last resort

Next: .venv/bin/python scripts/index_clip.py data/recordings/semantic-va --device mps
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.worker.app.infrastructure.semantic_va import (  # noqa: E402
    SemanticVa, SemanticVaError, camera_of, file_started_at)

GRID = 10.0
"""Seconds per piece. semantic-va's /recording gives at most 10 s per request."""


def floor_to_grid(t: dt.datetime, grid: float = GRID) -> dt.datetime:
    midnight = t.replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight + dt.timedelta(seconds=math.floor((t - midnight).total_seconds() / grid) * grid)


def pieces(file_start: dt.datetime, clip_start: float, clip_end: float, pad: float,
           grid: float = GRID) -> list[tuple[dt.datetime, float, float]]:
    """The grid pieces covering `clip_start - pad` .. `clip_end + pad` (seconds into
    the file) -> [(recording time of the piece, seconds into the file, duration)].
    The first piece of a file starts at the file, not before it."""
    lo = file_start + dt.timedelta(seconds=max(0.0, clip_start - pad))
    hi = file_start + dt.timedelta(seconds=clip_end + pad)
    out, t = [], floor_to_grid(lo, grid)
    while t < hi:
        start = max(t, file_start)
        out.append((start, (start - file_start).total_seconds(), (t + dt.timedelta(seconds=grid) - start).total_seconds()))
        t += dt.timedelta(seconds=grid)
    return out


def local_source(storage: pathlib.Path, source_url: str) -> pathlib.Path | None:
    """The recording on the mounted share, if it is there (and inside it)."""
    path = (storage / source_url).resolve()
    return path if path.is_relative_to(storage.resolve()) and path.is_file() else None


def snippet_seconds(frame_pts: float, lo: float, hi: float) -> float:
    """How many seconds either side of `frame_pts` a snippet needs to hold
    `lo`..`hi` (seconds into the recording). semantic-va allows 1 to 30."""
    return min(30.0, max(1.0, max(frame_pts - lo, hi - frame_pts) + 0.25))


def fetch_piece(sva: SemanticVa, clip_id: int, source: pathlib.Path | None, source_offset: float, via: str,
                start: float, duration: float, out: pathlib.Path) -> dict | None:
    """Write one piece -> what was written, or None when that stretch has no video.

    `source`: a video holding the stretch, whose second 0 is `source_offset`
    seconds into the recording (the share's own file: 0; a snippet: where it
    opens). None: ask semantic-va's /recording for the stretch instead."""
    import av

    from services.worker.app.infrastructure.video_cut import cut_video

    part = out.with_name(out.stem + ".part.mp4")
    try:
        if source is not None:
            frames, width, height = cut_video(source, start - source_offset, duration, part)
        else:
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(sva.recording(clip_id, start, duration))
            with av.open(str(part)) as c:
                v = c.streams.video[0]
                width, height, frames = v.width, v.height, v.frames
        if not frames:
            return None
        part.replace(out)  # whole files only: an interrupted download leaves a .part, not a broken piece
        return {"via": via, "frames": frames, "width": width, "height": height, "bytes": out.stat().st_size}
    finally:
        part.unlink(missing_ok=True)


def append(path: pathlib.Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="?", help="what to search for, e.g. 'people walking' (optional with --predicate)")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--predicate", default="has(person)",
                   help="semantic-va detector filter (default: clips with a person); '' for none")
    p.add_argument("--stream", action="append", help="only this stream_id (repeatable)")
    p.add_argument("--pad", type=float, default=4.0, help="seconds of video kept either side of a hit")
    p.add_argument("--source", choices=["auto", "share", "snippet", "api"], default="auto")
    p.add_argument("--refetch", action="store_true",
                   help="download pieces again even if they are here (e.g. to replace 960 px ones)")
    p.add_argument("--api", help="semantic-va API, e.g. http://10.1.2.89:3000/api"
                                 " (default: $SEMANTIC_VA_API, else the client's default)")
    p.add_argument("--storage", type=pathlib.Path,
                   default=pathlib.Path(os.environ.get("SEMANTIC_VA_STORAGE", "/Volumes/ai/datasets")),
                   help="where semantic-va's share is mounted")
    p.add_argument("--out", type=pathlib.Path, default=ROOT / "data" / "recordings" / "semantic-va")
    args = p.parse_args(argv)
    if not args.text and not args.predicate:
        p.error("give a text, a --predicate, or both")

    sva = SemanticVa(args.api)
    try:
        sva.health()
    except SemanticVaError as exc:
        print(f"{exc}\nIs the VPN on, and is semantic-va's API at {sva.base}? (--api)", file=sys.stderr)
        return 2
    if args.source == "share" and not args.storage.is_dir():
        print(f"--source share, but {args.storage} isn't mounted: `open smb://10.1.2.90/ai`", file=sys.stderr)
        return 2

    request: dict = {"rules": [{"id": "q", "text": args.text}] if args.text else [], "top_k": args.top_k}
    if args.predicate:
        request["predicate"] = args.predicate
    if args.stream:
        request["streams"] = args.stream
    try:
        hits = sva.search(request)
    except SemanticVaError as exc:
        print(f"the search failed: {exc}", file=sys.stderr)
        return 2
    where = {"share": f"the share ({args.storage}), full resolution",
             "snippet": "semantic-va snippets, full resolution",
             "api": "semantic-va's /recording (at most 960 px)",
             "auto": "the share if mounted, else semantic-va snippets (both full resolution),"
                     " else /recording (960 px)"}[args.source]
    print(f"{len(hits)} hits for {args.text or ''!r} {args.predicate or ''}; video from {where}; into {args.out}")

    log, new, reused, written = [], 0, 0, 0
    via_count: dict[str, int] = {"share": 0, "snippet": 0, "api": 0}
    replaced: set[pathlib.Path] = set()
    for rank, hit in enumerate(hits, start=1):
        entry = {"rank": rank, "clip_id": hit["id"], "score": hit.get("score"), "stream_id": hit["stream_id"]}
        try:
            detail = sva.clip(hit["id"])
        except SemanticVaError as exc:
            log.append(entry | {"skipped": str(exc)})
            continue
        h = detail["hit"]
        start = file_started_at(h)
        if not detail.get("has_media") or not detail.get("source_url") or start is None:
            log.append(entry | {"skipped": "no recording (live source)"})
            print(f"  #{rank:<3} clip {h['id']:>8}  skipped: no recording")
            continue
        camera = camera_of(h["stream_id"], detail["source_url"])
        wanted = [(args.out / camera / f"{at:%Y-%m-%d_%H-%M-%S}.mp4", at, offset, duration)
                  for at, offset, duration in pieces(start, h["clip_start"], h["clip_end"], args.pad)]
        todo = list(wanted) if args.refetch else [w for w in wanted if not w[0].exists()]
        todo = [w for w in todo if w[0] not in replaced]  # --refetch: each piece once per run
        reused += len(wanted) - len(todo)

        # Where this hit's missing pieces come from: share, else snippet, else /recording.
        share = None if args.source in ("snippet", "api") else local_source(args.storage, detail["source_url"])
        via = "share" if share else "snippet" if args.source in ("auto", "snippet") else "api"
        if args.source == "share" and share is None:
            log.append(entry | {"skipped": f"{detail['source_url']} not on the share"})
            print(f"  #{rank:<3} clip {h['id']:>8}  skipped: not on the share at {args.storage}")
            continue
        source, source_offset, snippet = share, 0.0, None
        if todo and via == "snippet":
            if detail.get("frame_pts") is None:
                via = "api"
            else:
                frame_pts = float(detail["frame_pts"])
                seconds = snippet_seconds(frame_pts, min(w[2] for w in todo), max(w[2] + w[3] for w in todo))
                try:
                    data, lead = sva.snippet(h["id"], seconds)
                except SemanticVaError as exc:
                    if args.source == "snippet":
                        log.append(entry | {"skipped": f"snippet: {exc}"})
                        print(f"  #{rank:<3} clip {h['id']:>8}  skipped: {exc}")
                        continue
                    print(f"  ! clip {h['id']}: no snippet ({exc}); using /recording (960 px)")
                    via = "api"
                else:
                    snippet = args.out / ".snippets" / f"{h['id']}.mp4"
                    snippet.parent.mkdir(parents=True, exist_ok=True)
                    snippet.write_bytes(data)
                    source, source_offset = snippet, max(0.0, frame_pts - seconds) - lead
        if todo and via == "api" and args.source == "auto":
            print(f"  ! clip {h['id']}: video from /recording at 960 px -- small people will fail the crop gate")

        fresh = 0
        try:
            for path, at, offset, duration in todo:
                try:
                    info = fetch_piece(sva, h["id"], source, source_offset, via, offset, duration, path)
                except SemanticVaError as exc:
                    print(f"           piece {path.name}: {exc}", file=sys.stderr)
                    continue
                if info is None:
                    continue
                new, fresh, written = new + 1, fresh + 1, written + info["bytes"]
                via_count[via] += 1
                replaced.add(path)
                append(args.out / "manifest.jsonl", {
                    "file": str(path.relative_to(args.out)), "camera": camera, "recorded_from": at,
                    "recorded_to": at + dt.timedelta(seconds=duration), "stream_id": h["stream_id"],
                    "source_url": detail["source_url"], "source_start": offset, "duration": duration,
                    "first_clip_id": h["id"], "fetched_at": dt.datetime.now().isoformat(timespec="seconds"), **info})
        finally:
            if snippet is not None:
                snippet.unlink(missing_ok=True)  # the pieces are cut; the snippet itself isn't kept
        files = [str(w[0].relative_to(args.out)) for w in wanted if w[0].exists()]
        recorded = start + dt.timedelta(seconds=h["clip_start"])
        log.append(entry | {"camera": camera, "recorded_at": recorded, "clip_start": h["clip_start"],
                            "clip_end": h["clip_end"], "source_url": detail["source_url"], "files": files})
        print(f"  #{rank:<3} clip {h['id']:>8}  {camera:<8} {recorded:%Y-%m-%d %H:%M:%S}"
              f"  score {hit.get('score', 0):.3f}  -> {len(files)} pieces ({fresh} new, {via})")

    append(args.out / "searches.jsonl", {"at": dt.datetime.now().isoformat(timespec="seconds"), "text": args.text,
                                         "predicate": args.predicate, "top_k": args.top_k, "streams": args.stream,
                                         "request": request, "hits": log})
    skipped = sum(1 for e in log if "skipped" in e)
    print(f"\n{new} new pieces ({written / 1e6:.1f} MB): {via_count['share']} from the share and"
          f" {via_count['snippet']} from snippets at full resolution, {via_count['api']} at 960 px"
          f" from /recording; {reused} already here; {skipped} hits skipped")
    if via_count["api"] and args.source == "auto":
        print("WARNING: some video came at 960 px, where small people fail the 96 px crop gate.")
    print(f"next: .venv/bin/python scripts/index_clip.py {args.out.relative_to(ROOT) if args.out.is_relative_to(ROOT) else args.out} --device mps")
    return 0


if __name__ == "__main__":
    sys.exit(main())
