"""Track people in one video clip and show each track's best crops.

    .venv/bin/python scripts/track_clip.py VIDEO [--fps 2] [--device mps] [--reid osnet fastreid]
                     [--camera cam01 --start 2026-10-20T08:00:00]

Milestone M2 (docs/VIP-9-PERSON-PLAN.md): reader -> YOLO -> ByteTrack -> person
crop gate -> best crops per track. Writes to data/exports/track-<camera>-<clip>/:

    tracks.jsonl   one line per person track: id, start, end, observations,
                   crops sent to ReID, best quality
    tracks.jpg     one row per track: its best crops, best first -- check by eye
                   that each row is ONE person (two people in a row = an ID switch;
                   one person in two rows = fragmentation)

With --reid (milestone M3), every track also gets a barcode per model -- the
mean of its best crops' barcodes -- and every pair of tracks is compared, for
each model (osnet = OSNet-AIN, fastreid = FastReID MSMT17):

    similarity-<model>.jpg   the most similar pairs, best crops side by side. Two tracks
                     on screen at the same time are certainly different people,
                     so their score shows how high a WRONG match can go; one track
                     ending before the other starts could be one person whose
                     track broke (ReID should score those high).
    similarity-<model>.csv   every track against every other

Scores of two models are not on the same scale: compare how each model ranks
the pairs, and its gap between "possibly the same" and "certainly different".

Nothing goes to the database yet (that waits for the agreed migrations).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.worker.app.detection.yolo import DEFAULT_WEIGHTS, YoloDetector  # noqa: E402
from services.worker.app.domain.models import ObjectType  # noqa: E402
from services.worker.app.indexing.person_crops import PersonCropSelector  # noqa: E402
from services.worker.app.indexing.track_crops import TrackCropSelector  # noqa: E402
from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader, describe_segment  # noqa: E402
from services.worker.app.tracking.bytetrack import ByteTrackTracker  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("video", type=pathlib.Path)
p.add_argument("--fps", type=float, default=2.0)
p.add_argument("--device", help="cpu | mps | cuda:0 (default: DEVICE env, else auto)")
p.add_argument("--camera")
p.add_argument("--start", help="recording start (ISO time) if the file name doesn't give one")
p.add_argument("--weights", type=pathlib.Path, default=ROOT / DEFAULT_WEIGHTS)
p.add_argument("--reid", nargs="+", choices=["osnet", "fastreid"], default=[],
               help="also compare tracks with these ReID models")
args = p.parse_args()

start = dt.datetime.fromisoformat(args.start) if args.start else None
if start is not None and start.tzinfo is None:
    start = start.replace(tzinfo=dt.timezone.utc)
reader = PyAVSegmentReader(camera_key=args.camera, started_at=start)
detector = YoloDetector(args.weights, device=args.device, object_types={ObjectType.PERSON})
tracker = ByteTrackTracker()
crops = TrackCropSelector(PersonCropSelector())
print(f"detector {detector.model_key} on {detector.device} (people only), ByteTrack, {args.fps} fps")

frames = observations = embed_calls = 0
tracks: dict[str, dict] = {}


def close(ended) -> None:
    for e in ended:
        t = tracks.setdefault(e.track_id, {"embeds": 0})
        t.update(track=e.track_id, start=e.first_seen.isoformat(), end=e.last_seen.isoformat(),
                 seconds=round((e.last_seen - e.first_seen).total_seconds(), 1),
                 observations=e.observations, best=crops.end(e.track_id))


for frame in reader.frames(str(args.video), args.fps):
    frames += 1
    for o in tracker.update(frame, detector.detect(frame)):
        observations += 1
        if crops.select(o) is not None:
            embed_calls += 1
            tracks.setdefault(o.track_id, {"embeds": 0})["embeds"] += 1
    close(tracker.ended())
close(tracker.finish())

camera = args.camera or describe_segment(args.video)[0]
out = ROOT / "data" / "exports" / f"track-{camera}-{args.video.stem}"
out.mkdir(parents=True, exist_ok=True)
rows = sorted((t for t in tracks.values() if "track" in t), key=lambda t: t["start"])
gallery = [t for t in rows if t["best"]]


def together(a, b) -> bool:
    """On screen at the same time: certainly two different people."""
    return a["start"] <= b["end"] and b["start"] <= a["end"]


def compare(name: str):
    """(model key, track-by-track similarity, pairs best first) for one ReID model."""
    if name == "osnet":
        from services.worker.app.reid.osnet import OsnetAinReID as Model
    else:
        from services.worker.app.reid.fastreid import FastReIDPersonReID as Model
    reid = Model(ROOT / Model.card.weights, device=args.device)
    crops_n = sum(len(t["best"]) for t in gallery)
    began = time.perf_counter()
    # One barcode per track: the mean of its best crops' barcodes, length 1 again.
    means = np.stack([reid.vectors([c.image for c in t["best"]]).mean(0) for t in gallery])
    took = time.perf_counter() - began
    vectors = means / np.linalg.norm(means, axis=1, keepdims=True)
    similarity = vectors @ vectors.T
    print(f"\n{reid.model_key} on {reid.device}: {crops_n} crops in {took:.2f} s"
          f" ({took / crops_n * 1000:.0f} ms per crop, including warm-up)")
    pairs = []
    for i, a in enumerate(gallery):
        others = [j for j in range(len(gallery)) if j != i]
        j = max(others, key=lambda j: similarity[i, j])
        a.setdefault("most_similar", {})[name] = {"track": gallery[j]["track"],
                                                  "score": round(float(similarity[i, j]), 3)}
        pairs += [(float(similarity[i, j]), a, gallery[j], together(a, gallery[j])) for j in others if j > i]
    return reid.model_key, similarity, sorted(pairs, key=lambda x: -x[0])


results = {name: compare(name) for name in args.reid} if len(gallery) > 1 else {}
with open(out / "tracks.jsonl", "w") as f:
    for t in rows:
        f.write(json.dumps({k: v for k, v in t.items() if k != "best"}
                           | {"best_quality": round(t["best"][0].quality, 3) if t["best"] else None}) + "\n")

for name, (model_key, similarity, pairs) in results.items():
    with open(out / f"similarity-{name}.csv", "w") as f:
        f.write("track," + ",".join(t["track"] for t in gallery) + "\n")
        for i, t in enumerate(gallery):
            f.write(t["track"] + "," + ",".join(f"{v:.3f}" for v in similarity[i]) + "\n")
    top = pairs[:8]
    cell, mid = 110, 190
    sheet = Image.new("RGB", (cell * 4 + mid, cell * len(top)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, (score, a, b, same_time) in enumerate(top):
        y = r * cell
        for c, im in enumerate([x.image for x in a["best"][:2]] + [x.image for x in b["best"][:2]]):
            im = Image.fromarray(im)
            im.thumbnail((cell - 6, cell - 6))
            x = c * cell + (mid if c >= 2 else 0)
            sheet.paste(im, (x + (cell - im.width) // 2, y + 3))
        draw.text((2 * cell + 8, y + 10), f"{a['track']}  vs", fill="black")
        draw.text((2 * cell + 8, y + 26), b["track"], fill="black")
        draw.text((2 * cell + 8, y + 46), f"{name} {score:.2f}", fill="black")
        draw.text((2 * cell + 8, y + 66), "same time: different people" if same_time
                  else "one after the other: same?", fill=(200, 0, 0) if same_time else (0, 130, 0))
        draw.line([(0, y + cell - 1), (sheet.width, y + cell - 1)], fill=(220, 220, 220))
    sheet.save(out / f"similarity-{name}.jpg", quality=88)

# The contact sheet: one row per track, best crops first.
cell, label = 110, 150
with_crops = [t for t in rows if t["best"]]
sheet = Image.new("RGB", (label + cell * 4, cell * max(1, len(with_crops))), "white")
draw = ImageDraw.Draw(sheet)
for r, t in enumerate(with_crops):
    y = r * cell
    draw.text((6, y + 8), t["track"], fill="black")
    draw.text((6, y + 26), f"{t['seconds']} s, {t['observations']} obs", fill="gray")
    draw.text((6, y + 44), f"ReID crops: {t['embeds']}", fill="gray")
    for c, crop in enumerate(t["best"][:4]):
        im = Image.fromarray(crop.image)
        im.thumbnail((cell - 6, cell - 6))
        sheet.paste(im, (label + c * cell + (cell - im.width) // 2, y + 3))
    draw.line([(0, y + cell - 1), (sheet.width, y + cell - 1)], fill=(220, 220, 220))
sheet.save(out / "tracks.jpg", quality=88)

short = sum(1 for t in rows if t["observations"] < 3)
print(f"\n{frames} frames: {observations} person observations -> {len(rows)} person tracks"
      f" ({short} shorter than 3 observations)")
print(f"crops sent to ReID: {embed_calls} (instead of {observations}: one per track, refreshed every 4)")
print(f"tracks with a usable crop: {len(with_crops)}")
for name, (model_key, similarity, pairs) in results.items():
    print(f"\n{model_key}: most similar pairs (same time = certainly different people):")
    for score, a, b, same_time in pairs[:6]:
        print(f"  {score:.3f}  {a['track']:>16} vs {b['track']:<16}"
              f" {'same time' if same_time else 'one after the other'}")
    wrong = [p[0] for p in pairs if p[3]]
    maybe = [p for p in pairs if not p[3]]
    if wrong and maybe:
        best = maybe[0]
        print(f"  best 'one after the other' pair {best[1]['track']} -> {best[2]['track']}: {best[0]:.3f};"
              f" highest certainly-different pair: {max(wrong):.3f}; gap {best[0] - max(wrong):+.3f}")
files = ["tracks.jsonl", "tracks.jpg"] + [f"similarity-{n}.jpg/.csv" for n in results]
print(f"wrote {out.relative_to(ROOT)}/: " + ", ".join(files))
