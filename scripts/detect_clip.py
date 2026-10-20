"""Run person (and vehicle) detection on one video clip and save the evidence.

    .venv/bin/python scripts/detect_clip.py VIDEO [--fps 2] [--device mps]
                     [--camera cam01 --start 2026-10-19T08:30:00]

Milestone M1 (docs/VIP-9-PERSON-PLAN.md): reads the clip at --fps, runs YOLO,
routes every box by class, and judges each person crop with the person quality
gate. Writes to data/exports/detect-<camera>-<clip>/:

    detections.jsonl   one line per box: camera, time, class, label, score, bbox,
                       and for people: quality, rejection reason, crop file
    crops/             accepted person crops (the future ReID inputs)
    preview.jpg        the frame with the most people, boxes drawn:
                       green = accepted person, amber = rejected person, blue = vehicle

Nothing goes to the database yet (that waits for the agreed migrations).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import statistics
import sys
import time
from collections import Counter

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from services.worker.app.detection.yolo import DEFAULT_WEIGHTS, YoloDetector  # noqa: E402
from services.worker.app.domain.models import ObjectType  # noqa: E402
from services.worker.app.indexing.person_crops import judge_person_crop  # noqa: E402
from services.worker.app.infrastructure.pyav_reader import PyAVSegmentReader  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("video", type=pathlib.Path)
p.add_argument("--fps", type=float, default=2.0, help="sampled frames per second (spec: 2 on CPU)")
p.add_argument("--device", help="cpu | mps | cuda:0 (default: DEVICE env, else auto)")
p.add_argument("--camera", help="camera key, if the file name doesn't give one")
p.add_argument("--start", help="recording start (ISO time), if the file name doesn't give one")
p.add_argument("--weights", type=pathlib.Path, default=ROOT / DEFAULT_WEIGHTS)
args = p.parse_args()

start = dt.datetime.fromisoformat(args.start) if args.start else None
if start is not None and start.tzinfo is None:
    start = start.replace(tzinfo=dt.timezone.utc)
reader = PyAVSegmentReader(camera_key=args.camera, started_at=start)
detector = YoloDetector(args.weights, device=args.device)
print(f"detector {detector.model_key} on {detector.device}, sampling {args.fps} fps")

from services.worker.app.infrastructure.pyav_reader import describe_segment  # noqa: E402

camera_key = args.camera or describe_segment(args.video)[0]
out = ROOT / "data" / "exports" / f"detect-{camera_key}-{args.video.stem}"
(out / "crops").mkdir(parents=True, exist_ok=True)
for old in (out / "crops").glob("*.jpg"):  # a re-run replaces the previous one
    old.unlink()
counts: Counter[str] = Counter()
reasons: Counter[str] = Counter()
frames = 0
timings: list[float] = []
best_preview = None  # (people, image, boxes)

with open(out / "detections.jsonl", "w") as log:
    for frame in reader.frames(str(args.video), args.fps):
        frames += 1
        t0 = time.monotonic()
        detections = detector.detect(frame)
        timings.append(time.monotonic() - t0)
        drawn = []
        for i, d in enumerate(detections):
            counts[d.object_type.value] += 1
            row = {"camera": frame.camera_key, "time": frame.captured_at.isoformat(),
                   "class": d.object_type.value, "label": d.label,
                   "score": round(d.confidence, 4), "bbox": [round(v, 1) for v in d.bbox]}
            colour = (60, 130, 230)
            if d.object_type is ObjectType.PERSON:
                verdict = judge_person_crop(frame.image, d.bbox, d.confidence)
                row.update(quality=verdict.quality, rejected=verdict.reason)
                if verdict.reason is None:
                    x1, y1, x2, y2 = verdict.box
                    name = f"{frame.captured_at:%H%M%S%f}_{i}.jpg"
                    Image.fromarray(frame.image[y1:y2, x1:x2]).save(out / "crops" / name, quality=92)
                    row["crop"] = f"crops/{name}"
                    colour = (40, 200, 90)
                else:
                    reasons[verdict.reason] += 1
                    colour = (240, 170, 40)
            drawn.append((d.bbox, colour))
            log.write(json.dumps(row) + "\n")
        people = sum(1 for d in detections if d.object_type is ObjectType.PERSON)
        if best_preview is None or people > best_preview[0]:
            best_preview = (people, frame.image, drawn)

if best_preview:
    img = Image.fromarray(best_preview[1])
    draw = ImageDraw.Draw(img)
    for (x1, y1, x2, y2), colour in best_preview[2]:
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=3)
    img.save(out / "preview.jpg", quality=85)

people = counts["person"]
accepted = people - sum(reasons.values())
# Median, and the first frame left out: it includes loading the model onto the device.
steady = timings[1:] or timings
print(f"\n{frames} frames, median {statistics.median(steady) * 1000:.0f} ms per frame on {detector.device}"
      f" (first frame {timings[0] * 1000:.0f} ms incl. warm-up)")
print(f"person detections: {people}  (accepted for ReID: {accepted}"
      + "".join(f", {n} {r}" for r, n in reasons.most_common()) + ")")
print(f"vehicle detections: {counts['vehicle']}  (handed to the vehicle pipeline, VIP-8)")
print(f"\nwrote {out.relative_to(ROOT)}/: detections.jsonl, crops/ ({accepted}), preview.jpg")
