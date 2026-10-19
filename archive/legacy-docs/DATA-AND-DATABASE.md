# Getting test data from semantic-va and saving it in a database

This guide answers two questions:

1. **How do I download scenes from semantic-va to test mini-reid?**
2. **How do I save search results in a database?** (your teacher's request)

Read [FOUNDATIONS.md](FOUNDATIONS.md) chapters 2, 7 and 10 first if video files,
vectors or SQL are new to you.

> **Note:** these commands were written from semantic-va's code and settings while
> its services were stopped, so they haven't been run against your data yet. If a
> command fails, keep the full error message: it usually says exactly what's wrong.

---

## Part 1. What you need from semantic-va

| You need | Why | Where it comes from |
|---|---|---|
| **Person crops** (small JPEGs) | ReID compares people's appearance | The crop endpoint, one crop per track |
| **Track facts** (camera, start and end time) | To know where and when each person was seen | `GET /clips/{id}` → `tracks` |
| **Full videos from 2–4 cameras at the same time** | Live tracking needs real video to run a detector on | Source files on the share |

semantic-va does **not** store a box for every frame (the `detections` table is
off by default). So for live tracking you have to run your own detector on the
video files. That's why you need the videos and not just the crops.

### The footage to use: MEVA

MEVA is footage from many fixed cameras at the same sites (`school`, `bus`,
`hospital`, `admin`), recorded at the same times. Every file is 5 minutes long
and its name tells you everything:

```text
drops-123-r13.school.2018-03-13.15-50-00.15-55-00.school.G336.r13
                     ^^^^^^^^^^ ^^^^^^^^ ^^^^^^^^ ^^^^^^ ^^^^
                     date       start    end      site   camera
```

The same string is the `stream_id` in semantic-va. Files from **the same site,
date and start time but different cameras** were recorded at the same time.
Those are your test sets.

**Bonus: `meva/drop-5-mevid`.** MEVID (WACV 2023) is a person-ReID dataset built
on these MEVA videos, with identity labels for people across cameras. The
semantic-va share has its video drop. If the MEVID identity annotations can be
used (check https://github.com/Kitware/MEVID and its license with your teacher),
they could replace most of your hand labelling.

---

## Part 2. Start semantic-va

Follow `semantic-va/doc/RUN-MAC.md` steps 1–5:

1. Connect FortiClient VPN and run `open smb://10.1.2.90/ai`
2. Open Docker Desktop, then `docker start semantic-va-local-db`
3. Start the API on port 8080 (terminal 1)
4. Start the web UI on port 3000 (terminal 2)

Check:

```bash
curl http://127.0.0.1:8080/health
```

```bash
ls /Volumes/ai/datasets/meva
```

---

## Part 3. Find cameras that recorded at the same time

This runs a read-only query (`SELECT`) on semantic-va's database. **Never** run
`INSERT`, `UPDATE`, `DELETE` or `DROP` on it. It's the team's index.

```bash
source ~/.config/semantic-va/shared.env.sh
psql "$SEMANTIC_VA_DSN"
```

Then, inside `psql`:

```sql
WITH meva AS (
  SELECT stream_id,
         substring(stream_id from '(\d{4}-\d{2}-\d{2}\.\d{2}-\d{2})')     AS start_minute,
         substring(stream_id from '\d{2}-\d{2}-\d{2}\.([a-z]+)\.G\d+')    AS site,
         substring(stream_id from '\.(G\d+)\.')                           AS camera
  FROM (SELECT DISTINCT stream_id FROM stream_runs) s
  WHERE stream_id ~ '\.G\d+\.'
),
people AS (
  SELECT stream_id, count(*) AS person_tracks
  FROM tracks WHERE class = 'person'
  GROUP BY stream_id
)
SELECT m.start_minute, m.site, count(*) AS cameras,
       sum(coalesce(p.person_tracks, 0)) AS people,
       string_agg(m.camera || ' (' || coalesce(p.person_tracks, 0) || ')', ', ' ORDER BY m.camera) AS per_camera
FROM meva m LEFT JOIN people p USING (stream_id)
GROUP BY m.start_minute, m.site
HAVING count(*) >= 2
ORDER BY people DESC
LIMIT 20;
```

What it does:

- `substring(... from 'regex')` cuts the date, site and camera out of the file name
- `people` counts person tracks per file
- the final `SELECT` groups files by **same minute + same site** and keeps
  groups with 2 or more cameras

**Pick one group** with 2–4 cameras and plenty of people. Write down the
`stream_id`s:

```sql
SELECT stream_id, url FROM stream_runs
WHERE stream_id LIKE '%2018-03-13.15-50%school%';   -- your group's minute and site
```

`url` is the file's path **relative to** the storage root (`/Volumes/ai/datasets`).
Type `\q` to leave psql.

**Look at them first:** search on http://127.0.0.1:3000/search and use the
**Streams** filter to see the clips from those cameras. Make sure people really
walk between the cameras.

---

## Part 4. Download the videos

### 4a. Copy the whole 5-minute files

```bash
mkdir -p /Applications/gw-junior/mini-reid/data/videos
cd /Applications/gw-junior/mini-reid/data/videos
cp "/Volumes/ai/datasets/<url from the query>" .
```

Each file is about 100–150 MB. If the VPN drops during `cp`, run it again.

### 4b. Check what you got

```bash
ffprobe -v error -show_entries format=duration:stream=codec_name,width,height,r_frame_rate -of compact <file>
```

### 4c. Make browser-friendly MP4s (and cut them if you want)

MEVA files are `.avi`. Browsers can't play AVI, and your Next.js page will want
to show video. Convert to H.264 MP4:

```bash
ffmpeg -i <file>.avi -c:v libx264 -crf 20 -preset fast -an -movflags +faststart <camera>.mp4
```

To keep only part of it, for example 2 minutes starting at 60 s, add `-ss 60 -t 120`
**after** `-i`. Putting `-ss` after `-i` cuts at the exact frame. Re-encoding with
`libx264` keeps the cut exact. With `-c copy`, ffmpeg can only cut at keyframes,
so the start would be off by up to a few seconds.

**Cut every camera in a group with the same `-ss` and `-t`.** They started
recording at the same minute, so the same offsets keep them in sync.

### 4d. Quick look at a single moment (optional)

semantic-va can make a short MP4 (10 s at most) from any clip:

```bash
curl -o look.mp4 "http://127.0.0.1:8080/clips/<clip_id>/recording?start=<seconds into the file>&duration=10"
```

It's handy for checking something, but too short for tracking.

---

## Part 5. What "save search result in database" means here

When you search in semantic-va, the results are clips, and each clip contains
tracks (people). Saving them means writing to **mini-reid's own database**:

1. the search you ran (text, filters, time)
2. the clips it returned, and their rank and score
3. the people (tracks) in those clips: camera, recording time, crop size
4. the crop images (as files on disk, with the path in the database)

Later you also save:

5. the ReID vector of each crop
6. every **Find similar person** search and its matches, plus your verdict on each
   match (same person or not)
7. labels: which tracks are really the same person (`person_labels`), built from
   your verdicts, for measuring accuracy
8. global identities: which tracks the system **decided** are the same person

**Check with your teacher** that this is what they meant. If they meant
something narrower (for example only item 6), you can keep just those tables.

### Why a separate database, and not semantic-va's?

- Your local semantic-va database is a **copy restored from a backup**.
  Refreshing it (`RUN-MAC.md`) runs `DROP DATABASE`, which would delete
  anything you added.
- New tables in the team's database should go through review, which is the
  final integration step, not the first.
- With the same technology (Postgres + pgvector) and the same keys
  (`run_id`, `track_key`, `clip_id`), integrating later is mostly copying the table
  definitions. You won't have to redesign anything.

---

## Part 6. Set up mini-reid's database

### 6a. Docker Compose file

Create `mini-reid/docker-compose.yml`:

```yaml
services:
  db:
    image: pgvector/pgvector:pg17        # same Postgres + pgvector as semantic-va
    container_name: mini-reid-db
    environment:
      POSTGRES_DB: mini_reid
      POSTGRES_USER: mini_reid
      POSTGRES_PASSWORD: mini_reid       # local practice only; bound to 127.0.0.1 below
    ports:
      - "127.0.0.1:5434:5432"            # 5433 is semantic-va's; 5434 is ours
    volumes:
      - mini_reid_data:/var/lib/postgresql/data
      - ./db/init:/docker-entrypoint-initdb.d:ro
volumes:
  mini_reid_data:
```

### 6b. The tables

Create `mini-reid/db/init/01-schema.sql`. Read every comment, because each one
explains a design decision:

```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- 1. One row per search you ran in semantic-va.
CREATE TABLE searches (
    id          BIGSERIAL PRIMARY KEY,
    query_text  TEXT NOT NULL,
    predicate   TEXT,                       -- e.g. has(person); NULL if none
    request     JSONB NOT NULL,             -- the exact request sent, so it can be re-run
    hit_count   INT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 2. Which clips a search returned, in order.
CREATE TABLE search_hits (
    search_id   BIGINT NOT NULL REFERENCES searches(id) ON DELETE CASCADE,
    rank        INT NOT NULL,               -- 1 = top result
    clip_id     BIGINT NOT NULL,            -- semantic-va's clip id
    score       DOUBLE PRECISION,           -- NULL when the search had no ranking
    PRIMARY KEY (search_id, rank)
);

-- 3. Clips, saved once even if many searches return them.
CREATE TABLE clips (
    clip_id     BIGINT PRIMARY KEY,         -- same id as semantic-va: easy to integrate
    run_id      UUID NOT NULL,
    stream_id   TEXT NOT NULL,
    camera_id   TEXT,                       -- e.g. G336, cut from stream_id
    source_url  TEXT,                       -- video path relative to the storage root
    clip_start  DOUBLE PRECISION NOT NULL,  -- seconds into the file
    clip_end    DOUBLE PRECISION NOT NULL,
    capture_ts  TIMESTAMP,                  -- RECORDING time, no time zone (as in semantic-va)
    detail      JSONB NOT NULL              -- the full GET /clips/{id} answer, for later questions
);

-- 4. People. One row per local track, keyed exactly like semantic-va.
--    `source` lets live tracks from step 8 live in the same table.
CREATE TABLE tracks (
    run_id      UUID NOT NULL,
    track_key   BIGINT NOT NULL,
    source      TEXT NOT NULL DEFAULT 'semantic-va',   -- or 'live'
    stream_id   TEXT NOT NULL,
    camera_id   TEXT,
    cls         TEXT NOT NULL,
    first_pts   DOUBLE PRECISION,           -- seconds into the file
    last_pts    DOUBLE PRECISION,
    start_time  TIMESTAMP,                  -- recording time = file start + first_pts
    end_time    TIMESTAMP,
    crop_px     INT,
    crop_file   TEXT,                       -- file name under mini-reid/data/crops
    clip_id     BIGINT REFERENCES clips(clip_id),      -- a clip it was seen in
    PRIMARY KEY (run_id, track_key)
);
CREATE INDEX tracks_camera_time ON tracks (camera_id, start_time);

-- 5. ReID vectors. Separate from appearance (SigLIP) vectors, and tagged with
--    the model that made them: vectors from different models can't be compared.
CREATE TABLE reid_embeddings (
    run_id       UUID NOT NULL,
    track_key    BIGINT NOT NULL,
    vector_space TEXT NOT NULL,             -- e.g. osnet_ain_x1_0-msmt17-v1
    sample       INT NOT NULL DEFAULT 0,    -- 0 = the stored best crop; more crops later
    embedding    vector(512) NOT NULL,      -- L2-normalised
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, track_key, vector_space, sample),
    FOREIGN KEY (run_id, track_key) REFERENCES tracks (run_id, track_key) ON DELETE CASCADE
);

-- 6. Every "Find similar person" search, and what it returned.
CREATE TABLE reid_queries (
    id           BIGSERIAL PRIMARY KEY,
    run_id       UUID NOT NULL,             -- the person you searched with
    track_key    BIGINT NOT NULL,
    vector_space TEXT NOT NULL,
    k            INT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE reid_matches (
    query_id     BIGINT NOT NULL REFERENCES reid_queries(id) ON DELETE CASCADE,
    rank         INT NOT NULL,
    run_id       UUID NOT NULL,
    track_key    BIGINT NOT NULL,
    score        REAL NOT NULL,             -- cosine similarity
    verdict      TEXT CHECK (verdict IN ('same', 'different', 'unsure')),  -- you fill this in the UI
    PRIMARY KEY (query_id, rank)
);

-- Ground truth for measuring: which tracks are really the same person.
-- Filled by hand, or automatically when you mark a match 'same'.
CREATE TABLE person_labels (
    run_id      UUID NOT NULL,
    track_key   BIGINT NOT NULL,
    person_id   TEXT NOT NULL,              -- e.g. p001; same id = same person
    labelled_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, track_key),
    FOREIGN KEY (run_id, track_key) REFERENCES tracks (run_id, track_key) ON DELETE CASCADE
);

-- 7. Global identities: "these local tracks are the same person". Steps 7-8.
CREATE TABLE global_entities (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE entity_track_links (
    entity_id   BIGINT NOT NULL REFERENCES global_entities(id) ON DELETE CASCADE,
    run_id      UUID NOT NULL,
    track_key   BIGINT NOT NULL,
    score       REAL,                       -- evidence: why we linked
    method      TEXT NOT NULL,              -- e.g. 'manual', 'offline-v1', 'live-v1'
    status      TEXT NOT NULL CHECK (status IN ('candidate', 'accepted', 'rejected')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id, run_id, track_key)
);
```

### 6c. Start it

```bash
cd /Applications/gw-junior/mini-reid
docker compose up -d
```

```bash
psql "postgresql://mini_reid:mini_reid@127.0.0.1:5434/mini_reid" -c '\dt'
```

The second command should list the tables.

**Important:** files in `db/init/` run **only the first time**, when the database
is empty. To change tables later, write a new file like
`db/migrations/002-add-x.sql` and run it with `psql -f`. To start over
completely (this **deletes all saved data**), run `docker compose down -v`, then
`docker compose up -d`.

---

## Part 7. The import script: search → database

### 7a. Install

```bash
cd /Applications/gw-junior/mini-reid/backend
source .venv/bin/activate
pip install "psycopg[binary]==3.3.5" pgvector==0.5.0 numpy
```

These are the same versions semantic-va uses.

### 7b. `backend/scripts/import_search.py`

```python
"""Run a semantic-va search and save the results in mini-reid's database.

    python scripts/import_search.py "a person walking" --predicate "has(person)" --top-k 50

Saves: the search, its hits, each clip, the people (tracks) in each clip,
and their crop images (into mini-reid/data/crops).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import re
import urllib.request

import psycopg
from psycopg.types.json import Jsonb

API = os.environ.get("SEMANTIC_VA_API", "http://127.0.0.1:8080")
DSN = os.environ.get("MINI_REID_DSN", "postgresql://mini_reid:mini_reid@127.0.0.1:5434/mini_reid")
ROOT = pathlib.Path(__file__).resolve().parents[2]      # mini-reid/
CROPS = ROOT / "data" / "crops"
MIN_PX = 90                                             # smaller crops are silhouettes
CAMERA = re.compile(r"\.(G\d+)\.")


def call(method: str, path: str, body: dict | None = None) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.status, r.read()


def camera_of(stream_id: str) -> str | None:
    m = CAMERA.search(stream_id + ".")
    return m.group(1) if m else None


def recording_time(capture_ts: str | None, clip_start: float, pts: float | None):
    """capture_ts is when THIS CLIP was recorded, so the file started clip_start earlier."""
    if capture_ts is None or pts is None:
        return None
    file_start = dt.datetime.fromisoformat(capture_ts) - dt.timedelta(seconds=clip_start)
    return file_start + dt.timedelta(seconds=pts)


def save_clip(conn: psycopg.Connection, clip_id: int) -> int:
    """Save one clip and its people. Returns how many people were saved."""
    _, body = call("GET", f"/clips/{clip_id}")
    detail = json.loads(body)
    hit = detail["hit"]
    camera = camera_of(hit["stream_id"])

    conn.execute(
        "INSERT INTO clips (clip_id, run_id, stream_id, camera_id, source_url,"
        " clip_start, clip_end, capture_ts, detail)"
        " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (clip_id) DO NOTHING",
        (clip_id, hit["run_id"], hit["stream_id"], camera, detail.get("source_url"),
         hit["clip_start"], hit["clip_end"], hit.get("capture_ts"), Jsonb(detail)))

    saved = 0
    for t in detail.get("tracks") or []:
        if t["cls"] != "person" or not t["has_crop"] or t["crop_px"] < MIN_PX:
            continue
        name = f"{hit['run_id']}_{t['track_key']}.jpg"
        path = CROPS / name
        if not path.exists():
            status, jpeg = call("GET", f"/clips/{clip_id}/tracks/{t['track_key']}/crop")
            if status == 204:          # semantic-va has the row but not the file
                print(f"  no crop file for track {t['track_key']} (is the share mounted?)")
                continue
            path.write_bytes(jpeg)
        conn.execute(
            "INSERT INTO tracks (run_id, track_key, stream_id, camera_id, cls,"
            " first_pts, last_pts, start_time, end_time, crop_px, crop_file, clip_id)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (run_id, track_key) DO NOTHING",
            (hit["run_id"], t["track_key"], hit["stream_id"], camera, t["cls"],
             t["first_pts"], t["last_pts"],
             recording_time(hit.get("capture_ts"), hit["clip_start"], t["first_pts"]),
             recording_time(hit.get("capture_ts"), hit["clip_start"], t["last_pts"]),
             t["crop_px"], name, clip_id))
        saved += 1
    return saved


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("text", help="what to search for, e.g. 'a person walking'")
    p.add_argument("--predicate", help="detector filter, e.g. 'has(person)'")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--stream", action="append", help="limit to a stream_id (repeatable)")
    args = p.parse_args()

    request = {"rules": [{"id": "q", "text": args.text}], "top_k": args.top_k}
    if args.predicate:
        request["predicate"] = args.predicate
    if args.stream:
        request["streams"] = args.stream

    _, body = call("POST", "/search", request)
    hits = json.loads(body)["results"][0]["hits"]
    print(f"{len(hits)} clips found")

    CROPS.mkdir(parents=True, exist_ok=True)
    # One transaction: if anything fails, nothing half-saved is left behind.
    with psycopg.connect(DSN) as conn:
        search_id = conn.execute(
            "INSERT INTO searches (query_text, predicate, request, hit_count)"
            " VALUES (%s, %s, %s, %s) RETURNING id",
            (args.text, args.predicate, Jsonb(request), len(hits))).fetchone()[0]
        people = 0
        for rank, hit in enumerate(hits, start=1):
            conn.execute(
                "INSERT INTO search_hits (search_id, rank, clip_id, score) VALUES (%s, %s, %s, %s)",
                (search_id, rank, hit["id"], hit.get("score")))
            people += save_clip(conn, hit["id"])
            print(f"  #{rank} clip {hit['id']} {hit['stream_id']}")
    print(f"saved search {search_id}: {len(hits)} clips, {people} person sightings")


if __name__ == "__main__":
    main()
```

### 7c. Run it

Search only your chosen cameras, so the people you save come from the same
place and time:

```bash
python scripts/import_search.py "people walking" --predicate "has(person)" --top-k 200 \
  --stream "<stream_id camera 1>" --stream "<stream_id camera 2>"
```

### 7d. Look at what you saved

```bash
psql "postgresql://mini_reid:mini_reid@127.0.0.1:5434/mini_reid"
```

```sql
SELECT id, query_text, hit_count, created_at FROM searches ORDER BY id DESC;

SELECT camera_id, count(*) AS people, min(start_time), max(end_time)
FROM tracks GROUP BY camera_id ORDER BY camera_id;

SELECT h.rank, h.score, c.camera_id, c.capture_ts
FROM search_hits h JOIN clips c USING (clip_id)
WHERE h.search_id = 1 ORDER BY h.rank LIMIT 10;
```

Open a few files in `data/crops/` to check they're people.

### 7e. Understand what you just wrote

| In the code | What it teaches |
|---|---|
| `ON CONFLICT ... DO NOTHING` | **Upsert.** Run the import twice and the clips and tracks aren't duplicated. A script you can safely re-run is called *idempotent*. |
| `with psycopg.connect(...)` | **Transaction.** Everything commits at the end, or nothing does if an error happens. |
| `%s` placeholders | **Never** put values into SQL with f-strings. Placeholders prevent SQL injection and quoting bugs. |
| `PRIMARY KEY (run_id, track_key)` | A `track_key` repeats across runs, so it's only unique together with `run_id`. |
| `Jsonb(detail)` | Keep the raw answer. Later you can answer new questions without calling the API again. |
| `capture_ts`, not `wall_ts` | `wall_ts` is when semantic-va **indexed** the clip, which isn't when it happened. |

---

## Part 8. What comes next

Once your tables have people in them:

1. **LEARNING-GUIDE.md step 3:** load OSNet-AIN
2. **Step 5:** embed every crop into `reid_embeddings`, then find similar people
   with SQL:

```sql
-- the 20 people from OTHER cameras who look most like track (run_id, track_key)
WITH q AS (
  SELECT e.embedding, t.camera_id
  FROM reid_embeddings e JOIN tracks t USING (run_id, track_key)
  WHERE e.run_id = '<run_id>' AND e.track_key = <track_key>
    AND e.vector_space = 'osnet_ain_x1_0-msmt17-v1' AND e.sample = 0
)
SELECT t.run_id, t.track_key, t.camera_id, t.start_time, t.crop_file,
       1 - (e.embedding <=> q.embedding) AS similarity
FROM reid_embeddings e
JOIN tracks t USING (run_id, track_key)
CROSS JOIN q
WHERE e.vector_space = 'osnet_ain_x1_0-msmt17-v1'
  AND t.camera_id IS DISTINCT FROM q.camera_id
ORDER BY e.embedding <=> q.embedding
LIMIT 20;
```

`<=>` is pgvector's **cosine distance** (0 = identical direction), so
`1 - distance` is cosine similarity.

3. **Step 6:** the API saves each search into `reid_queries` and `reid_matches`.
   The UI lets you mark each match `same` or `different`. Those verdicts are
   your labelled data for steps 5 and 9.
