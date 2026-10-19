# Learning person ReID, step by step

This guide builds mini-reid one step at a time. Each step teaches one idea,
has you build one small piece, and ends with a check for when the step is done.
Keep the steps in order, because each one uses the previous one.

**What you'll have at the end:**

1. After searching in semantic-va, you can pick a person and press **Find
   similar person** to see where else that person appears.
2. A live page follows that person across all cameras as new video arrives.
3. A plan for moving both features into semantic-va.

**Linear:** this work is [VIP-9](https://linear.app/gwave/issue/VIP-9) (OSNet-AIN
for people), under [VIP-5](https://linear.app/gwave/issue/VIP-5) (ReID for
cross-camera tracking). Vehicles are VIP-8 (FastReID) and are out of scope here.
Read the research document attached to VIP-5 before step 2. This guide follows
its recommendations.

**Where this fits:** [ROADMAP.md](ROADMAP.md) is the overall plan.
[FOUNDATIONS.md](FOUNDATIONS.md) has the background knowledge.
[DATA-AND-DATABASE.md](DATA-AND-DATABASE.md) covers downloading scenes and saving
search results. This guide is the hands-on ReID building.

| Step | You learn | You build |
|---|---|---|
| 0 | What ReID is, and which problem it solves | — |
| 1 | How semantic-va and mini-reid run | Both projects running |
| 2 | Embeddings, cosine similarity, Rank-1, mAP | — (reading) |
| 3 | Loading a ReID model correctly | `model.py`: crop → 512 numbers |
| 4 | Where crops come from, and labelling | Saved people in the database + labels |
| 5 | Gallery search and measuring it | Vectors in the database + Rank-1/mAP numbers |
| 6 | Serving it to a UI | API + **Find similar person** button |
| 7 | Linking tracks across cameras offline | `linker.py` + a person timeline |
| 8 | Doing it live | Live cross-camera tracking page |
| 9 | Measuring whether it's good enough | Threshold, false-merge rate, speed |
| 10 | Merging into semantic-va | An integration plan |

---

## Step 0. The big picture

### The problem

semantic-va already tracks people, but only **inside one camera**. Its tracker
(`src/core/tracking.py`) follows a box from frame to frame by overlap (IoU).
When a person walks out of camera A and into camera B, camera B starts a brand
new track. Nothing says it's the same person.

ReID (re-identification) answers: **"Is the person in this crop the same person
as the one in that crop?"** It uses only appearance: clothes, body shape and
colours. It doesn't recognise faces.

### Four words to keep apart

| Word | Meaning | In semantic-va |
|---|---|---|
| **Detection** | A box around a person in one frame | Detector output |
| **Local track** | The same person over many frames in **one** camera | `(run_id, track_key)` |
| **Embedding** | A list of numbers describing a crop's appearance | SigLIP: 1152 numbers. OSNet: 512. |
| **Global identity** | The same person across **many** cameras and times | Doesn't exist yet. This is what you're adding. |

ReID doesn't replace the local tracker. It **links local tracks together**
into global identities.

### Why not use the SigLIP vectors semantic-va already has?

SigLIP was trained to match an image with a **sentence** ("a person in a red
jacket"). Two different people in red jackets look nearly identical to it. A
ReID model is trained on thousands of people seen by several cameras. Its
training pushes crops of the **same person** together and crops of
**different people** apart, even when their clothes are similar. You'll measure
this difference yourself in step 5.

**Done when:** you can explain to a teammate, in two sentences, the difference
between a local track and a global identity.

---

## Step 1. Run both projects

### 1a. semantic-va search (to get real crops)

Follow `semantic-va/doc/RUN-MAC.md`: connect the VPN, mount the share, start the
database, the Python API (port 8080) and the web UI (port 3000). Then:

```bash
curl -s http://127.0.0.1:8080/streams | head -c 300
```

This should print JSON. Open http://localhost:3000/search, run any search,
open a result, and look at the tracks list with crops. Crops of people are
your raw material.

### 1b. mini-reid

```bash
cd /Applications/gw-junior/mini-reid/frontend
npm run dev
```

Open http://localhost:3001. You should see the mini-reid status page.

| Project | Web | API |
|---|---|---|
| semantic-va | 3000 | 8080 |
| mini-reid | 3001 | 8090 (from step 6) |

The mini-reid frontend forwards `/api/*` to port 8090, the same way
semantic-va forwards `/api/*` to 8080 (see `frontend/next.config.ts`).

**Done when:** both web pages open, and you can see person crops on a semantic-va
clip page.

---

## Step 2. ReID basics (reading)

Learn these ideas before writing ReID code. Keep notes in your own words.

### 2a. Embedding and cosine similarity

A ReID model turns a crop into a vector, for example 512 numbers. After
**L2-normalising** (dividing by its length, so the length becomes 1), the
similarity of two crops is their dot product, called **cosine similarity**:

- close to 1: very similar appearance
- close to 0: unrelated

There is **no universal "same person" threshold.** It depends on the model, the
cameras, the lighting and the crop size. You'll measure it on your own data in
step 9.

### 2b. Query and gallery

- **Query**: the one crop you're searching with ("this person")
- **Gallery**: all the crops you're searching in

"Find similar person" means: embed the query, compare it with every gallery
vector, and sort by similarity.

### 2c. Measuring retrieval

| Metric | Question it answers |
|---|---|
| **Rank-1** | Is the top result the same person? (% of queries) |
| **mAP** | Are all the correct matches near the top, not just the first one? |

In ReID tests, matches from the **same camera** are excluded. Finding the same
person in the next frame of the same camera is easy and proves nothing.

### 2d. Domain gap

Published numbers come from public datasets: Market-1501 (6 cameras, a
university campus) and MSMT17 (15 cameras). Your cameras differ in height,
angle, lighting and resolution. The research document notes that OSNet-AIN
trained on MSMT17 gets about 70% Rank-1 when tested on Market-1501, a different
dataset. **Expect lower accuracy on new cameras than the paper says.** That's
why you test on your own footage.

### 2e. Read

1. The research document attached to VIP-5 (all of it)
2. The OSNet paper, sections 1–3: "Omni-Scale Feature Learning for Person
   Re-Identification" (Zhou et al., 2019). Focus on why multi-scale features
   help, not on the maths.
3. The torchreid README and model zoo page:
   https://github.com/KaiyangZhou/deep-person-reid and
   https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO

**Done when:** you can explain Rank-1, mAP, and why same-camera matches are
excluded.

---

## Step 3. Load OSNet-AIN and embed one crop

### Goal

`backend/app/model.py`: give it a person crop, and it returns 512 normalised numbers.

### 3a. Install

```bash
cd /Applications/gw-junior/mini-reid/backend
source .venv/bin/activate
pip install torch torchvision numpy pillow
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

`mps` is the Apple GPU. When it prints `True`, inference can use it. If
torchreid fails to install, follow the install section of its README (clone,
`pip install -r requirements.txt`, then `pip install -e .`).

### 3b. Download the ReID-trained weights (common mistake)

In torchreid, `pretrained=True` loads **ImageNet** weights. Those are trained to
classify objects, not to identify people, so they look like they work but give
poor identity results. You need the **ReID checkpoint** from the model zoo.

1. Open the model zoo page and find **osnet_ain_x1_0** trained on **MSMT17**
   (the research document recommends this one).
2. Download it into `backend/weights/`. git ignores this folder.
3. Record the checksum:

```bash
shasum -a 256 weights/*.pth*
```

### 3c. Write a manifest

Create `backend/weights/manifest.json`. This records which model produced the
vectors. Vectors from two different checkpoints **can't be compared**, even when
they have the same length.

```json
{
  "vector_space": "osnet_ain_x1_0-msmt17-v1",
  "architecture": "osnet_ain_x1_0",
  "checkpoint_file": "<file name you downloaded>",
  "checkpoint_url": "<link from the model zoo>",
  "sha256": "<from shasum>",
  "trained_on": "MSMT17",
  "dim": 512,
  "input": "RGB, resized to 256x128 (height x width)",
  "pixel_norm": "ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]",
  "feature_norm": "L2",
  "license": "torchreid: MIT. Check the weights' terms on the model zoo page."
}
```

### 3d. `app/model.py`

```python
"""OSNet-AIN: person crop -> 512 L2-normalised numbers."""
from __future__ import annotations

import json
import pathlib

import numpy as np

WEIGHTS = pathlib.Path(__file__).parent.parent / "weights"


class ReidModel:
    def __init__(self, device: str | None = None):
        import torch

        try:
            from torchreid.utils import FeatureExtractor
        except ImportError:  # newer torchreid layout
            from torchreid.reid.utils import FeatureExtractor

        self.manifest = json.loads((WEIGHTS / "manifest.json").read_text())
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.extractor = FeatureExtractor(
            model_name=self.manifest["architecture"],
            model_path=str(WEIGHTS / self.manifest["checkpoint_file"]),
            image_size=(256, 128),
            device=device,
        )

    def embed(self, paths: list[str]) -> np.ndarray:
        """Return an (N, 512) array; each row has length 1."""
        feats = self.extractor(paths).cpu().numpy().astype(np.float32)
        return feats / np.linalg.norm(feats, axis=1, keepdims=True)
```

When it loads, torchreid may warn that the `classifier` layer was discarded.
That's expected: the classifier was only used during training, to name the
training identities. You only need the features before it.

### 3e. Try it

Take three crops: two of the **same** person from different moments, and one of a
**different** person. Use crops from semantic-va (right-click the image on a clip
page, then Save Image As) or your own photos.

```python
from app.model import ReidModel
m = ReidModel()
v = m.embed(["same_a.jpg", "same_b.jpg", "other.jpg"])
print(v.shape)             # (3, 512)
print(v[0] @ v[1])         # same person: should be higher
print(v[0] @ v[2])         # different person: should be lower
```

**Done when:** the same-person pair scores higher than the different-person pair.
If it doesn't, check that you loaded the ReID checkpoint and not ImageNet weights.

**For VIP-9:** note the checkpoint, its source, the preprocessing, the install
problems you hit, and how long one crop takes on MPS and on CPU.

---

## Step 4. Collect person crops and label some

### 4a. Save people from semantic-va into your database

Do [DATA-AND-DATABASE.md](DATA-AND-DATABASE.md) parts 2–7. At the end you
have:

- mini-reid's database (Postgres + pgvector, port 5434)
- `tracks`: people from 2–4 MEVA cameras that recorded at the same time, with
  camera, recording time and crop size
- `data/crops/`: one JPEG per track (crops under 90 px skipped)
- `data/videos/`: the full videos of those cameras, for step 8

Remember how semantic-va's crops work (see `SEMANTIC-VA-NOTES.md`): **one** crop
per track, the largest one seen over the whole track.

### 4b. Look before you label

```sql
SELECT camera_id, count(*) AS people,
       min(start_time) AS first_seen, max(end_time) AS last_seen,
       percentile_cont(0.5) WITHIN GROUP (ORDER BY crop_px) AS median_px
FROM tracks GROUP BY camera_id ORDER BY camera_id;
```

Check that the cameras overlap in time. If they don't, you picked files that
weren't recorded together.

### 4c. Label a small set

Labels go in the `person_labels` table: tracks with the same `person_id` are the
same person. There are two ways to fill it:

1. **From the UI (recommended, once step 6 works):** mark a match **same**, and
   both tracks get the same `person_id`.
2. **By hand, now:** look at crops and write SQL:

```sql
INSERT INTO person_labels (run_id, track_key, person_id) VALUES
  ('<run_id>', 17, 'p001'),   -- camera G331
  ('<run_id>', 4,  'p001'),   -- camera G336, same person
  ('<run_id>', 22, 'p002');
```

Aim for **at least 30 people, each seen by at least two cameras**, plus
look-alikes: different people in similar clothes. Those are the cases that
matter most. Only label what you're sure of. Leave out anything you're unsure
of rather than guessing.

```sql
-- how many labelled people are seen by 2+ cameras?
SELECT count(*) FROM (
  SELECT l.person_id FROM person_labels l JOIN tracks t USING (run_id, track_key)
  GROUP BY l.person_id HAVING count(DISTINCT t.camera_id) >= 2
) x;
```

**Done when:** the database has 100+ person tracks from 2+ cameras, and 30+
labelled people seen by 2+ cameras. You can finish the labels during step 6.

---

## Step 5. Gallery search, and measuring it

### 5a. Embed every saved crop: `backend/scripts/embed_crops.py`

```python
"""Give every saved person crop a ReID vector, skipping ones already done.

    python scripts/embed_crops.py
"""
import os
import pathlib
import sys

import psycopg
from pgvector.psycopg import register_vector

BACKEND = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))                    # so `from app...` works
from app.model import ReidModel                     # noqa: E402

DSN = os.environ.get("MINI_REID_DSN", "postgresql://mini_reid:mini_reid@127.0.0.1:5434/mini_reid")
CROPS = BACKEND.parent / "data" / "crops"
BATCH = 32                                          # crops per model call

model = ReidModel()
space = model.manifest["vector_space"]
with psycopg.connect(DSN) as conn:
    register_vector(conn)                           # lets psycopg send numpy arrays as vectors
    todo = conn.execute(
        "SELECT t.run_id, t.track_key, t.crop_file FROM tracks t"
        " WHERE t.crop_file IS NOT NULL AND NOT EXISTS ("
        "   SELECT 1 FROM reid_embeddings e"
        "   WHERE e.run_id = t.run_id AND e.track_key = t.track_key"
        "     AND e.vector_space = %s AND e.sample = 0)",
        (space,)).fetchall()
    print(len(todo), "crops to embed")
    for i in range(0, len(todo), BATCH):
        batch = todo[i:i + BATCH]
        vectors = model.embed([str(CROPS / name) for _, _, name in batch])
        for (run_id, track_key, _), v in zip(batch, vectors):
            conn.execute(
                "INSERT INTO reid_embeddings (run_id, track_key, vector_space, embedding)"
                " VALUES (%s, %s, %s, %s)", (run_id, track_key, space, v))
        conn.commit()                               # save progress after every batch
        print(f"  {i + len(batch)}/{len(todo)}")
```

It commits after every batch, so if it stops halfway, running it again continues
where it stopped. Then try the similar-person SQL in
[DATA-AND-DATABASE.md](DATA-AND-DATABASE.md) part 8 on a few people.

At this size (hundreds or thousands of rows), Postgres compares the query with
every row (exact search) in milliseconds. You don't need an HNSW index yet.

### 5b. Measure Rank-1 and mAP

Load the labelled vectors from the database:

```python
import numpy as np
import psycopg
from pgvector.psycopg import register_vector

with psycopg.connect(DSN) as conn:
    register_vector(conn)
    rows = conn.execute(
        "SELECT e.embedding, l.person_id, t.camera_id"
        " FROM reid_embeddings e"
        " JOIN tracks t USING (run_id, track_key)"
        " JOIN person_labels l USING (run_id, track_key)"
        " WHERE e.vector_space = %s AND e.sample = 0", (space,)).fetchall()
vectors = np.stack([r[0] for r in rows]).astype(np.float32)
pids = np.array([r[1] for r in rows])
cams = np.array([r[2] for r in rows])
```

Then, for every labelled crop as the query, search the **other cameras** only:

```python
def evaluate(vectors, pids, cams):
    rank1, aps = [], []
    for q in range(len(pids)):
        keep = cams != cams[q]                     # other cameras only
        if not (pids[keep] == pids[q]).any():
            continue                               # no true match to find
        scores = vectors[keep] @ vectors[q]
        hits = (pids[keep] == pids[q])[np.argsort(-scores)]
        rank1.append(hits[0])
        found = np.cumsum(hits)
        aps.append((found[hits] / (np.flatnonzero(hits) + 1)).mean())
    return np.mean(rank1), np.mean(aps)

print("Rank-1 %.3f  mAP %.3f" % evaluate(vectors, pids, cams))
```

### 5c. Compare with the baseline

Run the same evaluation with **SigLIP** vectors of the same crops (the model
semantic-va uses for appearance, `google/siglip2-so400m-patch16-naflex`). This is
the first comparison the research asks for: does a dedicated ReID model actually
beat what semantic-va already has?

**Shortcut:** you don't need to run SigLIP yourself. semantic-va already stored a
SigLIP vector for each track in `track_appearance.embedding` (where it was
embedded), keyed by the same `(run_id, track_key)` as your `tracks` table. Read
them from semantic-va's database (read-only `SELECT`, see
[DATA-AND-DATABASE.md](DATA-AND-DATABASE.md) part 3) for your labelled tracks.
Cast `halfvec` to `vector` with `embedding::vector` so pgvector-python returns a
NumPy array, then run the same `evaluate`. Skip tracks where `embedding` is NULL,
and report how many you skipped.

| Model | Rank-1 | mAP |
|---|---|---|
| SigLIP 2 (baseline) | | |
| OSNet-AIN x1.0 MSMT17 | | |

### 5d. Look at the failures

For the queries where Rank-1 was wrong, put the query and the top result side by
side. Group them by cause: tiny crop, back view against front view, occlusion,
lighting, look-alike clothes. This tells you more than the overall number.

**Done when:** you have the table filled in and know the three most common failure
types.

**For VIP-9:** this table, with the number of people and cameras, is the core
evidence. State clearly that it's measured on project footage, not copied from a
paper.

---

## Step 6. The "Find similar person" button

### 6a. Backend API (port 8090)

```bash
pip install fastapi "uvicorn[standard]"
```

`app/api.py` loads the model **once** at startup, and opens database
connections with `MINI_REID_DSN`. Endpoints:

| Endpoint | Does |
|---|---|
| `GET /health` | `{"ok": true, "people": N, "vector_space": "..."}` |
| `GET /people?camera=G331` | Saved person tracks (for the grid on the search page) |
| `POST /similar` | Body `{"run_id": "...", "track_key": 17, "k": 20}`. Runs the SQL from DATA-AND-DATABASE part 8, **saves** the search in `reid_queries` and every result in `reid_matches`, and returns `query_id` plus the matches (camera, time, score, crop file). |
| `PUT /similar/{query_id}/matches/{rank}` | Body `{"verdict": "same"}`. Saves your verdict, and for `same` also updates `person_labels` (rules below). |
| `GET /crops/{file}` | Serves a crop JPEG from `data/crops/`. Reject names containing `/` or `..`, so nobody can read other files. |

**How a `same` verdict becomes labels:**

- Neither track has a `person_id`: create a new one (`p001`, `p002`, ...) and
  give it to both.
- One has a `person_id`: give it to the other too.
- Both have **different** `person_id`s: don't change anything. Return a conflict
  (HTTP 409) and show it. Either an earlier label was wrong, or they really are
  one person. A human must decide; the code mustn't guess.

```bash
uvicorn app.api:app --port 8090 --reload
```

Open http://127.0.0.1:8090/docs to try each endpoint before building the UI.

### 6b. Frontend

Before writing code, read `frontend/AGENTS.md`. This Next.js version differs from
older tutorials, and the docs are in `frontend/node_modules/next/dist/docs/`.

1. `/search` page: a grid of person crops (`GET /api/people`), filterable by
   camera, each with a **Find similar person** button.
2. On click, call `POST /api/similar` and show the results: crop, camera,
   recording time and score.
3. Under each result, add **Same / Different / Unsure** buttons that call
   `PUT /api/similar/{query_id}/matches/{rank}`.
4. Show the score, but **don't** label results "same person" automatically. You
   don't have a threshold until step 9. For now these are "candidates", and
   **you** are the judge.

### 6c. How it maps to semantic-va later

In semantic-va the button belongs in `web/src/components/ClipTabs.tsx`
(`TrackRow`), next to each track's crop, shown when `cls === "person"` and the
track has a crop. The backend becomes an endpoint in `src/api/routes.py`. Keep
your request and response shapes close to semantic-va's style (`run_id`,
`track_key`, `clip_id`) so the move is a copy, not a rewrite.

**Done when:** you click a person, see candidates from other cameras, and the
correct person is near the top for most of your labelled examples.

---

## Step 7. Linking tracks across cameras (offline)

"Similar" is a ranked list. "Tracking across cameras" needs **decisions**: this
track belongs to that person. A high score alone isn't enough to decide.

### 7a. Three gates before appearance

For a track in camera B, the only candidates from camera A are tracks that:

1. **Are in a connected camera.** Write a small topology file:

   ```yaml
   cameras:
     G331: { site: bus, next: { G336: [5, 60] } }   # seconds from leaving G331 to entering G336
   ```

2. **Fit the time window.** Use the **recording time**, never `wall_ts` (that's
   when the clip was indexed). For a file:
   `track start ≈ capture_ts − clip_start + first_pts`.
   Check this formula on one clip by watching the video before trusting it.
3. **Aren't the same camera at the same moment.** One person can't be two
   different tracks in one camera at once. (Overlapping cameras can see the same
   person at the same time, which is allowed.)

### 7b. Then appearance, with an "unmatched" option

- Accept the best candidate only if its score is above a threshold **and** clearly
  higher than the second best (for example, the gap is more than 0.05). Measure the
  numbers in step 9; don't trust these examples.
- Otherwise, **leave it unmatched** and start a new global identity. A missed
  link is cheaper than a wrong one. A wrong merge combines two people's
  histories.

### 7c. Several tracks at once: the Hungarian algorithm

When three people enter camera B at once, match them all together, not one by one:

```python
from scipy.optimize import linear_sum_assignment
cost = 1 - scores            # rows: new tracks, cols: candidates
rows, cols = linear_sum_assignment(cost)
# then drop pairs below the threshold
```

### 7d. Data you produce

Match the tables the research proposes, so semantic-va can reuse them:

- `global_entities`: `entity_id`, created time
- `entity_track_links`: `entity_id`, `run_id`, `track_key`, `score`, method, status
  (`candidate` or `accepted`)

Keep the evidence for every link so a wrong link can be removed without
rebuilding everything.

### 7e. Person timeline

For one `entity_id`, list its linked tracks ordered by recording time: camera,
start, end and crop. This is the "what route did this person take?" view. Add a
`/person/[entity_id]` page in the frontend for it.

**Done when:** for your labelled people, the timeline shows the correct camera
sequence, and you've counted the wrong links (false merges) and missed links.

---

## Step 8. Live tracking across all cameras

Now do step 7 continuously while video plays.

### 8a. Simulate live cameras

You don't need real cameras. Play 2–4 MEVA files from **connected cameras, same
time window** at real speed, as if they were live streams. Each file is one
"camera".

### 8b. Pipeline per camera

```text
video → detector (person only) → local tracker (ByteTrack)
      → track confirmed: take 1-4 sharp, large crops
      → OSNet embed → match against active identities (step 7 gates)
      → assign global ID → event to frontend
```

For the mini project, Ultralytics gives you a detector and ByteTrack in a few lines:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
for r in model.track(source="cam_G331.mp4", classes=[0], tracker="bytetrack.yaml",
                     stream=True, persist=True):
    if r.boxes.id is None:
        continue
    for box, tid in zip(r.boxes.xyxy, r.boxes.id):
        ...  # crop r.orig_img[y1:y2, x1:x2] (BGR, convert to RGB) for track tid
```

Ultralytics is **AGPL-3.0** licensed. That's fine for learning, but don't
copy it into semantic-va. semantic-va has its own detector (`detector.py`) and
tracker (`tracking.py`). The ReID part (crop → vector → match) is what you'll
reuse.

### 8c. Rules for live

- **Bounded active gallery:** only identities seen in the last N minutes (the
  longest travel time in your topology, plus a margin). Older ones are archived.
- **Bounded queues:** if embedding falls behind, drop or skip crops and count
  them. Never let the queue grow without limit.
- **Embed per track, not per frame:** wait until a track is confirmed (for
  example 10 frames), then embed a few good crops. Average them, then L2-normalise.
- **Run each camera in its own thread or process;** one shared matcher assigns
  global IDs.

### 8d. Push to the frontend

Send events from FastAPI with Server-Sent Events (`StreamingResponse`,
`media_type="text/event-stream"`), for example
`{"entity_id": 7, "camera": "G336", "time": "...", "crop": "..."}`.
In the frontend, `new EventSource("/api/events")`.

Page `/live`: one tile per camera, plus a "followed person" panel. Click a person
to follow them. When they appear in another camera, that camera's tile lights up
and the timeline grows.

**Done when:** you follow a person walking from camera A to camera B, and B picks
them up with the same global ID, without restarting anything.

---

## Step 9. Measure whether it's good enough

### 9a. Choose the threshold from data

From your labels, collect scores of **same-person** pairs and
**different-person** pairs (across cameras). Plot both histograms. Then, for each
threshold, compute:

- **False merges**: different people wrongly linked. This is the most harmful
  error.
- **Recall**: same-person pairs correctly linked
- **Abstentions**: left unmatched

Ask the team how many false merges are acceptable (for example "at most 1 in
100 accepted links"). Pick the threshold that meets it. Choose it on one part of
the footage and **test it on a different part** (a different recording
session), or you'll fool yourself.

### 9b. Ablation table

Each row adds one thing, so you can see what helps:

| Setup | Rank-1 | False merges | Recall |
|---|---|---|---|
| SigLIP, 1 crop | | | |
| OSNet, 1 crop | | | |
| OSNet, up to 4 crops averaged | | | |
| OSNet + time and camera gates | | | |

### 9c. Speed

On the Mac (MPS and CPU), measure: crops embedded per second, delay from
"track confirmed" to "global ID assigned" (p95), queue length, and dropped crops.
Model parameter count says nothing about speed, so measure it.

### 9d. Harder tracking metrics (optional)

For full multi-camera tracking quality, IDF1 and HOTA are the standard metrics.
TrackEval (https://github.com/JonathonLuiten/TrackEval) implements them, but it
needs ground-truth boxes for every frame, which takes much more labelling. Do
this only if the simpler numbers above look promising.

**Done when:** you have a threshold, its false-merge rate on held-out footage,
the ablation table, and the speed numbers.

---

## Step 10. Plan the merge into semantic-va

Write `docs/INTEGRATION-PLAN.md` covering each item below. Read
`SEMANTIC-VA-NOTES.md` first.

| Area | Change |
|---|---|
| Database | New `track_reid_samples` table (`run_id`, `track_key`, crop reference, media time, quality, `halfvec(512)`, `vector_space`), plus `global_entities` and `entity_track_links`. Separate from `track_appearance`. |
| Cameras | Stable `camera_id` and `site_id` for streams, a time zone per site, and the camera topology file |
| Capture | Keep up to 4 good crops per track instead of 1 (`detect.py`), and embed them with OSNet in a bounded queue |
| Backfill | An offline tool (like semantic-va's other `tools/*.py`) that embeds existing crops |
| API | `GET /clips/{id}/tracks/{key}/same-identity` and `GET /entities/{id}/timeline` in `src/api/routes.py` |
| Web | Button in `ClipTabs.tsx` `TrackRow`, plus an entity timeline page |
| Code rules | Lazy torch imports (`tests/test_core_boundary.py`), CPU-only search, and an updated comment in `tracking.py` |
| Rollout | Show candidates only first. Turn on automatic links only after step 9 passes the agreed false-merge rate. |

---

## What goes into VIP-9

From VIP-5's acceptance criteria, the people (OSNet-AIN) side:

- [ ] Recommended checkpoint: name, source link, checksum, training data,
      preprocessing, license (step 3)
- [ ] Runtime: install steps, Python 3.12 compatibility, MPS and CPU speed (steps 3, 9)
- [ ] Results on project footage: Rank-1 and mAP against the SigLIP baseline
      (step 5), and threshold with false-merge rate (step 9)
- [ ] Limitations: small crops, back and front views, look-alike clothes, clothing
      changes (step 5d)
- [ ] Association design and storage changes (steps 7 and 10)

Keep published numbers (from papers) and measured numbers (from your footage) in
separate columns, and label which is which.
