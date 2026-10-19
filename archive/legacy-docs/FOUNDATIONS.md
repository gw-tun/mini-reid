# Foundations: everything you need to know for mini-reid

This is the knowledge behind the project, from the beginning. Each chapter
explains one topic, gives a small exercise, and ends with questions to check
yourself. If you can answer them without looking, move on.

Don't try to read it all in one day. [ROADMAP.md](ROADMAP.md) says which
chapters you need before each phase of the project.

| # | Chapter | Needed before |
|---|---|---|
| 1 | How to learn with this course | Start |
| 2 | Video: frames, codecs, time | Downloading scenes |
| 3 | Images as numbers | Loading a model |
| 4 | Object detection | Live tracking |
| 5 | Tracking inside one camera | Understanding semantic-va's tracks |
| 6 | Neural networks, just enough | Loading a model |
| 7 | Embeddings and similarity search | ReID search |
| 8 | Person ReID | ReID search |
| 9 | Multi-camera tracking | Cross-camera linking |
| 10 | Databases: SQL, Postgres, pgvector, Docker | Saving search results |
| 11 | Backend: HTTP, JSON, FastAPI | The API |
| 12 | Frontend: Next.js and live updates | The button and live page |
| 13 | How semantic-va is built | Integration |
| 14 | Glossary | Any time |

---

## 1. How to learn with this course

- **Run everything.** Reading code teaches less than running it and changing
  one thing.
- **Keep a learning log** (`docs/LOG.md`): what you did, what broke, how you
  fixed it. It becomes your report to your teacher, and the VIP-9 write-up.
- **When stuck, ask a good question.** Include what you ran, the full error, and
  what you expected. "It doesn't work" can't be answered. A pasted traceback
  usually can.
- **Measure, don't assume.** semantic-va's docs show this everywhere: every
  claim has a number behind it. Do the same.

---

## 2. Video: frames, codecs, time

### Frames and frame rate

A video is a sequence of still images (**frames**). **fps** (frames per second)
says how many frames per second. MEVA is 1920×1080 at 30 fps, so 5 minutes is
9,000 frames.

You rarely need every frame. semantic-va runs its detector at about **2 fps**,
because people don't move far in half a second.

### Codec vs container

| | What it is | Examples |
|---|---|---|
| **Codec** | How the pixels are compressed | H.264, H.265 |
| **Container** | The file format that holds the compressed video, audio and timestamps | `.mp4`, `.avi`, `.mkv` |

The same H.264 video can sit in an AVI or an MP4. Browsers play H.264 **in MP4**,
not in AVI. That's why [DATA-AND-DATABASE.md](DATA-AND-DATABASE.md) converts MEVA's
AVI files.

### Keyframes

H.264 stores a few full pictures (**keyframes**) and, between them, only the
changes. To decode frame 100, the decoder starts from the last keyframe before
it. So `ffmpeg -c copy` (no re-encoding) can only cut at keyframes, and exact
cuts need re-encoding.

### Three kinds of time (important)

| Time | Meaning | In semantic-va |
|---|---|---|
| **Media time (PTS)** | Seconds from the start of the file | `clip_start`, `first_pts`, `best_pts` |
| **Recording time** | When the camera actually filmed it | `capture_ts` (from the file name) |
| **Processing time** | When a computer processed it | `wall_ts` |

To compare sightings across cameras you need **recording time**. A file indexed
in 2026 of footage filmed in 2018 has a `wall_ts` in 2026. For a MEVA file:

```text
recording time of a moment = file start (from the name) + media time
```

### Exercise

```bash
ffprobe -v error -show_entries stream=codec_name,width,height,r_frame_rate -of compact <a video>
```

Then extract one frame at 30 s as a JPEG:

```bash
ffmpeg -ss 30 -i <a video> -frames:v 1 frame30.jpg
```

### Check yourself

1. Why can't a browser play a MEVA `.avi` file directly?
2. A clip has `capture_ts = 2018-03-13 15:52:00` and `clip_start = 120`. When did
   the file start recording?
3. Why is `wall_ts` wrong for ordering sightings across cameras?

---

## 3. Images as numbers

A colour image is a 3-D array: **height × width × 3** channels. Each value is
0–255.

```python
from PIL import Image
import numpy as np
img = np.asarray(Image.open("frame30.jpg"))
print(img.shape, img.dtype)   # (1080, 1920, 3) uint8
```

### Things that often go wrong

| Issue | Explanation |
|---|---|
| **RGB vs BGR** | PIL and PyTorch models use RGB. OpenCV (and so Ultralytics' `orig_img`) uses **BGR**. Feeding BGR to an RGB model swaps red and blue, so a red jacket "looks" blue. Convert with `img[:, :, ::-1]`. |
| **Crop = slicing** | `crop = img[y1:y2, x1:x2]`, rows (y) first, then columns (x) |
| **Normalised boxes** | semantic-va stores boxes as fractions 0–1 of the width and height. Multiply by the image size to get pixels. |
| **Resize** | OSNet expects **256×128** (height × width, a standing person). A 60 px tall crop gets stretched 4×, which invents no detail. |
| **Normalisation** | Models expect pixels scaled to 0–1, then `(x - mean) / std` per channel, using the same numbers as in training (ImageNet mean and std). torchreid does this for you. |

### Check yourself

1. A box is `(0.5, 0.2, 0.6, 0.8)` normalised xyxy on a 1920×1080 frame. What
   are its pixel coordinates?
2. What happens if you give an RGB model a BGR crop?

---

## 4. Object detection

A **detector** looks at one frame and returns boxes, each with a **class**
(`person`, `car`, ...) and a **confidence** (0–1).

- **YOLO** is a popular family of fast detectors. semantic-va uses a detector
  trained on **COCO** classes, and keeps six of them: `person`, `car`, `truck`,
  `bus`, `motorcycle` and `bicycle`.
- **NMS** (non-maximum suppression): the detector proposes many overlapping boxes
  for one object, and NMS keeps the best one.
- **Small objects are hard.** A person 40 px tall has very few pixels to work
  with, and this limits ReID even more than detection.

### Exercise (in step 8 you'll need this anyway)

```bash
pip install ultralytics
```

```python
from ultralytics import YOLO
r = YOLO("yolo11n.pt")("frame30.jpg", classes=[0])[0]   # class 0 = person in COCO
print(r.boxes.xyxy, r.boxes.conf)
r.save("frame30_boxes.jpg")
```

### Check yourself

1. What three things does a detector return for each object?
2. Why is a person far from the camera harder to re-identify?

---

## 5. Tracking inside one camera

A detector has no memory: frame 101 doesn't know about frame 100. A **tracker**
links detections over time into **tracks**, each with an ID.

### IoU

**Intersection over Union** measures how much two boxes overlap:
`area of overlap / area of union`. 1 = identical, 0 = no overlap.

```python
def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter)
```

### ByteTrack (what semantic-va uses)

1. Predict where each existing track will be in this frame (a Kalman filter
   assumes roughly constant velocity).
2. Match **high-confidence** detections to tracks by IoU.
3. Match the leftover tracks to **low-confidence** detections too. This is
   ByteTrack's idea: a person behind a pole still gets a weak detection.
4. Unmatched detections start new tracks. A track that's missing too long ends.

### Errors

| Error | What happens |
|---|---|
| **ID switch** | Two people cross, and their IDs swap |
| **Fragmentation** | One person becomes two tracks (for example after being hidden) |

### Why ReID is needed

IoU only works when boxes overlap between frames, so inside one camera. When a
person walks into **another camera**, there's no box to overlap. Only
appearance can connect them. semantic-va's `tracking.py` says it excludes ReID
on purpose. Your project adds it **on top**, as a separate layer.

### Check yourself

1. Why can't IoU link a person across two cameras?
2. What is the difference between an ID switch and fragmentation?

---

## 6. Neural networks, just enough

- A **model** is a function with millions of numbers inside (**weights**).
- **Training** adjusts the weights using labelled examples. **Inference** just
  uses them. You'll only do inference.
- A **checkpoint** (`.pth`) is a saved set of weights. The same architecture
  with different checkpoints behaves completely differently, which is why the
  manifest records which one you use.
- **Backbone and head:** the backbone turns an image into a **feature vector**.
  The head turns features into an answer, such as class scores. For ReID you keep
  the **features** and throw away the head.
- **Device:** NVIDIA GPU (`cuda`), Apple GPU (`mps`), or `cpu`. Your Mac has
  `mps`. The team's server has `cuda` (an A40).
- **Batch:** process several crops at once. It's faster than one at a time.
- **Tensor:** PyTorch's array, like a NumPy array that can live on a GPU.

```python
import torch
x = torch.rand(8, 3, 256, 128)   # a batch of 8 RGB crops, 256×128
print(x.shape)
```

### Check yourself

1. Why must you load a **ReID** checkpoint and not ImageNet weights?
2. Which part of the model do you keep for ReID, and which do you throw away?

---

## 7. Embeddings and similarity search

An **embedding** is a vector (a list of numbers) that describes something, so
that **similar things get similar vectors**.

| Model in semantic-va | Turns ... into | Length |
|---|---|---|
| Cosmos-Embed1 | a 2 s video clip, or a sentence | 768 |
| SigLIP 2 | an object crop, or a short phrase | 1152 |
| OSNet-AIN (yours) | a person crop | 512 |

### Cosine similarity

```python
import numpy as np
a = np.array([3.0, 4.0]); b = np.array([4.0, 3.0])
a /= np.linalg.norm(a); b /= np.linalg.norm(b)   # L2-normalise: length becomes 1
print(a @ b)                                      # 0.96 = very similar direction
```

After L2 normalisation, **the dot product is the cosine similarity**. semantic-va
stores normalised vectors for this reason.

### Rules

1. **Only compare vectors from the same model and checkpoint.** A SigLIP vector
   and an OSNet vector can't be compared, and two OSNet checkpoints can't either.
   That's why tables carry a `vector_space` column.
2. **A score is relative, not a probability.** 0.7 doesn't mean "70% sure".
3. **Nearest neighbours:** comparing the query with every row is *exact search*.
   At millions of rows, databases use an approximate index (**HNSW**) that's much
   faster and occasionally misses a result.

### pgvector (vectors in Postgres)

| Operator | Meaning |
|---|---|
| `<->` | Euclidean distance |
| `<=>` | Cosine distance (`1 - cosine similarity`) |
| `<#>` | Negative inner product (semantic-va uses this on normalised vectors) |

`halfvec` stores each number in 16 bits instead of 32: half the space, slightly
less precise. semantic-va measured that this was acceptable for its vectors.

### Check yourself

1. Why do we L2-normalise before storing?
2. Why does the `reid_embeddings` table have a `vector_space` column?

---

## 8. Person ReID

### The task

Given a **query** crop, rank a **gallery** of crops so that the same person comes
first, even from a different camera, angle or lighting.

### How a ReID model learns

Training uses datasets where every crop is labelled with a person ID, seen by
several cameras:

- **ID loss:** a classifier head learns to name each training identity. This
  forces the backbone to find identity features.
- **Triplet loss:** for an anchor crop, a crop of the same person (positive) and a
  crop of someone else (negative), the positive must be closer than the negative
  by a margin.

After training, the head is thrown away, and the features cluster by identity
**even for people never seen in training**.

### OSNet-AIN

- **OSNet** (Omni-Scale Network) mixes features at several sizes at once: small
  details (a logo, shoes) and the whole body shape. It's small: about 2.2M
  parameters, so it's fast.
- **AIN** (adaptive instance normalisation) helps it generalise to **new
  cameras** (new "domains"), which is exactly your situation.
- Input is 256×128, output is 512 numbers.

### Datasets and metrics

| Dataset | Identities | Cameras |
|---|---|---|
| Market-1501 | 1,501 | 6 |
| MSMT17 | 4,101 | 15 |
| MEVID (on MEVA) | 158 | 33 |

- **Rank-1:** % of queries whose top result is correct
- **mAP:** quality of the whole ranked list
- **CMC curve:** Rank-k for k = 1, 5, 10, ...

### Limits (write these into VIP-9)

| Problem | Why |
|---|---|
| Changed clothes | ReID learns clothing more than faces |
| Same uniform | Different people look the same |
| Back vs front view | Different visible clothing |
| Occlusion | Half the crop is a car or another person |
| Tiny crops | Not enough pixels |
| Domain gap | New cameras look different from training data, and accuracy drops |

### Privacy

ReID follows people across places. Use it only for the project's purpose, keep
data on approved systems, and don't post crops of real people publicly.

### Check yourself

1. In your own words: what does triplet loss make the model do?
2. Why are same-camera matches excluded when measuring?
3. Name three situations where ReID will fail.

---

## 9. Multi-camera tracking

Also called **MTMC** (multi-target multi-camera) tracking.

```text
each camera:  detect → local track → good crops → ReID vector
across cameras:  candidate filter (where + when) → appearance score → decide → global ID
```

### Ideas

| Idea | Meaning |
|---|---|
| **Local vs global ID** | Local = inside one camera (`run_id, track_key`). Global = the same person everywhere (`global_entities`). |
| **Topology** | Which cameras connect, and how many seconds walking takes between them |
| **Time gate** | Only compare sightings that are possible in time. Someone can't leave camera A at 15:50 and be in camera B at 15:49. |
| **Overlapping cameras** | Two cameras see the same spot, so the same person can appear in both **at the same time** |
| **Threshold + margin** | Accept a match only if it's high enough **and** clearly better than the second best |
| **Open set** | The person may have **no** match: they came from outside. "Unmatched" must be a valid answer. |
| **Assignment** | When several people arrive at once, match them together (Hungarian algorithm), so two people can't claim the same match |
| **Online vs offline** | Offline sees all the data and can decide later. Online (live) must decide now, with only the past. |

### The two errors

| Error | Example | Cost |
|---|---|---|
| **False merge** | Two different people get one global ID | High: a wrong route, the wrong person followed |
| **Missed link** | One person gets two global IDs | Lower: the route is split, but nothing is wrong |

So the system should **prefer leaving things unmatched** when unsure.

### Check yourself

1. Why is a time gate needed if appearance is already compared?
2. Which error is worse, and why?

---

## 10. Databases: SQL, Postgres, pgvector, Docker

### Basics

- A **table** is like a spreadsheet: rows (records) and columns (fields), and every
  column has a type.
- A **primary key** uniquely identifies a row, for example `(run_id, track_key)`.
- A **foreign key** says "this value must exist in that table", for example
  `reid_embeddings → tracks`.
- An **index** makes lookups fast, like a book's index.

### The SQL you'll use

```sql
SELECT camera_id, count(*) FROM tracks GROUP BY camera_id;          -- read + summarise
INSERT INTO searches (query_text, request, hit_count) VALUES (...); -- add
INSERT ... ON CONFLICT (clip_id) DO NOTHING;                        -- add unless it exists (upsert)
UPDATE reid_matches SET verdict = 'same' WHERE query_id = 3 AND rank = 1;
SELECT ... FROM a JOIN b USING (run_id, track_key);                 -- combine tables
```

### Transactions

Several statements that succeed or fail **together**. In psycopg,
`with psycopg.connect(...) as conn:` commits at the end, or rolls back if an
error happens.

### JSONB

A column that holds a whole JSON document. It's useful for "keep the raw API
answer". Query inside it with `detail->'hit'->>'stream_id'`.

### Docker

Docker runs software in an isolated **container**, so you don't install Postgres
on your Mac. `docker compose up -d` starts what `docker-compose.yml` describes. A
**volume** keeps the data when the container stops. `docker compose down -v`
deletes the volume, and so **deletes all the data**.

### Check yourself

1. Why is the primary key of `tracks` two columns and not just `track_key`?
2. What does `ON CONFLICT DO NOTHING` protect against?
3. What's the difference between `docker compose down` and `docker compose down -v`?

---

## 11. Backend: HTTP, JSON, FastAPI

- **HTTP:** the browser (client) sends a **request** (method + path + optional
  body) and the server sends a **response** (status code + body).
- **Methods:** `GET` reads, `POST` creates or runs something, `PUT`/`PATCH`
  update, `DELETE` removes.
- **Status codes:** 200 OK, 201 created, 204 no content, 400 bad request, 404 not
  found, 500 server error.
- **JSON:** the text format for data, `{"run_id": "...", "track_key": 17}`.

### FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class SimilarRequest(BaseModel):
    run_id: str
    track_key: int
    k: int = 20

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/similar")
def similar(req: SimilarRequest):
    ...  # look up the vector, run the SQL from DATA-AND-DATABASE.md part 8
    return {"matches": []}
```

```bash
uvicorn app.api:app --port 8090 --reload
```

Open http://127.0.0.1:8090/docs. FastAPI draws a test page for every endpoint.

- **Pydantic models** check the request: a missing `track_key` gives a 422 error
  automatically.
- **Load the model once** at startup, not on every request. Loading takes seconds.

### Check yourself

1. Which method and status code fit "run a similar-person search"?
2. Why load the model at startup and not inside the endpoint?

---

## 12. Frontend: Next.js and live updates

Read `frontend/AGENTS.md` first: this Next.js version (16) has changes your
memory of older tutorials won't have. The docs are in
`frontend/node_modules/next/dist/docs/`.

| Idea | Meaning |
|---|---|
| **App Router** | Each folder in `src/app/` is a URL: `src/app/search/page.tsx` → `/search` |
| **Server component** | Default. Runs on the server, can fetch data directly, has no clicks. |
| **Client component** | Starts with `"use client"`. Runs in the browser, with buttons, state and effects. |
| **Proxy rewrite** | `next.config.ts` forwards `/api/*` to the Python backend, so the browser talks to one origin |
| **Tailwind** | Style with class names like `rounded-lg border p-2` |

### Live updates: SSE

**Server-Sent Events:** the browser opens one long request, and the server keeps
sending lines like `data: {...}` whenever something happens. It's one-way
(server → browser), which is all a live tracking page needs.

```tsx
"use client";
import { useEffect, useState } from "react";

export function LiveSightings() {
  const [events, setEvents] = useState<string[]>([]);
  useEffect(() => {
    const source = new EventSource("/api/events");
    source.onmessage = (e) => setEvents((old) => [e.data, ...old].slice(0, 50));
    return () => source.close();   // stop listening when the page closes
  }, []);
  return <ul>{events.map((e, i) => <li key={i}>{e}</li>)}</ul>;
}
```

### Check yourself

1. Does a component with an `onClick` need `"use client"`?
2. Why use SSE and not ask the server every second?

---

## 13. How semantic-va is built

Read [SEMANTIC-VA-NOTES.md](SEMANTIC-VA-NOTES.md) for the details. The short
version:

```text
video → decode → 2 s clips → Cosmos vector ──────────┐
      → detector → ByteTrack → tracks ───────────────┼─→ Postgres + pgvector → search API (8080) → web (3000)
      → best crop per track → SigLIP vector ─────────┘
```

### Read these files, in this order

| File | Why |
|---|---|
| `README.md` §0a, §11, §12 | Storage, search and the detector/tracker |
| `doc/SETUP-BACKEND.md`, `doc/RUN-MAC.md` | How your Mac runs it |
| `src/core/tracking.py` | The local tracker you'll build on |
| `src/core/detect.py` (crops part) | How the best crop is chosen and saved |
| `src/core/capture_time.py` | Recording time |
| `src/api/routes.py` (clips, crop) | The endpoints you call |
| `web/src/components/ClipTabs.tsx` | Where your button goes |
| `docker/initdb/*.sql` | The tables, and how new ones get added |
| `tools/migrate.py` | How schema changes reach existing databases |

### Its habits (copy them)

- Every number in a comment was **measured**, and the comment says where.
- Comments explain **why**, not what.
- Separate things stay separate: vector spaces, clocks, local and global IDs.

---

## 14. Glossary

| Term | Meaning |
|---|---|
| **AIN** | Adaptive instance normalisation; helps OSNet work on new cameras |
| **Crop** | The part of a frame inside a detection box |
| **Embedding** | A vector describing something; similar things get similar vectors |
| **False merge** | Two different people wrongly given one identity |
| **Gallery** | The set of crops you search in |
| **Global ID** | One identity across all cameras |
| **HNSW** | A fast, approximate nearest-neighbour index |
| **IoU** | Overlap of two boxes, 0–1 |
| **Local track** | One person followed inside one camera |
| **mAP** | Mean average precision; quality of a ranked list |
| **MTMC** | Multi-target multi-camera tracking |
| **PTS** | Presentation timestamp; media time inside a file |
| **Query** | The crop you search with |
| **Rank-1** | % of queries whose first result is correct |
| **ReID** | Re-identification: matching the same person across views |
| **SSE** | Server-Sent Events; the server pushes updates to the browser |
| **Topology** | Which cameras connect, and travel times between them |
| **Upsert** | Insert, or do nothing or update if the row exists |
| **Vector space** | The set of vectors one specific model checkpoint produces |
