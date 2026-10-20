# semantic-va: what mini-reid needs to know

Notes from reading semantic-va (branch `test/search`, 16 Oct 2026). These are
only the parts that matter for ReID. Paths are relative to the semantic-va repo.

## How a video becomes searchable

```text
video file / camera
  → decode → 2-second clips → Cosmos-Embed1 scene vector     (search by description)
  → detector (6 classes) → ByteTrack local tracker → tracks  (filters: has, count, dwell)
  → best crop per track → SigLIP 2 vector (1152 numbers)     (search by appearance: "red jacket")
```

The detector only knows six classes: `person`, `car`, `truck`, `bus`,
`motorcycle` and `bicycle`.

## A 2-second clip is a database row, not a video file

`config.yaml` → `clips: window_seconds: 2.0, stride_seconds: 1.0`. A new window
starts every second, so windows overlap (0–2 s, 1–3 s, 2–4 s, ...). A 5-minute
file gives about 300 clips. Cosmos-Embed1 looks at 8 frames from each window
(`model.num_frames: 8`).

Each clip is one row in `clip_vectors` (`src/core/index.py`): `id`, `run_id`,
`stream_id`, `clip_start`/`clip_end` (seconds into the file), `media_ts`,
`wall_ts`, `capture_ts`, and a 768-number `embedding`. **No video is cut out at
indexing time.** The video stays in its original file on the share, at
`stream_runs.url`.

**Searching saves nothing.** A search compares vectors and returns clip IDs.
There's no search history table, which is why mini-reid saves searches itself.

When you **view** a result, the API cuts from the original file on demand and
caches the result on local disk. These are derived files, safe to delete:

| What | Endpoint | Cached as | How |
|---|---|---|---|
| Thumbnail | `/clips/{id}/thumbnail` | `thumbnails/{clip_id}_{width}_{media_ts in ms}_v2.jpg` | One frame, JPEG. Cache capped at 512 MB. |
| Snippet | `/clips/{id}/snippet` | `snippets/{clip_id}_{media_ts in ms}_{seconds×10}.mp4` | 1–30 s either side, **copied** without re-encoding, so it starts at a keyframe (up to 2 s early on MEVA). Capped at 2 GB. |
| Playback | `/clips/{id}/recording` | `thumbnails/playback/{sha256}.mp4` | Up to 10 s, **re-encoded** to H.264 at max 960 px so AVI sources play in a browser |

Saved at other times, not at search:

- **Crops:** at indexing, one per track, on the share
- **Evidence:** only when an alert rule fires, under `evidence_dir`

## The pieces ReID connects to

| Piece | Where | What it means for ReID |
|---|---|---|
| Local tracker | `src/core/tracking.py` | ByteTrack with IoU matching only. **No ReID, on purpose**: the opening comment says so. It links one object from frame to frame in **one** camera. It cannot tell that two cameras saw the same person. |
| Track key | `tracks`, `track_appearance` tables | A track is identified by `(run_id, track_key)`. `track_key` restarts per run, so `track_key` alone is **not unique**. |
| Best crop | `src/core/detect.py` (`_keep_best_crops`, `_INSERT_APPEARANCE`) | Each track keeps **one** crop: the largest one seen over the whole track, at source resolution. When a bigger crop arrives, the old vector is cleared. |
| Crop files | `src/core/crops.py` | JPEGs live under the crop storage root (the `\\10.1.2.90\ai` share). The database stores relative paths. |
| Appearance vectors | `src/core/appearance.py` | SigLIP 2 (`google/siglip2-so400m-patch16-naflex`), 1152 numbers, stored as `halfvec(1152)` in `track_appearance`. Trained to match an image with a **text phrase**, not trained for identity. |
| Small crops | `web/src/components/CropCard.tsx` | Crops under **90 px** are flagged as unclear. The same applies to ReID: a tiny crop has too little detail to identify anyone. |
| Recording time | `capture_ts` on a clip, `src/core/capture_time.py` | The time the footage was **recorded**, read from the file name. It is a naive local time with no time zone. |
| Index time | `wall_ts` on a clip | The time the clip was **indexed**. For files this is NOT when the event happened (March 2018 footage was indexed in September 2026). **Never order cross-camera sightings by `wall_ts`.** |
| Media time | `first_pts`, `last_pts`, `best_pts` on a track | Seconds into the recording. Recording time of a sighting = file recording start + media offset. |

## HTTP endpoints mini-reid can use

The Python API runs on port 8080 (see `doc/RUN-MAC.md`). These need no GPU.

| Endpoint | Returns |
|---|---|
| `GET /clips/{clip_id}` | `ClipDetail`: the clip (`hit`), its `tracks` (each with `track_key`, `cls`, `has_crop`, `crop_px`, `best_ts`, `best_pts`, `first_pts`, `last_pts`), and `source_url` |
| `GET /clips/{clip_id}/tracks/{track_key}/crop` | The stored best crop as a JPEG. **204** (not 404) when the file is missing, for example when the share is not mounted. |
| `GET /clips/{clip_id}/similar` | Clips with a similar **scene** vector. This is not identity search. |
| `GET /streams` | Every stream (camera or file) in the index |

Type definitions for all of these are in `web/src/lib/api.ts`, mirroring
`src/core/models.py`.

## Where the "Find similar person" button goes later

The clip detail page `web/src/app/clips/[id]/page.tsx` renders
`web/src/components/ClipTabs.tsx`. Its `TrackRow` component shows every track
with its crop. The button belongs there, shown when `t.cls === "person"` and
`t.has_crop` is true. It would call a new endpoint such as
`GET /clips/{clip_id}/tracks/{track_key}/same-identity`.

## Rules to keep when merging into semantic-va

These come from the VIP-5 research document and from semantic-va's own design:

1. **Separate vector space.** OSNet-AIN gives 512 numbers. They go in their own
   table (for example `track_reid_samples` with `halfvec(512)`), never in the
   SigLIP `halfvec(1152)` column. Vectors from different models can't be compared,
   even when they have the same length.
2. **Local tracks and global people are different things.** Keep
   `(run_id, track_key)` as it is. Add `global_entities` and
   `entity_track_links` tables that link local tracks. Keep each link's evidence
   so a wrong link can be removed.
3. **Lazy torch imports.** `tests/test_core_boundary.py` checks that `src.core`
   imports without torch. Import torch inside functions, as `appearance.py`
   and `detect.py` already do.
4. **Search runs on CPU.** Searching an existing index needs no GPU. Embedding
   new crops can need one, but looking up stored vectors must not.
5. **Update the tracking rationale.** `tracking.py` says ReID is excluded on
   purpose. Adding ReID means updating that comment.
6. **Stable camera IDs.** Today a "stream" can be one file. Two files from
   the same physical camera must map to the same `camera_id`.

## Test footage

MEVA (`meva/` under the storage root, see `doc/DATASETS.md`) is multi-camera
footage of the same sites: 25 cameras, CC-BY-4.0. File names carry the
recording time and camera, for example `2018-03-07.15-15-00.15-20-00.bus.G331`
(camera G331). It is the best data in the index for practising cross-camera
matching.
