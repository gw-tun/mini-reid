# mini-reid — YOLO + FastReID CCTV VMS — Project Specification

This single specification combines the project proposal, MVP requirements, model decisions, CPU/GPU plan, Record Search, Live Follow, development phases, and acceptance criteria.

## Document map

- **Part I:** Project proposal and development plan.
- **Part II:** MVP technical recommendations.
- **Part III:** Record Search and Live Follow operating model.

---

# Part I — Project proposal and development plan
## Multi-Camera CCTV VMS with YOLO Detection and FastReID
### Project Proposal and Development Plan — MVP Finalization

**Document status:** Final MVP proposal  
**Primary execution target:** Local CPU-only development  
**Secondary execution target:** NVIDIA GPU acceleration  
**Frontend:** Next.js  
**Backend and AI:** Python  
**Stream control and recording:** MediaMTX  
**Database:** PostgreSQL-compatible Neon with optional pgvector  
**Language:** English

---

### 1. Executive summary

This project is a multi-camera CCTV Video Management System (VMS) extension that detects people and vehicles in RTSP streams, creates class-specific ReID embeddings, and searches for likely appearances across cameras and recordings.

The Minimum Viable Product (MVP) has two workflows:

1. **Record Search** — an operator selects a detected person or vehicle from a recording and searches other recordings for likely matches.
2. **Live Follow** — an operator selects a live track in one camera and receives ranked likely sightings from other cameras over time.

The system will use MediaMTX for RTSP path management and video recording, YOLO11m for person/vehicle detection, a local tracker for short-term continuity, and FastReID Stronger Baseline models for the first implementation:

- Person: `msmt17_sbs_R50-IBN.pth`
- Vehicle: `veri_sbs_R50-IBN.pth`

The implementation will be model-swappable from the beginning. A common ReID interface will allow OSNet-AIN or another model to be evaluated later without changing the search, tracking, database, or frontend layers.

The primary local development mode is CPU-only. The architecture must remain correct and usable without CUDA; GPU support will be added as an acceleration path rather than a hard dependency.

---

### 2. Goals and non-goals

#### 2.1 MVP goals

- Accept RTSP camera paths configured in MediaMTX.
- Record streams through MediaMTX.
- Sample frames from live streams and recordings.
- Detect at least `person` and `vehicle` classes with YOLO11m.
- Maintain short-term per-camera tracks.
- Generate separate Person ReID and Vehicle ReID embeddings.
- Store detections, tracks, embeddings, recordings, and match results.
- Search recorded video for likely same-person or same-vehicle appearances.
- Follow a selected live track and rank likely next camera sightings.
- Expose a clean Python API for inference and search.
- Present evidence with camera, timestamp, score, crop, and source recording.
- Keep model choice, embedding version, distance metric, and thresholds explicit.
- Run locally on CPU for development and testing.

#### 2.2 Deferred capabilities

The following are deliberately outside the first MVP unless required by evaluation:

- Automatic discovery of cameras.
- Dynamic camera registration from the UI.
- Face recognition or biometric identification.
- License plate recognition.
- Audio analytics.
- Fully automatic identity decisions without operator review.
- Distributed inference across multiple hosts.
- Online model retraining from operator feedback.
- Cloud video storage and cloud inference.
- End-to-end mobile applications.

---

### 3. Product scope and user stories

#### 3.1 Record Search

**As an operator**, I can open a recording, select a person or vehicle detection, and search a time range and camera set for likely matches.

The result should include:

- camera name and ID,
- timestamp,
- recording reference,
- thumbnail or crop,
- object class,
- detector confidence,
- ReID similarity,
- reranked confidence,
- model key and version,
- a link to seek the source recording near the match.

The first search implementation should be request/response based. A second refinement pass around high-ranking timestamps can improve evidence quality without decoding every frame at maximum rate.

#### 3.2 Live Follow

**As an operator**, I can select a live person or vehicle track in one camera and see the most likely recent sightings from other cameras.

The live-follow view should show:

- selected source camera and local track ID,
- last-seen timestamp,
- candidate camera ranking,
- current similarity and rerank score,
- confidence margin over the next candidate,
- evidence crop,
- temporal continuity,
- a manual confirm or reject action.

Live follow is an operator-guided workflow. A ReID score alone must not be treated as proof of identity.

---

### 4. System architecture

```text
RTSP cameras
    │
    ▼
MediaMTX
    ├── RTSP path control
    ├── stream relay
    └── recording output
          │
          ├── live frame sampler
          └── recording frame reader
                    │
                    ▼
             Python AI service
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   YOLO11m      local tracker   crop quality gate
        │           │           │
        └───────────┴───────────┘
                    │
                    ▼
          class-specific ReID
        ┌───────────┴───────────┐
        ▼                       ▼
 Person ReID              Vehicle ReID
 MSMT17 SBS R50-IBN       VeRi SBS R50-IBN
        │                       │
        └───────────┬───────────┘
                    ▼
       vector search + temporal reranking
                    │
                    ▼
          PostgreSQL / Neon API
                    │
                    ▼
             Next.js frontend
```

#### 4.1 Responsibilities by service

| Component | Responsibility |
|---|---|
| MediaMTX | RTSP paths, stream relay, recording, time-aligned media access |
| Python ingestion | Read live/recorded frames, sampling, backpressure |
| YOLO worker | Person and vehicle detection |
| Tracker | Short-term identity continuity inside one camera |
| ReID service | Crop preprocessing, embedding extraction, similarity search |
| Match service | Top-K retrieval, temporal/camera reranking, threshold policy |
| PostgreSQL/Neon | Durable metadata, embeddings, tracks, search results |
| Next.js frontend | Camera/recording search, live-follow UI, evidence display |

---

### 5. Technology decisions

#### 5.1 Frontend

- Next.js with TypeScript.
- Server components or API routes only where they simplify the UI; the inference service remains Python.
- WebSocket or Server-Sent Events for live-follow updates.
- HTML5 video or a MediaMTX-compatible playback path for evidence and live views.
- No model inference in the browser for the MVP.

#### 5.2 Backend and AI

- Python 3.11 or the version required by the selected model dependencies.
- FastAPI for the HTTP API.
- Pydantic for request/response validation.
- OpenCV or PyAV for frame access.
- PyTorch for model execution.
- FastReID for the initial Person and Vehicle ReID models.
- A model adapter for future Torchreid OSNet-AIN support.
- NumPy for vector operations.
- Optional ONNX Runtime CPU path after the PyTorch baseline is validated.

#### 5.3 Stream and recording

- MediaMTX with hard-coded RTSP paths for the MVP.
- One explicit path per camera.
- Recording filenames must encode camera ID and time or be referenced by a recording metadata table.
- The AI service must not assume that a live stream is the same thing as a recording file; both are input sources with a common frame interface.

#### 5.4 Storage

- PostgreSQL-compatible Neon database for metadata and search state.
- pgvector is optional for the first CPU-only prototype, but recommended once the matching flow is stable.
- Store media files and crops on local disk, NAS, or object storage; do not store large video blobs in PostgreSQL.
- Store the media URI and crop URI in database records.

#### 5.5 Reference implementations

- [FastReID official repository](https://github.com/JDAI-CV/fast-reid)
- [FastReID MSMT17 SBS R50-IBN configuration](https://github.com/JDAI-CV/fast-reid/blob/master/configs/MSMT17/sbs_R50-ibn.yml)
- [FastReID VeRi SBS R50-IBN configuration](https://github.com/JDAI-CV/fast-reid/blob/master/configs/VeRi/sbs_R50-ibn.yml)
- [FastReID Base-SBS configuration](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-SBS.yml)
- [OSNet-AIN research paper](https://arxiv.org/abs/1910.06827)
- [Torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)
- [MediaMTX repository](https://github.com/bluenviron/mediamtx)
- [Ultralytics YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [pgvector repository](https://github.com/pgvector/pgvector)

---

### 6. Model and algorithm choices

#### 6.1 YOLO detection

YOLO11m will be the detector for the MVP.

Required outputs:

- class label,
- bounding box,
- detector confidence,
- frame timestamp,
- camera ID,
- source type (`live` or `recording`).

The system should support a class allow-list initially containing `person` and `vehicle`. It should be possible to add class-specific thresholds later.

#### 6.2 Frame sampling

The first operational baseline is:

- **5 sampled FPS** for moving or busy scenes.
- **2 sampled FPS** for static scenes, higher camera counts, or CPU pressure.
- Use tracking between detections where possible.
- Do not run YOLO at the raw camera frame rate by default.

For CPU-only development, begin with one camera and 2 FPS. Validate correctness first, then increase sampling or camera count. CPU inference should use bounded queues and stale-frame dropping so latency does not grow without limit.

#### 6.3 Tracking

The tracker provides short-term continuity within one camera. It is not the cross-camera identity system.

Recommended behavior:

- Create a local track after a confirmed detection.
- Associate detections across sampled frames.
- Maintain a track age and expiry window.
- Keep an appearance gallery for each active track.
- Do not create a new ReID embedding for every raw frame.
- Refresh the embedding every 3–5 sampled frames or when the crop changes materially.
- Reject low-quality crops before adding them to a gallery.

A simple CPU-friendly baseline can begin with IoU plus centroid/velocity association. A stronger tracker can be added after the detector pipeline is stable.

#### 6.4 Person ReID

Initial model:

```text
FastReID MSMT17 SBS R50-IBN
models/fastreid/msmt17_sbs_R50-IBN.pth
models/fastreid/configs/msmt17_sbs_R50-IBN.yml
```

The official FastReID Base-SBS configuration uses a ResNet-50 feature dimension of 2048, an embedding head, Generalized Mean Pooling, a non-local block, and Cross Entropy plus Triplet losses. Confirm the actual runtime output tensor shape in the inference adapter before creating the production index.

#### 6.5 Vehicle ReID

Initial model:

```text
FastReID VeRi SBS R50-IBN
models/fastreid/veri_sbs_R50-IBN.pth
models/fastreid/configs/veri_sbs_R50-IBN.yml
```

Person and vehicle embeddings must never be compared with one another. They must have separate galleries, indexes, thresholds, and model metadata.

#### 6.6 OSNet-AIN as a swap candidate

OSNet-AIN is not required for the first implementation, but the architecture must permit it as a later backend.

Typical Torchreid OSNet-AIN x1.0 output:

- 512-dimensional feature,
- cosine distance/similarity,
- 256 × 128 input in the official model-zoo setup,
- substantially lower compute cost than a ResNet-50 ReID model.

The OSNet-AIN and FastReID embeddings are different spaces. They must not share a gallery directly. If the model changes, re-embed historical candidates or maintain a versioned index per model.

#### 6.7 Similarity and threshold policy

Both model families can use L2-normalized features with cosine similarity:

```python
query = query / ||query||
gallery = gallery / ||gallery||
similarity = query @ gallery.T
```

There is no universal production threshold. Thresholds must be calibrated with representative local CCTV data because score distributions change with:

- camera pair,
- lighting,
- crop size,
- occlusion,
- camera angle,
- gallery size,
- model version,
- detector quality.

The MVP matching policy should be:

1. Retrieve Top-K candidates.
2. Apply a model-specific similarity gate.
3. Rerank with detector confidence and crop quality.
4. Apply temporal continuity and camera topology rules.
5. Require a score margin over the next candidate for auto-follow.
6. Require two or three consistent sightings before showing a strong live-follow confirmation.
7. Keep an operator review state for ambiguous matches.

Candidate generation may use Top-K without a hard threshold. The confirmation threshold should be selected from a local positive/negative score distribution and a target false-match rate.

---

### 7. Model-swappable design

The application must not import FastReID or Torchreid throughout the codebase. Keep framework-specific behavior behind an adapter.

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class ReIDOutput:
    vectors: "Tensor"
    model_key: str
    model_version: str
    feature_dim: int
    metric: str
    normalized: bool

class ReIDBackend(Protocol):
    def load(self) -> None: ...
    def embed(self, crops: list["Image"]) -> ReIDOutput: ...
    def distance(self, query, gallery): ...
    def close(self) -> None: ...
```

A registry should select the backend from configuration:

```yaml
person:
  backend: fastreid
  model_key: msm17_sbs_r50_ibn
  weights: models/fastreid/msmt17_sbs_R50-IBN.pth
  config: models/fastreid/configs/msmt17_sbs_R50-IBN.yml
  input_size: [384, 128]
  metric: cosine
  normalize: true

vehicle:
  backend: fastreid
  model_key: veri_sbs_r50_ibn
  weights: models/fastreid/veri_sbs_R50-IBN.pth
  config: models/fastreid/configs/veri_sbs_R50-IBN.yml
  input_size: [384, 128]
  metric: cosine
  normalize: true
```

Later, OSNet-AIN can be added by changing the backend and model metadata rather than changing the search service.

---

### 8. Database design

#### 8.1 Core tables

#### `cameras`

| Column | Purpose |
|---|---|
| `id` | Stable camera identifier |
| `name` | Operator-facing name |
| `mediamtx_path` | Hard-coded MediaMTX path |
| `location_label` | Human-readable location |
| `timezone` | Timestamp interpretation |
| `active` | Whether the path is enabled |

#### `recordings`

| Column | Purpose |
|---|---|
| `id` | Recording ID |
| `camera_id` | Source camera |
| `started_at` | Start timestamp |
| `ended_at` | End timestamp |
| `media_uri` | File or object-storage URI |
| `status` | Ingest/indexing status |

#### `detections`

| Column | Purpose |
|---|---|
| `id` | Detection ID |
| `camera_id` | Source camera |
| `recording_id` | Nullable for live detections |
| `track_id` | Local camera track |
| `timestamp` | Frame timestamp |
| `class_name` | `person` or `vehicle` |
| `bbox` | Bounding box coordinates |
| `detector_score` | YOLO confidence |
| `crop_uri` | Evidence crop |

#### `tracks`

| Column | Purpose |
|---|---|
| `id` | Durable track ID |
| `camera_id` | Camera scope |
| `local_track_id` | Tracker-generated ID |
| `class_name` | Person or vehicle |
| `started_at` | Track start |
| `ended_at` | Track end |
| `quality_summary` | Aggregated quality metadata |

#### `embeddings`

| Column | Purpose |
|---|---|
| `id` | Embedding ID |
| `track_id` | Source track |
| `model_key` | Exact model identifier |
| `model_version` | Weight/config version |
| `backend` | `fastreid`, `torchreid`, etc. |
| `feature_dim` | Vector dimension |
| `distance_metric` | Usually `cosine` |
| `normalized` | Whether L2 normalization was applied |
| `vector` | pgvector value or external vector URI |
| `quality_score` | Crop quality gate result |
| `created_at` | Extraction timestamp |

#### `matches`

| Column | Purpose |
|---|---|
| `id` | Match result |
| `query_id` | Search or follow query |
| `source_track_id` | Selected source track |
| `candidate_track_id` | Candidate track |
| `similarity_score` | Raw model similarity |
| `rerank_score` | Combined score |
| `rank` | Candidate rank |
| `decision_state` | `candidate`, `review`, `confirmed`, `rejected` |
| `created_at` | Match time |

#### 8.2 Indexing

Start with SQL filtering and a bounded in-memory gallery during CPU-only development. Add pgvector once the matching behavior is correct and the gallery size justifies it.

Recommended indexes:

- `detections(camera_id, timestamp)`
- `detections(class_name, timestamp)`
- `tracks(camera_id, started_at)`
- `embeddings(model_key, model_version, class_name)`
- vector index per model key and feature dimension

Do not create one vector index containing mixed model versions or mixed dimensions.

---

### 9. Software requirements

#### 9.1 Functional requirements

| ID | Requirement | Priority |
|---|---|---|
| FR-01 | Load hard-coded camera paths from MediaMTX configuration | Must |
| FR-02 | Record or access recordings with camera/time metadata | Must |
| FR-03 | Sample live and recorded frames | Must |
| FR-04 | Detect person and vehicle objects with YOLO11m | Must |
| FR-05 | Maintain per-camera local tracks | Must |
| FR-06 | Generate class-specific ReID embeddings | Must |
| FR-07 | Search recordings by source detection/track | Must |
| FR-08 | Return ranked cross-camera candidates | Must |
| FR-09 | Start and stop live follow for a selected track | Must |
| FR-10 | Show evidence crop and source timestamp | Must |
| FR-11 | Persist model and threshold metadata | Must |
| FR-12 | Support CPU-only execution | Must |
| FR-13 | Support GPU execution through configuration | Should |
| FR-14 | Support model backend swap through an adapter | Should |
| FR-15 | Export or inspect match results | Should |

#### 9.2 Non-functional requirements

- The API must return validation errors instead of silently accepting malformed boxes or timestamps.
- Inference queues must be bounded.
- Stale frames may be dropped when CPU inference falls behind.
- Every result must be traceable to a camera, timestamp, recording, model, and evidence crop.
- The frontend must remain usable when the AI worker is offline.
- GPU must be optional at startup.
- Logs must include camera ID, source type, frame timestamp, queue size, latency, and model key.
- Database migrations must be version-controlled.
- Configuration must be environment-based; secrets must not be committed.
- Retention and access controls must be configurable for recordings, crops, and embeddings.

---

### 10. API plan

#### 10.1 Camera and recording endpoints

```text
GET  /api/cameras
GET  /api/cameras/{camera_id}/health
GET  /api/recordings?camera_id=&from=&to=
GET  /api/recordings/{recording_id}
```

#### 10.2 Detection and search endpoints

```text
GET  /api/tracks/{track_id}
GET  /api/tracks/{track_id}/evidence
POST /api/search/record
GET  /api/search/{query_id}
```

Example record search request:

```json
{
  "source_track_id": "track_000123",
  "class_name": "person",
  "camera_ids": ["cam_01", "cam_02", "cam_05"],
  "from": "2026-10-16T08:00:00Z",
  "to": "2026-10-16T12:00:00Z",
  "top_k": 20
}
```

#### 10.3 Live follow endpoints

```text
POST /api/follow
GET  /api/follow/{follow_id}
POST /api/follow/{follow_id}/stop
GET  /api/follow/{follow_id}/events
```

Use WebSocket or SSE for `/events` when the live-follow flow is implemented.

#### 10.4 Operations endpoints

```text
GET /health
GET /ready
GET /metrics
GET /api/models
```

Metrics should include:

- frames received,
- frames sampled,
- detections per minute,
- active tracks,
- embeddings per minute,
- average inference latency,
- queue depth,
- dropped frames,
- database write latency,
- match query latency.

---

### 11. CPU-only development plan

#### 11.1 CPU mode principles

- Begin with one recorded clip and one camera.
- Use 2 FPS sampling initially.
- Use small batches, commonly 1–4 crops for a CPU smoke test.
- Avoid decoding full-resolution frames when a lower decode size is sufficient.
- Keep live queues bounded.
- Persist evidence asynchronously after inference.
- Use an in-memory gallery before introducing a vector database.
- Keep GPU-specific code behind a device selector.

#### 11.2 Device configuration

```yaml
runtime:
  device: cpu
  detector_device: cpu
  reid_device: cpu
  batch_size: 1
  sample_fps: 2
  max_queue_size: 32
  drop_stale_frames: true
```

GPU mode should be a configuration change:

```yaml
runtime:
  device: cuda:0
  detector_device: cuda:0
  reid_device: cuda:0
  batch_size: 16
  sample_fps: 5
```

The application should report the active device and model at startup.

#### 11.3 CPU-first sequence

1. Validate model loading with one image.
2. Validate YOLO detection on a short clip.
3. Validate crop extraction and preprocessing.
4. Validate ReID embedding shape and normalization.
5. Validate local tracking.
6. Validate in-memory Top-K matching.
7. Persist detections and tracks.
8. Add recording search.
9. Add live follow.
10. Add GPU acceleration and benchmark comparison.

---

### 12. Hardware requirements

#### 12.1 Development hardware — CPU-only baseline

| Component | Minimum | Recommended |
|---|---:|---:|
| CPU | 4 physical cores | 8+ physical cores |
| RAM | 16 GB | 32 GB |
| Storage | 50 GB free SSD | 200+ GB SSD for clips and crops |
| GPU | None | Optional NVIDIA GPU |
| Network | Stable LAN for RTSP | Gigabit LAN |
| OS | Windows or Linux | Linux preferred for deployment parity |

CPU-only development is suitable for correctness, API, database, UI, and short-clip validation. It should not be used to claim final multi-camera real-time capacity until measured with the target CPU.

#### 12.2 Secondary GPU profile

| Component | Development acceleration | Multi-camera target profile |
|---|---|---|
| GPU | NVIDIA GPU with 6–8 GB VRAM | NVIDIA GPU with 12+ GB VRAM |
| CUDA | Version compatible with PyTorch | Pinned and tested in deployment image |
| Runtime | PyTorch CUDA | PyTorch CUDA, optional TensorRT/ONNX |
| Batch | Small crop batches | Per-class micro-batching |

GPU capacity depends on YOLO input resolution, crop count, sampled FPS, number of cameras, and whether recording search runs concurrently with live inference.

#### 12.3 Storage sizing

Estimate storage separately for:

- original recordings,
- thumbnails/crops,
- database metadata,
- vector indexes,
- logs and metrics.

Keep a retention policy from the first prototype even if the local MVP uses short clips.

---

### 13. Development phases

#### Phase 0 — Repository and environment setup

**Deliverables**

- Python environment.
- Next.js application shell.
- MediaMTX local configuration with hard-coded camera paths.
- Database connection and migrations.
- Model directory validation.
- CPU device startup path.

**Exit criteria**

- A health endpoint returns model, device, database, and MediaMTX status.
- A developer can run the stack locally without a GPU.

#### Phase 1 — Detection and evidence

**Deliverables**

- Read a recorded clip.
- Sample at 2 FPS.
- Run YOLO11m.
- Save detections and crops.
- Render a basic evidence list.

**Exit criteria**

- Every saved detection has a timestamp, bbox, class, confidence, and crop reference.
- A clip can be replayed and linked to detections.

#### Phase 2 — Local tracking

**Deliverables**

- Per-camera track IDs.
- Track lifecycle and expiry.
- Quality score and best-crop selection.
- Track gallery in memory.

**Exit criteria**

- A person or vehicle remains associated with a local track across sampled frames.
- Low-quality crops do not become gallery exemplars.

#### Phase 3 — ReID embeddings and indexing

**Deliverables**

- FastReID Person adapter.
- FastReID Vehicle adapter.
- Batch crop inference.
- L2 normalization.
- Model/version metadata.
- In-memory Top-K search.

**Exit criteria**

- Person vectors only search the person gallery.
- Vehicle vectors only search the vehicle gallery.
- Embedding dimensions and score ranges are logged and tested.

#### Phase 4 — Record Search

**Deliverables**

- Source track selection.
- Camera and time filters.
- Candidate retrieval.
- Temporal and quality reranking.
- Recording seek links.

**Exit criteria**

- An operator can select a source track and receive ranked candidates from other recordings.
- Each candidate links to evidence.

#### Phase 5 — Live Follow

**Deliverables**

- Select a live track.
- Maintain a short-lived follow gallery.
- Stream candidate updates through SSE/WebSocket.
- Manual confirm/reject actions.

**Exit criteria**

- The UI updates without a full page refresh.
- The system reports last-seen time and confidence state.
- Ambiguous matches remain reviewable rather than silently auto-confirmed.

#### Phase 6 — GPU acceleration and model comparison

**Deliverables**

- CUDA device option.
- Batch-size tuning.
- YOLO and ReID latency benchmark.
- OSNet-AIN adapter.
- FastReID versus OSNet-AIN A/B evaluation.

**Exit criteria**

- CPU and GPU modes produce the same schema and comparable result semantics.
- Model-specific galleries and thresholds are versioned.

---

### 14. Testing and evaluation

#### 14.1 Unit tests

- bbox validation and coordinate conversion,
- frame timestamp conversion,
- crop extraction,
- input normalization,
- embedding shape validation,
- L2 normalization,
- cosine similarity,
- Top-K ranking,
- threshold and margin policies,
- camera/time filtering,
- database migrations,
- model registry resolution.

#### 14.2 Integration tests

- MediaMTX path to frame sampler,
- recording file to detections,
- detections to tracks,
- tracks to embeddings,
- embeddings to match results,
- API to frontend evidence view,
- CPU startup with no CUDA available.

#### 14.3 Evaluation dataset

Create a small labelled local validation set from representative target cameras:

- daylight,
- night or low light,
- occlusion,
- crowded scenes,
- small targets,
- different camera angles,
- camera pairs with known transitions,
- camera pairs with no expected transition.

#### 14.4 Metrics

- Person and vehicle detection precision/recall.
- Track fragmentation rate.
- Top-1 and Top-5 ReID retrieval.
- mAP on a labelled local set where practical.
- False matches per camera-hour.
- Cross-camera handoff accuracy.
- Median and p95 inference latency.
- Queue depth and dropped-frame rate.
- CPU utilization and RAM usage.
- GPU memory and throughput in the accelerated mode.

Do not use public MSMT17 or VeRi scores as a substitute for local CCTV validation. They are useful baselines, but target-camera performance depends heavily on crop quality, camera style, and scene conditions.

---

### 15. Acceptance criteria for the MVP

The MVP is complete when all of the following are true:

- A local CPU-only installation starts successfully.
- MediaMTX accepts the configured RTSP paths.
- At least one recorded clip can be processed end to end.
- YOLO outputs person and vehicle detections with timestamps.
- Tracks are created and expire correctly.
- FastReID generates separate person and vehicle embeddings.
- Embeddings are stored with model/version metadata.
- Record Search returns ranked candidates with evidence.
- Live Follow produces event updates for a selected track.
- Search results can be filtered by camera and time range.
- The UI identifies score, threshold policy, model, and timestamp.
- CPU-only mode remains functional when no CUDA device is present.
- A small local validation report records accuracy, latency, and false-match behavior.
- Model-specific indexes prevent mixing FastReID and OSNet-AIN vectors.

---

### 16. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| CPU inference falls behind | Live latency grows | Start at 2 FPS, bound queues, drop stale frames |
| Detector crop is too small | ReID false matches | Quality gate, minimum crop size, evidence review |
| Similar clothing or vehicles | Identity confusion | Gallery aggregation, temporal rules, camera topology |
| Lighting/domain shift | Cross-camera score degradation | Local calibration, OSNet-AIN A/B test, camera-specific thresholds |
| Mixed embedding spaces | Invalid match scores | Versioned model-specific galleries and indexes |
| Recording timestamps drift | Incorrect handoff order | Normalize timestamps and verify MediaMTX recording metadata |
| Database vector search too early | Complexity without value | Start in memory, add pgvector after behavior is stable |
| GPU-only assumptions | Development blocked | CPU device path and no-CUDA integration test |
| Large media retention | Storage exhaustion | Retention policy, URI-based storage, scheduled cleanup |
| Privacy and access exposure | Sensitive data leakage | Authentication, role checks, audit logs, encryption in transit, retention limits |

---

### 17. Security and privacy requirements

The system processes CCTV imagery and appearance embeddings. The MVP should include:

- authenticated operator access,
- role-based access for recordings and search,
- audit logging of searches and confirmations,
- encrypted database connections,
- protected crop and recording paths,
- configurable retention for recordings and embeddings,
- no public exposure of MediaMTX paths,
- explicit documentation that ReID results are similarity rankings, not identity proof.

Legal and organizational requirements vary by deployment location. Retention, access, and operator workflows must be reviewed before production use.

---

### 18. Final implementation decision

The first implementation will use:

```text
Detector:       YOLO11m
Person ReID:    FastReID MSMT17 SBS R50-IBN
Vehicle ReID:   FastReID VeRi SBS R50-IBN
Tracker:        CPU-friendly local tracker baseline
Sampling:       2 FPS in CPU development, 5 FPS after measurement
Distance:       L2-normalized cosine similarity
Search:         Top-K retrieval + temporal/camera/quality reranking
Database:       PostgreSQL-compatible Neon
Vector search:  In-memory first, pgvector after validation
Streams:        MediaMTX with hard-coded RTSP paths
Frontend:       Next.js
Backend:        Python FastAPI + PyTorch
Device:         CPU required, CUDA optional
```

The codebase should preserve a model-independent ReID interface from day one. This allows OSNet-AIN to be tested later for lower compute cost or stronger cross-camera generalization without rewriting the VMS workflow.

---

### 19. Open decisions before implementation

These do not block the proposal, but they must be fixed before capacity testing:

1. Target number of simultaneous cameras.
2. Camera resolution and nominal FPS.
3. CPU model and available RAM.
4. Optional NVIDIA GPU model and VRAM.
5. Recording location and retention period.
6. Minimum acceptable crop size for people and vehicles.
7. Expected camera-to-camera transition time windows.
8. Authentication and operator roles.
9. Whether pgvector is enabled in the initial Neon database.
10. Whether live playback uses a browser-compatible MediaMTX output path.


### Database design companion

See [Database and Performance Specification](DATA_AND_PERFORMANCE.md) for database performance theory, pgvector algorithm choices, query patterns, retention, partitioning, and the PostgreSQL/Neon schema.


### Record Search and Live Follow operating model

The Record Search and Live Follow operating model is included in this document, covering segment indexing, midnight backfill, route-aware camera activation, shared workers, and live-follow performance controls.


### 20. Operational execution decisions (clarified)

#### Record Search indexing

Record Search uses nearline indexing per completed MediaMTX recording segment. A segment-finalized event creates an asynchronous indexing job; the system does not wait until midnight to process the whole day and does not run ReID on every raw frame.

The indexing unit is a confirmed local track:

```text
segment finalized → sampled YOLO detection → local tracking
→ quality-approved crops → class-specific ReID gallery
→ track prototype/vector index update
```

Use 1–5 minute segments where practical. CPU development starts at 2 FPS; GPU validation starts at 5 FPS. ReID is refreshed every 3–5 sampled track observations or when the crop changes materially.

The midnight job is a maintenance job for failed segments, indexing gaps, retries, model-version re-embedding, and daily reports. It must not duplicate already completed jobs.

#### Live Follow routing

Live Follow keeps a low-rate sentinel on every active camera for stream health, YOLO detection, and local tracking. When a target is selected, the system activates a higher-rate ReID burst only on cameras that are plausible next destinations in the directed camera graph.

For a T-junction:

```text
CAM-01 target at T0
  ├── CAM-02 ReID window: T0 + min_time → T0 + max_time
  └── CAM-03 ReID window: T0 + min_time → T0 + max_time
```

Both branches may run in parallel because the target can choose either path. Once CAM-02 confirms the target, the CAM-03 branch can be reduced or cancelled. If neither branch confirms, the system can extend the window, expand to second-hop cameras, or use a bounded global fallback.

The graph is a routing and time-window prioritization layer. ReID similarity, detector confidence, crop quality, temporal continuity, and operator confirmation still determine the final match state.

#### Shared worker requirement

The system must not load a new YOLO or ReID model process per camera. Use shared detector and ReID workers with per-camera queues and micro-batching across cameras. Keep model weights warm and activate/deactivate camera work through scheduling and queue priority.

### Interactive project briefing

Open [Interactive project briefing](../index.html) for the interactive architecture, Record Search, Live Follow, T-junction branch scheduling, performance controls, model registry, database overview, and development plan.


---

# Part II — MVP technical recommendations
## VMS ReID MVP — Technical Recommendations

### Scope

The minimum valuable product is a two-mode multi-camera search system:

1. **Record search:** select a detected person or vehicle from a recording, then search other recordings for likely sightings.
2. **Live follow:** select a live track in one camera and rank the next likely camera sightings over time.

The shared pipeline is:

`RTSP camera → MediaMTX → sampled frames → YOLO11m → local tracker → class-specific FastReID → candidate search → evidence API → Next.js UI`

MediaMTX should own RTSP path configuration and recording. The Python service should own inference and matching. Neon/PostgreSQL should store searchable metadata and vectors; keep video files and image crops in object storage or a mounted media volume rather than database blobs.

### Sampling recommendation

Start at **5 sampled frames per second** for moving scenes and use **2 FPS** for static scenes or when GPU pressure grows. Do not run detection at the camera’s raw frame rate by default.

A practical first configuration:

- YOLO detection: 5 FPS for people and vehicles in active scenes.
- Local tracker: update between detections so motion remains smooth.
- FastReID embedding: every 3–5 sampled frames, or sooner when the crop changes materially.
- Quality gate: skip crops that are too small, blurred, heavily occluded, or outside the expected aspect ratio.
- Recording search: start at 2 FPS, then resample around candidate timestamps for a second pass.

The correct setting depends on scene speed, camera count, GPU, and the smallest target size. Measure recall on short labelled clips before choosing a global default.

### ReID performance controls

1. **Batch crops.** Batch person crops together and vehicle crops together before FastReID inference.
2. **Separate class pipelines.** Use `msmt17_sbs_R50-IBN` for people and `veri_sbs_R50-IBN` for vehicles. Never compare vectors across classes.
3. **Use a track gallery.** Keep several quality-filtered embeddings per track instead of one vector. Aggregate with a median or top-quality score.
4. **Use two-stage search.** First retrieve a small candidate set with vector similarity. Then rerank with time continuity, camera topology, crop quality, detector confidence, and appearance consistency.
5. **Keep the hot path in memory.** Cache active track galleries and recent camera candidates in the Python worker. Persist durable sightings asynchronously.
6. **Control concurrency.** Bound the per-camera queue, drop stale frames, and expose queue depth and inference latency as metrics.
7. **Use explicit thresholds.** Store the model key, embedding version, detector confidence, ReID score, and threshold used for every result.
8. **Do not auto-follow on score alone.** Require a confidence margin over the second-best candidate and a short temporal confirmation window.

### Suggested PostgreSQL entities

- `cameras`: camera id, name, location label, MediaMTX path, timezone, active flag.
- `recordings`: camera id, start/end time, media URI, ingest status.
- `detections`: recording/live session id, camera id, timestamp, class, bbox, detector confidence, crop URI.
- `tracks`: camera id, start/end time, class, local track id, quality summary.
- `embeddings`: track/detection id, model key, vector reference, quality score, created at.
- `matches`: query id, source track, candidate track, ReID score, rerank score, rank, decision state.

If using a vector extension, use pgvector for the first implementation. Keep the vector dimension and model key in the row so model migrations are explicit.

### API shape for the MVP

- `GET /cameras`
- `GET /recordings?camera_id=&from=&to=`
- `POST /search/record` with a source detection/track and time/camera filters
- `POST /follow/start` with a live camera and track id
- `GET /follow/{id}/events`
- `GET /matches/{id}`
- `GET /health` and `GET /metrics`

Use WebSocket or Server-Sent Events for live-follow updates. Keep record search request/response based until the ranking flow is stable.

### Evaluation plan

Create a small, labelled validation set from the target cameras. Measure:

- detection precision and recall by class,
- track fragmentation,
- top-1 and top-5 ReID retrieval,
- false matches per camera-hour,
- end-to-end latency from frame arrival to match event,
- GPU memory and queue depth at 2 FPS and 5 FPS.

Tune on representative scenes: daylight, night, occlusion, camera handoff, crowded areas, and camera pairs with no direct line of travel.

### Open decisions before production

- GPU model and expected camera count per worker.
- Target minimum crop size for people and vehicles.
- Whether recordings are stored locally, on NAS, or object storage.
- Camera topology and expected handoff windows.
- Retention period and access controls for biometric-like embeddings.
- Whether operators need a manual merge/split action for tracks.

These decisions do not block the MVP demo, but they determine capacity and the threshold policy for real deployments.

### Operational indexing and live-follow execution

#### Record Search

Process each completed MediaMTX segment asynchronously. Do not wait for the full day and do not create embeddings from every raw frame.

Recommended sequence:

1. MediaMTX segment closes.
2. A `recording_index_job` is queued.
3. Frames are sampled at 2 FPS for CPU development or 5 FPS after GPU measurement.
4. YOLO detects objects and a local tracker creates per-camera tracks.
5. Quality-approved crops are selected.
6. Person and vehicle ReID embeddings are extracted separately.
7. Track galleries and prototypes are stored and indexed.
8. The job is marked complete with detector and ReID model versions.

The midnight task is reserved for retries, missing-segment checks, failed jobs, re-embedding, and daily reports.

#### Live Follow

Keep a low-rate sentinel on active cameras. When a live track is selected, schedule ReID bursts on the outgoing edges of the source camera's directed transition graph.

For a T-junction, both downstream cameras are candidates and receive a bounded ReID window. A match on one branch can cancel or reduce the other branch. The system should expand to second-hop cameras or a global low-rate fallback when the route is uncertain.

The high-rate workflow is therefore activated by a target-specific time window; it is not permanently run at high rate on every camera.

#### Runtime performance rules

- Share one YOLO worker and one worker per ReID class/model instead of one process per camera.
- Use bounded per-camera queues and drop stale frames when overloaded.
- Batch crops across cameras.
- Run ReID every 3–5 sampled track observations or when the appearance changes materially.
- Give active-follow work higher priority than background indexing.
- Measure queue depth, frame age, dropped frames, detector latency, ReID latency, and active branch count.
---

# Part III — Record Search and Live Follow operating model
## Record Search and Live Follow Operating Model

### Decision summary

#### Record Search

Use **nearline indexing per completed recording segment**, not one full-day batch at midnight and not a new inference workflow for every raw frame.

Recommended flow:

~~~text
MediaMTX segment finalized
  → enqueue indexing job
  → sample frames at configured FPS
  → YOLO detection
  → local tracking
  → crop-quality gate
  → ReID embeddings for selected track samples
  → batch database write
  → vector index update
~~~

Use the midnight job for retry, gap detection, failed segments, threshold recalibration, and optional re-embedding after a model change.

#### Live Follow

Do not start a full YOLO + ReID process separately for every camera when an operator selects a target.

Use three layers:

1. **Always-on camera layer:** stream health, low-rate sampling, YOLO, and local tracking.
2. **On-demand ReID layer:** activate ReID on the predicted camera group and time window.
3. **Fallback layer:** keep a low-rate global sentinel or expand to all cameras when the route graph is unknown or no candidate is found.

Load one detector/ReID model per worker and batch crops from multiple cameras. Do not load a separate model instance for every camera.

---

### 1. Record Search processing options

| Strategy | Data availability | Compute behavior | Recommendation |
|---|---|---|---|
| Process every raw frame immediately | Fastest possible availability | Highest compute and storage | Do not use for the MVP |
| Process every sampled frame while recording is live | Near real-time | Continuous CPU/GPU load | Use only if live indexing is required |
| Process each completed segment | Usually 1–5 minutes behind live | Predictable asynchronous load | Recommended |
| Process one full day at midnight | Results available next day | Large batch and slow recovery | Use for backfill/retry only |
| Process only when a search is requested | No indexing cost up front | First search becomes slow | Fallback for old/unindexed records |

A MediaMTX recording should be treated as a sequence of completed segments. The indexing trigger should be the segment-finalized event, or a short polling job that detects a newly closed segment.

---

### 2. Recommended Record Search pipeline

#### Step 1: Register the segment

Create a recording row with:

- camera ID;
- segment media URI;
- start and end timestamps;
- duration;
- checksum or stable source key;
- indexing status.

Use a job table such as:

~~~text
recording_index_jobs
  job_id
  recording_id
  requested_from
  requested_to
  detector_version
  person_model_version
  vehicle_model_version
  status
  attempt_count
  last_error
  created_at
  completed_at
~~~

A deterministic job key prevents duplicate processing when the watcher retries.

#### Step 2: Decode at a sampling rate

Do not decode and infer every raw frame by default.

Initial settings:

- CPU development: 2 FPS;
- GPU validation: 5 FPS;
- busy scenes: increase only after measurement;
- static scenes or CPU pressure: reduce sampling.

If a 30 FPS segment is sampled at 2 FPS, the system decodes or seeks only the selected timestamps for the indexing pass.

#### Step 3: Detect and track

Run YOLO on sampled frames and associate detections into local camera tracks.

Store:

- detection timestamp;
- bounding box;
- detector score;
- class;
- local track ID;
- crop reference;
- quality score.

A track is the unit used for later ReID aggregation. The system should not treat every sampled detection as a separate identity candidate.

#### Step 4: Select representative crops

Keep only quality-approved crops for ReID:

- sufficient pixel size;
- limited blur;
- limited occlusion;
- reasonable aspect ratio;
- not heavily truncated;
- detector confidence above the indexing policy.

Select a small gallery per track, for example the best three to ten crops. Store additional samples only when evidence review needs them.

#### Step 5: Extract embeddings

For each confirmed track:

- run person ReID only on person crops;
- run vehicle ReID only on vehicle crops;
- batch crops by class and model;
- refresh every 3–5 sampled track observations or when the crop changes materially;
- L2-normalize features before cosine search;
- persist model key, version, feature dimension, metric, and quality score.

Do not create a ReID vector for every raw video frame.

#### Step 6: Write and index in batches

Insert detections, tracks, observations, and embeddings in batches. The video processing worker should not hold a database transaction open while it decodes or runs inference.

For a small MVP gallery:

- write to PostgreSQL;
- use exact filtered search;
- add HNSW only after the gallery grows and exact-search behavior is validated.

#### Step 7: Midnight backfill job

The midnight job should not redo every successful segment. It should:

- find missing or failed segments;
- retry incomplete jobs;
- identify gaps in expected time coverage;
- reprocess segments after a detector/model version change;
- rebuild or compact derived indexes if needed;
- produce a daily indexing report.

Use status and version checks so successful rows are not duplicated.

---

### 3. Record Search indexing granularity

Use multiple levels of data:

| Level | Purpose | Recommended volume |
|---|---|---|
| Frame | Evidence and debugging | Sampled subset only |
| Detection | Bounding boxes and detector output | Every sampled detection |
| Track | Local identity over time | One row per local track |
| Track gallery | ReID appearance evidence | 3–10 quality-approved samples |
| Track prototype | Fast candidate retrieval | One aggregated vector per track |
| Match result | Search output and audit | Only for requested searches |

For fast retrieval, search the track prototype or a small gallery first. Fetch the strongest evidence crops only after ranking.

---

### 4. Live Follow operating model

#### 4.1 Do not spawn one workflow per camera

A naive design does this:

~~~text
target selected
  → start YOLO process for camera 1
  → start YOLO process for camera 2
  → start YOLO process for camera 3
  → ...
~~~

This causes duplicated model memory, process startup cost, queue contention, and difficult cancellation.

Use this instead:

~~~text
one shared detector worker
one shared person ReID worker
one shared vehicle ReID worker
per-camera input queues
micro-batching across eligible camera crops
route-aware camera activation
~~~

A worker can receive crops from multiple cameras in one batch. The camera ID remains metadata on every crop and result.

#### 4.2 Three-layer live design

#### Layer A — always-on sentinel

For every active camera, keep a lightweight process for:

- stream health;
- low-rate frame sampling;
- YOLO detection;
- local tracking;
- recent track metadata.

The sentinel can operate at 1–2 FPS in an initial CPU-oriented design and increase after GPU measurement. It does not need to run ReID on every track.

#### Layer B — active follow burst

When an operator selects a target:

- create a target gallery from the source track;
- determine candidate cameras and an expected arrival window;
- increase sampling for the candidate group;
- run ReID on quality-approved track crops;
- retrieve against the target gallery;
- emit candidate events;
- reduce the burst when the follow session expires or is stopped.

#### Layer C — fallback expansion

If the camera graph is unknown, the candidate group produces no useful result, or a target can plausibly take an unmodeled route:

- keep the sentinel running;
- expand to second-hop cameras;
- use a low-rate global gallery search;
- activate all cameras only for a bounded window when the operational requirement justifies the cost.

---

### 5. Camera groups and geo-map routing

A geo map is useful, but a static group list is not enough. Represent cameras as a directed transition graph.

Each edge should contain:

- source camera;
- destination camera;
- minimum travel time;
- maximum travel time;
- transition prior;
- active/inactive flag;
- optional direction or corridor label.

Example:

~~~text
CAM-01 → CAM-02
  expected window: 10–45 seconds
  prior: 0.80

CAM-02 → CAM-05
  expected window: 20–90 seconds
  prior: 0.65
~~~

Build the first graph manually from the site map. Later, update the priors from operator-confirmed transitions.

#### Route activation

When a target is selected in CAM-01:

1. Activate the first-hop neighbors of CAM-01.
2. Search those cameras inside their expected time windows.
3. Confirm or reject candidates.
4. Expand to second-hop cameras if required.
5. Keep a low-rate sentinel on the remaining cameras.
6. Stop activation when the target expires, the operator stops follow, or the session reaches its maximum duration.

This is better than scanning every camera at the same high ReID rate, but it must retain a fallback for unexpected routes.

#### Why groups alone can fail

A person can:

- enter an unseen camera;
- move in the opposite direction;
- leave the site and re-enter;
- appear in an overlapping field of view;
- be delayed by an obstacle or crowd;
- be missed by the source tracker.

Therefore, use groups for prioritization, not as an irreversible filter.

---

### 6. Live Follow performance controls

#### 6.1 Sampling

Initial values:

- CPU development: one or two cameras, 1–2 FPS;
- GPU validation: 2–5 FPS across the active group;
- increase only when p95 latency and queue depth remain acceptable.

#### 6.2 Tracking before ReID

Use the local tracker to avoid repeated embedding extraction:

- YOLO updates sampled tracks;
- ReID refreshes every 3–5 sampled observations;
- refresh earlier when the crop changes materially;
- discard blur, severe occlusion, and tiny crops;
- use the best crop for gallery updates.

#### 6.3 Shared model workers

- Load one YOLO model per detector worker.
- Load one person ReID model per person worker.
- Load one vehicle ReID model per vehicle worker.
- Batch crops across cameras.
- Keep model initialization outside the per-query path.

#### 6.4 Queues and backpressure

Use a bounded queue per camera and a bounded shared inference queue.

When overloaded:

1. drop stale frames;
2. preserve the newest frame for each active track;
3. preserve high-quality crops;
4. preserve active-follow camera work over background indexing;
5. expose queue depth and dropped-frame metrics.

Never allow an unbounded queue to hide latency growth.

#### 6.5 Search scope

Limit every live vector query by:

- object class;
- active model key/version;
- candidate camera IDs;
- expected time window;
- recent observation window;
- quality floor.

Retrieve more than the final UI Top-K and apply temporal/camera reranking in the Python service.

#### 6.6 Follow session limits

Make these configurable:

- maximum follow duration;
- active camera group;
- burst sampling rate;
- ReID refresh interval;
- maximum candidates per camera;
- fallback expansion delay;
- gallery expiry time.

A follow session should expire cleanly and release its burst resources.

---

### 7. CPU-only development behavior

CPU-only development should prove correctness, not claim final multi-camera real-time capacity.

Recommended local sequence:

1. One recorded segment and one camera.
2. 2 FPS detection.
3. Track and crop quality checks.
4. Exact vector search.
5. One live-follow session.
6. Two or three cameras with route activation.
7. Measure queue depth, latency, and dropped frames.
8. Add GPU acceleration only after the data flow is correct.

For a CPU-only demo with several cameras, process a short recorded clip per camera or use low-rate sentinel sampling. A full multi-camera 5 FPS live deployment should be validated on the target GPU profile.

---

### 8. Recommended MVP configuration

#### Record Search

~~~yaml
record_indexing:
  trigger: segment_finalized
  segment_target_minutes: 1-5
  cpu_sample_fps: 2
  gpu_sample_fps: 5
  reid_refresh_observations: 3-5
  process_unit: confirmed_track
  midnight_job: retry_backfill_and_gap_check
  search_mode: exact_then_hnsw
~~~

#### Live Follow

~~~yaml
live_follow:
  sentinel:
    enabled: true
    cpu_sample_fps: 1-2
    gpu_sample_fps: 2-5
    reid_enabled: false
  burst:
    route_aware: true
    reid_enabled: true
    refresh_observations: 3-5
    max_active_cameras: configurable
  fallback:
    expand_to_second_hop: true
    global_low_rate_scan: true
  queues:
    per_camera_max_frames: 32
    drop_stale_frames: true
~~~

Exact values must be measured with target cameras, crop sizes, and hardware.

---

### 9. Final recommendation

For Record Search, process each **completed MediaMTX segment asynchronously**. This provides nearline availability without waiting until midnight and avoids re-running a full-day batch. Use midnight as a retry, gap-check, and reprocessing window.

For Live Follow, do not launch a full high-rate YOLO + ReID workflow on every camera for every query. Keep low-rate YOLO and tracking available across cameras, activate ReID in a route-aware camera group, share model workers across cameras, and expand to broader coverage only when the route is uncertain or the target is not found.

The best first implementation is therefore:

~~~text
Record Search:
  segment-finalized event → async index job → track-level embeddings → vector index

Live Follow:
  all-camera low-rate sentinel
  + route-aware ReID burst
  + bounded fallback expansion
  + shared batched model workers
~~~

This design gives the MVP fast search availability, manageable CPU/GPU load, and a path to grow from a few cameras to a larger VMS without rewriting the workflow.


### Clarification: scheduled downstream-camera activation

The directed graph can be used as a time-window scheduler. If a target is indexed in CAM-01 at time T0 and the configured edge is CAM-01 → CAM-02 with an expected travel window of 10–45 seconds, the system schedules a higher-rate ReID burst for CAM-02 around:

```text
T0 + 10 seconds − safety margin
through
T0 + 45 seconds + safety margin
```

The safety margin absorbs tracker delay, walking-speed variation, clock drift, and missed detections. CAM-02 does not need a full high-rate workflow before this window.

Recommended behavior:

1. Keep a low-rate sentinel or stream health process warm on every camera.
2. When CAM-01 produces the target event, schedule the CAM-02 ReID burst.
3. Use the source target gallery to search only CAM-02 during the expected window.
4. If CAM-02 confirms the target, schedule the next graph edge from CAM-02.
5. If there is no result, extend the window or branch to other outgoing edges.
6. Keep a bounded global low-rate fallback so an unexpected route is not permanently missed.

Example:

```text
CAM-01 target at 10:00:00
  → CAM-02 burst: 10:00:05–10:00:55
  → if confirmed at 10:00:32, schedule CAM-03 from 10:00:42–10:01:42
  → if no confirmation, try the next eligible edge or bounded fallback
```

The graph is therefore a routing and scheduling layer, not an identity decision by itself. Camera transition priors should narrow the search window and camera set; ReID similarity, time continuity, and operator review still determine the match state.



