# CCTV VMS — Database and Performance Specification
## PostgreSQL / Neon Design Companion

**Scope:** recordings, detections, tracks, ReID embeddings, vector indexes, search results, and live-follow events.  
**Primary mode:** local CPU-only development.  
**Initial models:** FastReID MSMT17 SBS R50-IBN for person and VeRi SBS R50-IBN for vehicle.  
**Future model:** OSNet-AIN through a versioned, model-specific index.

---

## 1. Database responsibilities

The database stores searchable metadata and durable evidence. It should not become the video-processing loop.

It must support:

- camera and recording time-range searches;
- detections, tracks, crops, and evidence references;
- class-specific and model-specific vector search;
- reproducible match results;
- live-follow event history;
- idempotent ingestion and retry;
- retention by time range;
- CPU-friendly development;
- future model swaps without mixing incompatible embeddings.

Separate the data into operational metadata, event data, and large assets. Store recordings and crops as file or object-storage URIs. Store vectors in pgvector only when the gallery size and query workload justify it.

## 2. Workload model

The live write path is:

~~~text
RTSP frame → sampled frame → YOLO detection → local track
→ quality gate → optional ReID embedding → batched database write
~~~

Record search is:

~~~text
source track → time/camera/class/model filters → vector Top-K
→ temporal/quality reranking → evidence results
~~~

Live follow is:

~~~text
active track → recent candidates → vector Top-K
→ camera transition/time filtering → event stream
~~~

Estimate scale with:

~~~text
sampled_frames_per_day
  = cameras × sample_fps × active_seconds_per_day

embedding_rows_per_day
  = confirmed_tracks × embedding_refreshes_per_track

raw_vector_bytes
  = embedding_rows × feature_dim × bytes_per_value
~~~

Measure target-camera data before making capacity claims.

## 3. Search and indexing theory

### B-tree

Use B-tree indexes for equality, ordering, and range predicates such as camera, recording, class, timestamps, statuses, and model versions. Design indexes around real query shapes and constrain leading columns of multi-column indexes. See [PostgreSQL multicolumn indexes](https://www.postgresql.org/docs/current/indexes-multicolumn.html).

### BRIN

BRIN (Block Range Index) stores summaries for physical page ranges. It is useful for very large append-oriented tables where timestamps correlate with insertion order. It is small and cheap but lossy, so PostgreSQL rechecks heap rows. See [PostgreSQL BRIN documentation](https://www.postgresql.org/docs/17/brin.html).

### Exact nearest-neighbor search

Without an approximate vector index, pgvector scans eligible vectors and orders them by distance. Use this for early CPU-only galleries, narrow camera/time windows, threshold calibration, ground-truth recall, and validating HNSW or IVFFlat.

### HNSW

HNSW (Hierarchical Navigable Small World) builds a layered proximity graph. It usually provides a strong speed/recall trade-off, but uses more memory and build time than IVFFlat. It does not need a training phase and is the likely first approximate index for a growing gallery.

### IVFFlat

IVFFlat groups vectors into centroid lists and probes only a subset. It usually builds faster and uses less memory, but needs representative data before index creation and requires list/probe tuning.

The pgvector guidance suggests starting around rows / 1000 lists up to one million rows, sqrt(rows) lists above that, and sqrt(lists) probes as an initial experiment. These are starting points only. See [pgvector indexing guidance](https://github.com/pgvector/pgvector/blob/master/README.md).

| Workload | Initial choice |
|---|---|
| CPU development, small filtered gallery | Exact search |
| Growing interactive gallery | HNSW |
| Large mostly-static historical corpus | HNSW or IVFFlat after benchmark |
| Threshold and recall evaluation | Exact search as ground truth |
| Strong camera/time filtering | B-tree/BRIN/partition pruning before vector search |

Approximate indexes can lose recall when metadata filters are applied after the graph/list scan. Use filter-column indexes, partial indexes for a few fixed values, partitioning for many values, and iterative scans. See [pgvector filtering and iterative scans](https://github.com/pgvector/pgvector/blob/master/README.md).

## 4. Application-level matching algorithm

Vector similarity is only one signal. Let s_reid be cosine similarity, s_det detector confidence, s_quality crop quality, s_time temporal continuity, s_camera the camera-transition prior, and s_margin the gap to the second-best candidate.

~~~text
s_final =
    w_reid    × s_reid
  + w_det     × s_det
  + w_quality × s_quality
  + w_time    × s_time
  + w_camera  × s_camera
  + w_margin  × s_margin
~~~

Normalize every component first and record the weights with the match result.

A simple temporal score is:

~~~text
s_time = exp(-abs(t_candidate - expected_time) / tau)
~~~

Maintain a small quality-filtered gallery per track. A robust first aggregation is the mean of the best three similarities. Calibrate thresholds per model version, object class, camera pair/group, crop-quality bucket, and query mode. Use Top-K for candidate generation, then threshold + score margin + temporal confirmation for confirmed or auto-follow states.

## 5. Data lifecycle

1. Register each MediaMTX recording segment.
2. Process selected live or recorded time ranges.
3. Insert detections and tracks in batches.
4. Queue stable, quality-approved crops for ReID.
5. Insert embeddings with model/version metadata.
6. Retrieve and rerank candidates.
7. Store match results and operator decisions.

Use deterministic keys:

~~~text
detection ingest key:
(camera_id, recording_id, frame_timestamp_ms, detector_version)

embedding sample key:
(track_id, model_key, model_version, sample_timestamp_ms)
~~~

Use unique constraints so retries do not duplicate rows. Keep hot data (current tracks, recent embeddings, live follow) separate from cold data (historical indexing, re-embedding, calibration, reports).

## 6. Logical schema

~~~mermaid
erDiagram
    cameras ||--o{ recordings : produces
    cameras ||--o{ detections : observes
    recordings ||--o{ detections : contains
    cameras ||--o{ tracks : owns
    tracks ||--o{ track_observations : has
    detections ||--o| track_observations : references
    tracks ||--o{ reid_embeddings_2048 : describes
    reid_models ||--o{ reid_embeddings_2048 : generated_by
    search_queries ||--o{ match_results : returns
    follow_sessions ||--o{ follow_events : emits
    cameras ||--o{ camera_transitions : source
~~~

The initial FastReID models use a fixed vector(2048) table. OSNet-AIN uses a different feature dimension in the official Torchreid setup, so create a separate vector(512) table/index later. Do not mix model spaces.

## 7. PostgreSQL / Neon schema

The following is a migration template.

### 7.1 Extensions and types

~~~sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TYPE object_class AS ENUM ('person', 'vehicle');
CREATE TYPE source_kind AS ENUM ('live', 'recording');
CREATE TYPE recording_status AS ENUM ('registered', 'ready', 'processing', 'complete', 'failed');
CREATE TYPE query_kind AS ENUM ('record_search', 'live_follow');
CREATE TYPE match_state AS ENUM ('candidate', 'review', 'confirmed', 'rejected');
~~~

### 7.2 Cameras, recordings, and models

~~~sql
CREATE TABLE cameras (
    camera_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_key TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    mediamtx_path TEXT NOT NULL UNIQUE,
    location_label TEXT,
    timezone_name TEXT NOT NULL DEFAULT 'UTC',
    nominal_width INTEGER,
    nominal_height INTEGER,
    nominal_fps REAL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE recordings (
    recording_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    source_kind source_kind NOT NULL DEFAULT 'recording',
    media_uri TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    duration_ms BIGINT,
    media_format TEXT,
    status recording_status NOT NULL DEFAULT 'registered',
    file_size_bytes BIGINT,
    checksum_sha256 TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (camera_id, media_uri)
);

CREATE INDEX recordings_camera_time_idx ON recordings (camera_id, started_at DESC);
CREATE INDEX recordings_time_idx ON recordings (started_at, ended_at);

CREATE TABLE reid_models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_key TEXT NOT NULL,
    backend TEXT NOT NULL,
    object_class object_class NOT NULL,
    model_version TEXT NOT NULL,
    weights_uri TEXT NOT NULL,
    config_uri TEXT,
    feature_dim INTEGER NOT NULL CHECK (feature_dim > 0),
    input_height INTEGER NOT NULL CHECK (input_height > 0),
    input_width INTEGER NOT NULL CHECK (input_width > 0),
    distance_metric TEXT NOT NULL CHECK (distance_metric IN ('cosine','l2','inner_product')),
    l2_normalized BOOLEAN NOT NULL DEFAULT TRUE,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_key, model_version)
);

CREATE TABLE reid_threshold_policies (
    threshold_policy_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_id BIGINT NOT NULL REFERENCES reid_models(model_id),
    camera_from_id BIGINT REFERENCES cameras(camera_id),
    camera_to_id BIGINT REFERENCES cameras(camera_id),
    quality_bucket TEXT,
    candidate_top_k INTEGER NOT NULL DEFAULT 20 CHECK (candidate_top_k > 0),
    review_threshold REAL,
    confirm_threshold REAL,
    min_score_margin REAL,
    min_confirm_events INTEGER NOT NULL DEFAULT 2 CHECK (min_confirm_events > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
~~~

### 7.3 Detections, tracks, and observations

~~~sql
CREATE TABLE detections (
    detection_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    recording_id BIGINT REFERENCES recordings(recording_id),
    source_kind source_kind NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    frame_number BIGINT,
    class_name object_class NOT NULL,
    detector_key TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    detector_score REAL NOT NULL CHECK (detector_score >= 0 AND detector_score <= 1),
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    crop_uri TEXT,
    crop_width INTEGER,
    crop_height INTEGER,
    quality_score REAL,
    payload JSONB,
    ingest_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (bbox_x2 > bbox_x1),
    CHECK (bbox_y2 > bbox_y1)
);

CREATE INDEX detections_camera_time_idx ON detections (camera_id, observed_at DESC);
CREATE INDEX detections_class_time_idx ON detections (class_name, observed_at DESC);
CREATE INDEX detections_recording_time_idx ON detections (recording_id, observed_at);

CREATE TABLE tracks (
    track_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    recording_id BIGINT REFERENCES recordings(recording_id),
    source_kind source_kind NOT NULL,
    class_name object_class NOT NULL,
    local_track_key TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    ended_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'active',
    best_crop_uri TEXT,
    best_quality_score REAL,
    quality_summary JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (camera_id, source_kind, local_track_key, started_at)
);

CREATE INDEX tracks_camera_time_idx ON tracks (camera_id, started_at DESC);
CREATE INDEX tracks_class_time_idx ON tracks (class_name, started_at DESC);

CREATE TABLE track_observations (
    observation_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    track_id BIGINT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    detection_id BIGINT REFERENCES detections(detection_id),
    observed_at TIMESTAMPTZ NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    detector_score REAL NOT NULL,
    quality_score REAL,
    crop_uri TEXT,
    UNIQUE (track_id, observed_at)
);

CREATE INDEX track_observations_track_time_idx
    ON track_observations (track_id, observed_at);
~~~

For very large append-only tables, add BRIN after measuring correlation:

~~~sql
CREATE INDEX detections_observed_at_brin_idx
    ON detections USING brin (observed_at) WITH (pages_per_range = 128);

CREATE INDEX track_observations_time_brin_idx
    ON track_observations USING brin (observed_at);
~~~

### 7.4 FastReID 2048-D embeddings

~~~sql
CREATE TABLE reid_embeddings_2048 (
    embedding_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    track_id BIGINT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    detection_id BIGINT REFERENCES detections(detection_id),
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    class_name object_class NOT NULL,
    model_id BIGINT NOT NULL REFERENCES reid_models(model_id),
    model_key TEXT NOT NULL,
    model_version TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    feature_dim INTEGER NOT NULL DEFAULT 2048 CHECK (feature_dim = 2048),
    distance_metric TEXT NOT NULL DEFAULT 'cosine',
    l2_normalized BOOLEAN NOT NULL DEFAULT TRUE,
    quality_score REAL NOT NULL CHECK (quality_score >= 0 AND quality_score <= 1),
    embedding vector(2048) NOT NULL,
    sample_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX reid_embeddings_2048_metadata_idx
    ON reid_embeddings_2048 (model_key, class_name, camera_id, observed_at DESC);

CREATE INDEX reid_embeddings_2048_track_time_idx
    ON reid_embeddings_2048 (track_id, observed_at DESC);

CREATE INDEX reid_person_msmt17_hnsw_idx
    ON reid_embeddings_2048
    USING hnsw (embedding vector_cosine_ops)
    WHERE model_key = 'msmt17_sbs_r50_ibn' AND class_name = 'person';

CREATE INDEX reid_vehicle_veri_hnsw_idx
    ON reid_embeddings_2048
    USING hnsw (embedding vector_cosine_ops)
    WHERE model_key = 'veri_sbs_r50_ibn' AND class_name = 'vehicle';
~~~

### 7.5 Future OSNet-AIN 512-D embeddings

~~~sql
CREATE TABLE reid_embeddings_512 (
    embedding_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    track_id BIGINT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    detection_id BIGINT REFERENCES detections(detection_id),
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    class_name object_class NOT NULL,
    model_id BIGINT NOT NULL REFERENCES reid_models(model_id),
    model_key TEXT NOT NULL,
    model_version TEXT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    feature_dim INTEGER NOT NULL DEFAULT 512 CHECK (feature_dim = 512),
    distance_metric TEXT NOT NULL DEFAULT 'cosine',
    l2_normalized BOOLEAN NOT NULL DEFAULT TRUE,
    quality_score REAL NOT NULL CHECK (quality_score >= 0 AND quality_score <= 1),
    embedding vector(512) NOT NULL,
    sample_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX reid_osnet_person_hnsw_idx
    ON reid_embeddings_512
    USING hnsw (embedding vector_cosine_ops)
    WHERE model_key = 'osnet_ain_x1_0_msmt17' AND class_name = 'person';
~~~

### 7.6 Search, match, and follow tables

~~~sql
CREATE TABLE search_queries (
    query_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_kind query_kind NOT NULL,
    source_track_id BIGINT REFERENCES tracks(track_id),
    source_detection_id BIGINT REFERENCES detections(detection_id),
    class_name object_class NOT NULL,
    model_id BIGINT NOT NULL REFERENCES reid_models(model_id),
    requested_from TIMESTAMPTZ,
    requested_to TIMESTAMPTZ,
    requested_cameras BIGINT[],
    top_k INTEGER NOT NULL DEFAULT 20,
    status TEXT NOT NULL DEFAULT 'queued',
    config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE match_results (
    match_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    query_id UUID NOT NULL REFERENCES search_queries(query_id) ON DELETE CASCADE,
    source_track_id BIGINT REFERENCES tracks(track_id),
    candidate_track_id BIGINT REFERENCES tracks(track_id),
    candidate_embedding_id BIGINT,
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    observed_at TIMESTAMPTZ NOT NULL,
    raw_similarity REAL NOT NULL,
    rerank_score REAL,
    detector_score REAL,
    quality_score REAL,
    temporal_score REAL,
    camera_prior_score REAL,
    score_margin REAL,
    rank_position INTEGER NOT NULL,
    state match_state NOT NULL DEFAULT 'candidate',
    evidence_uri TEXT,
    explanation JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX match_results_query_rank_idx ON match_results (query_id, rank_position);
CREATE INDEX match_results_camera_time_idx ON match_results (camera_id, observed_at DESC);

CREATE TABLE follow_sessions (
    follow_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_track_id BIGINT NOT NULL REFERENCES tracks(track_id),
    model_id BIGINT NOT NULL REFERENCES reid_models(model_id),
    status TEXT NOT NULL DEFAULT 'active',
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at TIMESTAMPTZ,
    config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE follow_events (
    follow_event_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    follow_id UUID NOT NULL REFERENCES follow_sessions(follow_id) ON DELETE CASCADE,
    candidate_track_id BIGINT REFERENCES tracks(track_id),
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    observed_at TIMESTAMPTZ NOT NULL,
    raw_similarity REAL NOT NULL,
    rerank_score REAL,
    state match_state NOT NULL DEFAULT 'candidate',
    evidence_uri TEXT,
    payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX follow_events_session_time_idx ON follow_events (follow_id, observed_at DESC);

CREATE TABLE camera_transitions (
    camera_transition_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    target_camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    min_seconds INTEGER NOT NULL DEFAULT 0,
    max_seconds INTEGER NOT NULL,
    prior_score REAL NOT NULL DEFAULT 0.5,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    CHECK (max_seconds >= min_seconds),
    CHECK (prior_score >= 0 AND prior_score <= 1),
    UNIQUE (source_camera_id, target_camera_id)
);
~~~

## 8. Query patterns

### Time-filtered evidence

~~~sql
SELECT d.detection_id, d.camera_id, d.observed_at, d.class_name,
       d.detector_score, d.crop_uri, r.media_uri
FROM detections d
LEFT JOIN recordings r ON r.recording_id = d.recording_id
WHERE d.camera_id = :camera_id
  AND d.observed_at >= :from_ts
  AND d.observed_at < :to_ts
  AND d.class_name = :class_name
ORDER BY d.observed_at;
~~~

### Exact cosine search

For L2-normalized vectors, pgvector cosine distance uses the <=> operator.

~~~sql
SELECT e.embedding_id, e.track_id, e.camera_id, e.observed_at,
       1 - (e.embedding <=> :query_vector) AS cosine_similarity,
       e.quality_score
FROM reid_embeddings_2048 e
WHERE e.model_key = :model_key
  AND e.class_name = :class_name
  AND e.camera_id = ANY(:camera_ids)
  AND e.observed_at >= :from_ts
  AND e.observed_at < :to_ts
ORDER BY e.embedding <=> :query_vector
LIMIT :candidate_limit;
~~~

### HNSW search with filters

~~~sql
BEGIN;
SET LOCAL hnsw.ef_search = 80;
SET LOCAL hnsw.iterative_scan = strict_order;

SELECT e.embedding_id, e.track_id, e.camera_id, e.observed_at,
       1 - (e.embedding <=> :query_vector) AS cosine_similarity,
       e.quality_score
FROM reid_embeddings_2048 e
WHERE e.model_key = 'msmt17_sbs_r50_ibn'
  AND e.class_name = 'person'
  AND e.camera_id = ANY(:camera_ids)
  AND e.observed_at >= :from_ts
  AND e.observed_at < :to_ts
ORDER BY e.embedding <=> :query_vector
LIMIT 100;

COMMIT;
~~~

Retrieve more candidates than the UI Top-K so the Python service can rerank.

## 9. Partitioning strategy

Do not partition every table at the start. Partition only after measured row counts, retention requirements, and query plans justify it.

Good candidates are detections by observed_at, track observations by observed_at, match results by created_at, and large embedding tables by model dimension/model key.

~~~sql
CREATE TABLE detections_by_time (
    detection_id BIGINT NOT NULL,
    camera_id BIGINT NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    class_name object_class NOT NULL,
    detector_score REAL NOT NULL,
    bbox_x1 REAL NOT NULL,
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    crop_uri TEXT,
    PRIMARY KEY (detection_id, observed_at)
) PARTITION BY RANGE (observed_at);

CREATE TABLE detections_2026_10
    PARTITION OF detections_by_time
    FOR VALUES FROM ('2026-10-01T00:00:00Z')
                 TO   ('2026-11-01T00:00:00Z');

CREATE INDEX detections_2026_10_camera_time_idx
    ON detections_2026_10 (camera_id, observed_at DESC);
~~~

Too many partitions increase planning and memory overhead. The partition key should match common WHERE predicates, and time-aligned partitions make retention easier. See [PostgreSQL partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html).

## 10. Ingestion and maintenance performance

- Insert 100–1,000 detections per transaction and 16–128 embeddings per transaction as initial ranges.
- Use COPY or multi-row inserts when ORM overhead appears.
- Keep inference out of long database transactions.
- Use separate pool limits for ingestion and search if possible.
- Apply statement timeouts to interactive searches.
- Run ANALYZE after bulk ingestion or major distribution changes.
- Use VACUUM (ANALYZE) according to the deployed PostgreSQL/Neon policy.
- Use EXPLAIN (ANALYZE, BUFFERS) for every important query shape.
- Build HNSW after a representative batch exists.
- Build IVFFlat only after representative data exists.
- Compare p50/p95 latency and recall against exact search.
- Do not rebuild a large vector index during the live-follow peak window.

## 11. Performance targets and benchmark plan

| Metric | CPU development | GPU validation |
|---|---:|---:|
| Camera count | 1 | measured incrementally |
| Sampling | 2 FPS | 5 FPS baseline |
| Interactive search p95 | < 3 s for a small filtered gallery | < 1 s target after indexing |
| Live-follow update | < 5 s from sampled event | < 2 s target |
| Vector recall | exact baseline | HNSW/IVFFlat versus exact |

Benchmark 1, 3, 6, and target camera counts at 2 FPS and 5 FPS. Record rows and embeddings inserted per second, p50/p95 query latency, index build time and size, exact-vs-approximate recall@K, filtered-query recall, database CPU/memory, queue depth, and dropped frames.

Choose an approximate index only when it meets the latency target while preserving agreed recall on local CCTV validation data.

## 12. Security and retention

CCTV crops and ReID embeddings are sensitive operational data. Require authenticated database connections, protected media URI access, role-based search/export permissions, audit logs, encryption in transit, retention policies, no public MediaMTX paths, and model/version traceability.

A vector match is a similarity ranking, not identity proof. Store candidate, review, confirmed, or rejected states and keep operator actions.

## 13. Recommended implementation order

### Stage A — correctness

- tables without approximate vector indexes;
- in-memory gallery plus exact SQL search;
- B-tree metadata indexes;
- CPU-only ingestion and record search;
- query-plan and latency logging.

### Stage B — durable vector search

- enable pgvector;
- add fixed-dimensional FastReID tables;
- add HNSW partial indexes per model/class;
- compare HNSW recall to exact search;
- enable iterative scans for filtered queries.

### Stage C — scale and retention

- batch ingestion and COPY;
- time partition large event tables;
- BRIN indexes for append-correlated timestamps;
- retention jobs that detach/drop old partitions;
- separate hot and historical workers.

### Future — model swap and optimization

- add OSNet-AIN vector(512) storage/index;
- re-embed historical galleries per model version;
- compare full and half precision;
- test ONNX/TensorRT or CPU ONNX Runtime;
- learn camera transition priors from confirmed events.

## 14. Final recommendations

1. Use PostgreSQL/Neon for metadata, event state, and the first vector implementation.
2. Keep recordings and crops outside the database; store URIs and metadata.
3. Start with exact vector search for small CPU-only galleries.
4. Use HNSW as the likely first approximate index once the gallery grows.
5. Keep model, class, dimension, metric, and version explicit.
6. Use vector(2048) for the initial FastReID R50-IBN models.
7. Use a separate vector(512) table/index for OSNet-AIN.
8. Add B-tree indexes for camera/class/time and BRIN for very large append-only timestamp tables.
9. Add time partitioning only after measured growth justifies it.
10. Treat approximate recall under metadata filters as a measured property.
11. Use Top-K plus temporal, camera, and quality reranking.
12. Calibrate thresholds per model, class, camera context, and quality bucket.
13. Batch writes, use idempotency keys, and keep inference out of long transactions.
14. Validate every approximate index against exact local CCTV search.

## References

- [pgvector official README](https://github.com/pgvector/pgvector/blob/master/README.md)
- [PostgreSQL table partitioning](https://www.postgresql.org/docs/current/ddl-partitioning.html)
- [PostgreSQL multicolumn indexes](https://www.postgresql.org/docs/current/indexes-multicolumn.html)
- [PostgreSQL BRIN indexes](https://www.postgresql.org/docs/17/brin.html)
- [FastReID Base-SBS configuration](https://github.com/JDAI-CV/fast-reid/blob/master/configs/Base-SBS.yml)
- [FastReID model zoo](https://github.com/JDAI-CV/fast-reid/blob/master/MODEL_ZOO.md)
- [OSNet-AIN paper](https://arxiv.org/abs/1910.06827)
- [Torchreid model zoo](https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO)


## 16. Recording indexing jobs and route-aware Live Follow

### 16.1 Segment-finalized indexing

Record Search should index each completed MediaMTX segment asynchronously. The database needs a durable job state so retries and midnight maintenance do not duplicate successful work.

```sql
CREATE TABLE recording_index_jobs (
    job_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    recording_id BIGINT NOT NULL REFERENCES recordings(recording_id) ON DELETE CASCADE,
    requested_from TIMESTAMPTZ,
    requested_to TIMESTAMPTZ,
    detector_key TEXT NOT NULL,
    detector_version TEXT NOT NULL,
    person_model_id BIGINT REFERENCES reid_models(model_id),
    vehicle_model_id BIGINT REFERENCES reid_models(model_id),
    sample_fps REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    attempt_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    source_key TEXT NOT NULL UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX recording_index_jobs_status_idx
    ON recording_index_jobs (status, created_at);

CREATE INDEX recording_index_jobs_recording_idx
    ON recording_index_jobs (recording_id, created_at DESC);
```

The midnight job should query `queued`, `failed`, or gap-detected jobs. It should not re-run jobs whose detector/model versions and source range already completed successfully.

### 16.2 Transition windows

The existing `camera_transitions` table defines the graph. Live Follow uses it to schedule a bounded ReID window. If a target is observed in source camera S at T0 and the edge S → D has a minimum and maximum travel time, schedule D for:

```text
[T0 + min_seconds - safety_margin,
 T0 + max_seconds + safety_margin]
```

Keep the low-rate camera sentinel warm, but activate the higher-rate ReID burst only during that window.

### 16.3 Branch execution

At a T-junction, multiple outgoing edges can be active at the same time. Use a target-scoped branch record so one branch can be confirmed while other branches are reduced or cancelled.

```sql
CREATE TABLE follow_branches (
    branch_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    follow_id UUID NOT NULL REFERENCES follow_sessions(follow_id) ON DELETE CASCADE,
    source_camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    target_camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    window_start TIMESTAMPTZ NOT NULL,
    window_end TIMESTAMPTZ NOT NULL,
    priority_score REAL NOT NULL DEFAULT 0.5,
    status TEXT NOT NULL DEFAULT 'scheduled',
    activated_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    close_reason TEXT,
    CHECK (window_end > window_start)
);

CREATE INDEX follow_branches_active_idx
    ON follow_branches (follow_id, status, window_start);
```

A branch is a routing and scheduling candidate, not an identity decision. The match service still evaluates vector similarity, crop quality, temporal continuity, and operator state.

### 16.4 Shared-worker runtime

Do not create one model process per camera. Load shared YOLO and ReID workers, maintain per-camera bounded queues, batch eligible crops across cameras, and prioritize active-follow branches over cold recording indexing.

Useful runtime metrics are queue depth, frame age, dropped frames, active branches, detector latency, ReID latency, and branch confirmation time.


