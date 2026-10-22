BEGIN;

DO $$ BEGIN CREATE TYPE object_class AS ENUM ('person', 'vehicle'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE source_kind AS ENUM ('live', 'recording'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE recording_status AS ENUM ('registered', 'ready', 'processing', 'complete', 'failed'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE query_kind AS ENUM ('record_search', 'live_follow'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE TYPE match_state AS ENUM ('candidate', 'review', 'confirmed', 'rejected'); EXCEPTION WHEN duplicate_object THEN NULL; END $$;

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
CREATE INDEX track_observations_track_time_idx ON track_observations (track_id, observed_at);
CREATE INDEX detections_observed_at_brin_idx ON detections USING brin (observed_at) WITH (pages_per_range = 128);
CREATE INDEX track_observations_time_brin_idx ON track_observations USING brin (observed_at);

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
CREATE INDEX reid_embeddings_2048_metadata_idx ON reid_embeddings_2048 (model_key, class_name, camera_id, observed_at DESC);
CREATE INDEX reid_embeddings_2048_track_time_idx ON reid_embeddings_2048 (track_id, observed_at DESC);
-- pgvector HNSW indexes reject vector dimensions above 2000. The initial
-- 2048-D gallery therefore uses exact cosine search plus B-tree metadata
-- indexes. Add a halfvec expression index after measuring the installed
-- pgvector version and query workload.

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

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMIT;
