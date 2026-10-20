-- 003: ReID models and their embeddings (the "barcodes"), for people (VIP-9)
-- and vehicles (VIP-8). Needs 002.
--
-- DRAFT: agree with VIP-8 before any code writes to these tables.
-- Taken from docs/DATA_AND_PERFORMANCE.md §7.2, §7.4 and §7.5, with three changes:
--
--   * reid_embeddings_2048.embedding is halfvec(2048), not vector(2048).
--     pgvector's HNSW index takes at most 2,000 dimensions of `vector`; the
--     template's two indexes fail with "column cannot have more than 2000
--     dimensions for hnsw index". halfvec stores 16-bit floats: half the disk,
--     indexable up to 4,000 dimensions. Measured on unit vectors, it moves a
--     cosine similarity by at most 0.00004.
--   * reid_embeddings_512 gets the same model/camera/time index as the 2048
--     table, for camera- and time-filtered searches.
--   * Safe to run twice, like 002.
--
-- One table per vector size and one HNSW index per model, so vectors from two
-- models -- or a person and a car -- are never compared. A search must filter
-- on the same model_key and class_name as the index's WHERE, or the index is
-- not used.
--
-- Threshold policies and the search/follow tables (§7.2, §7.6) come in a later
-- file, with the Record Search milestone (M4).
--
-- Apply to an existing database:  .venv/bin/python scripts/migrate.py

CREATE TABLE IF NOT EXISTS reid_models (
    model_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    model_key TEXT NOT NULL,                    -- "msmt17_sbs_r50_ibn", "osnet_ain_x1_0_msmt17"
    backend TEXT NOT NULL,                      -- "fastreid", "torchreid"
    object_class object_class NOT NULL,
    model_version TEXT NOT NULL,                -- the weights file's SHA-256
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

-- FastReID: people (MSMT17) and vehicles (VeRi), both 2048 numbers.
CREATE TABLE IF NOT EXISTS reid_embeddings_2048 (
    embedding_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    track_id BIGINT NOT NULL REFERENCES tracks(track_id) ON DELETE CASCADE,
    detection_id BIGINT REFERENCES detections(detection_id),  -- the crop it came from
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
    embedding halfvec(2048) NOT NULL,
    sample_key TEXT NOT NULL UNIQUE,            -- track + model key + version + crop time (ms)
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS reid_embeddings_2048_metadata_idx
    ON reid_embeddings_2048 (model_key, class_name, camera_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS reid_embeddings_2048_track_time_idx
    ON reid_embeddings_2048 (track_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS reid_person_msmt17_hnsw_idx
    ON reid_embeddings_2048
    USING hnsw (embedding halfvec_cosine_ops)
    WHERE model_key = 'msmt17_sbs_r50_ibn' AND class_name = 'person';

CREATE INDEX IF NOT EXISTS reid_vehicle_veri_hnsw_idx
    ON reid_embeddings_2048
    USING hnsw (embedding halfvec_cosine_ops)
    WHERE model_key = 'veri_sbs_r50_ibn' AND class_name = 'vehicle';

-- OSNet-AIN (VIP-9's comparison model): people, 512 numbers.
CREATE TABLE IF NOT EXISTS reid_embeddings_512 (
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

CREATE INDEX IF NOT EXISTS reid_embeddings_512_metadata_idx
    ON reid_embeddings_512 (model_key, class_name, camera_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS reid_osnet_person_hnsw_idx
    ON reid_embeddings_512
    USING hnsw (embedding vector_cosine_ops)
    WHERE model_key = 'osnet_ain_x1_0_msmt17' AND class_name = 'person';
