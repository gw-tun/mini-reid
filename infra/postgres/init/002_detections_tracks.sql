-- 002: cameras, recordings, detections and tracks -- shared by people (VIP-9)
-- and vehicles (VIP-8); `class_name` keeps them apart.
--
-- DRAFT: agree with VIP-8 before any code writes to these tables.
-- Taken from docs/DATA_AND_PERFORMANCE.md §7.1-7.3, changed only so that it is
-- safe to run twice (IF NOT EXISTS; a type is created only when missing), which
-- lets the same file be replayed on a database that already exists.
--
-- Docker runs this file by itself on an EMPTY database. On one that already
-- holds data:  .venv/bin/python scripts/migrate.py

DO $$ BEGIN
    CREATE TYPE object_class AS ENUM ('person', 'vehicle');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE source_kind AS ENUM ('live', 'recording');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

DO $$ BEGIN
    CREATE TYPE recording_status AS ENUM ('registered', 'ready', 'processing', 'complete', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL;
END $$;

CREATE TABLE IF NOT EXISTS cameras (
    camera_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_key TEXT NOT NULL UNIQUE,            -- Frame.camera_key: "cam01", "G336"
    display_name TEXT NOT NULL,
    mediamtx_path TEXT NOT NULL UNIQUE,
    location_label TEXT,
    timezone_name TEXT NOT NULL DEFAULT 'UTC',  -- how recording file names are read (RECORDING_TZ)
    nominal_width INTEGER,
    nominal_height INTEGER,
    nominal_fps REAL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS recordings (
    recording_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    source_kind source_kind NOT NULL DEFAULT 'recording',
    media_uri TEXT NOT NULL,                    -- the segment file MediaMTX wrote
    started_at TIMESTAMPTZ NOT NULL,            -- from the file name (pyav_reader.describe_segment)
    ended_at TIMESTAMPTZ,
    duration_ms BIGINT,
    media_format TEXT,
    status recording_status NOT NULL DEFAULT 'registered',
    file_size_bytes BIGINT,
    checksum_sha256 TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (camera_id, media_uri)
);

CREATE INDEX IF NOT EXISTS recordings_camera_time_idx ON recordings (camera_id, started_at DESC);
CREATE INDEX IF NOT EXISTS recordings_time_idx ON recordings (started_at, ended_at);

-- Every box the detector returned, kept or rejected: the evidence for
-- "why was this person never found?".
CREATE TABLE IF NOT EXISTS detections (
    detection_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    recording_id BIGINT REFERENCES recordings(recording_id),  -- NULL for a live camera
    source_kind source_kind NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,           -- Frame.captured_at
    frame_number BIGINT,
    class_name object_class NOT NULL,
    detector_key TEXT NOT NULL,                 -- YoloDetector.model_key: "yolo11m@640"
    detector_version TEXT NOT NULL,             -- the weights file's SHA-256
    detector_score REAL NOT NULL CHECK (detector_score >= 0 AND detector_score <= 1),
    bbox_x1 REAL NOT NULL,                      -- frame pixels, like Detection.bbox
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    crop_uri TEXT,                              -- NULL when the crop gate rejected the box
    crop_width INTEGER,
    crop_height INTEGER,
    quality_score REAL,                         -- crop gate score, 0..1
    payload JSONB,                              -- e.g. {"label": "bus"}, {"rejected": "too_small"}
    ingest_key TEXT NOT NULL UNIQUE,            -- recording + frame time + box number: a retried segment can't insert twice
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (bbox_x2 > bbox_x1),
    CHECK (bbox_y2 > bbox_y1)
);

CREATE INDEX IF NOT EXISTS detections_camera_time_idx ON detections (camera_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS detections_class_time_idx ON detections (class_name, observed_at DESC);
CREATE INDEX IF NOT EXISTS detections_recording_time_idx ON detections (recording_id, observed_at);

-- One person (or vehicle) followed inside ONE camera by ByteTrackTracker.
-- Who is the same person across cameras is not stored here: that is ReID.
CREATE TABLE IF NOT EXISTS tracks (
    track_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    camera_id BIGINT NOT NULL REFERENCES cameras(camera_id),
    recording_id BIGINT REFERENCES recordings(recording_id),
    source_kind source_kind NOT NULL,
    class_name object_class NOT NULL,
    local_track_key TEXT NOT NULL,              -- the tracker's id: "cam01-17"
    started_at TIMESTAMPTZ NOT NULL,            -- EndedTrack.first_seen
    ended_at TIMESTAMPTZ,                       -- NULL while the track is still live
    status TEXT NOT NULL DEFAULT 'active',
    best_crop_uri TEXT,                         -- TrackCropSelector.end()[0]
    best_quality_score REAL,
    quality_summary JSONB,                      -- e.g. the other best crops, rejection counts
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- Tracker ids restart at 1 on every run, so the start time is part of the key.
    UNIQUE (camera_id, source_kind, local_track_key, started_at)
);

CREATE INDEX IF NOT EXISTS tracks_camera_time_idx ON tracks (camera_id, started_at DESC);
CREATE INDEX IF NOT EXISTS tracks_class_time_idx ON tracks (class_name, started_at DESC);

-- The track's confirmed boxes, frame by frame (TrackObservation).
CREATE TABLE IF NOT EXISTS track_observations (
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

CREATE INDEX IF NOT EXISTS track_observations_track_time_idx
    ON track_observations (track_id, observed_at);
