/**
 * The mini-reid Python API (services/api), called from Server Components.
 *
 * The server reaches the API directly; the browser goes through this app's
 * own /api/* proxy (next.config.ts) -- crops, videos, and the "Search similar"
 * calls (components/similar-search.tsx) -- so the API needs no CORS. Types
 * mirror services/api/app/schemas.py.
 */
import type { RecordingGroup } from "@/lib/recordings";

const API = process.env.MINI_REID_API ?? "http://127.0.0.1:8000";

export type PersonTrack = {
  track_id: number;
  camera_key: string;
  local_track_key: string;
  started_at: string;
  ended_at: string | null;
  seconds: number | null;
  observations: number;
  best_crop_uri: string | null;
  best_quality: number | null;
  models: string[];
};

export type PersonCrop = { uri: string; quality: number; observed_at: string };

export type PersonTrackDetail = PersonTrack & {
  recording_uri: string | null;
  crops: PersonCrop[];
};

export type SimilarTrack = {
  track: PersonTrack;
  /** Cosine of the two tracks' mean barcodes. Scales differ between models. */
  similarity: number;
  /** Same camera, overlapping in time: certainly a different person. */
  same_time: boolean;
};

export type SimilarTracksResponse = {
  model_key: string;
  model_version: string;
  query_track_id: number;
  results: SimilarTrack[];
};

export type ReIDModelInfo = {
  model_key: string;
  backend: string;
  feature_dim: number;
  tracks: number;
};

/** Human names for the vector spaces (model keys) the indexer writes. */
export const MODEL_NAMES: Record<string, string> = {
  msmt17_sbs_r50_ibn: "FastReID MSMT17",
  osnet_ain_x1_0_msmt17: "OSNet-AIN",
};

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function get<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API}${path}`);
  } catch {
    throw new ApiError(
      "The mini-reid API is not running. Start it with: .venv/bin/python -m uvicorn services.api.app.main:app --port 8000",
      0,
    );
  }
  if (!res.ok) {
    const body = (await res.json().catch(() => ({}))) as { detail?: string };
    if (res.status === 404 && body.detail === "Not Found") {
      // FastAPI's answer for a route it doesn't have: an API started before the code changed.
      throw new ApiError(
        `The mini-reid API doesn't have ${path.split("?")[0]} -- it is probably running older code. ` +
          "Restart it: Ctrl+C, then .venv/bin/python -m uvicorn services.api.app.main:app --port 8000 --reload",
        404,
      );
    }
    throw new ApiError(body.detail ?? res.statusText, res.status);
  }
  return res.json() as Promise<T>;
}

export const getTracks = () => get<PersonTrack[]>("/person/tracks?limit=500");
export const getRecordings = () => get<RecordingGroup[]>("/recordings");
export const getTrack = (id: string) => get<PersonTrackDetail>(`/person/tracks/${id}`);
export const getModels = () => get<ReIDModelInfo[]>("/person/models");
export const getSimilar = (id: string, modelKey: string) =>
  get<SimilarTracksResponse>(
    `/person/tracks/${id}/similar?model_key=${encodeURIComponent(modelKey)}&top_k=8`,
  );

/** Browser URL of an evidence crop (served by the API through the /api proxy). */
export const cropUrl = (uri: string) =>
  `/api/crops/${uri.split("/").map(encodeURIComponent).join("/")}`;

/** "15:50:05" -- recordings are named in UTC (RECORDING_TZ), so times are shown in UTC. */
export const clock = (iso: string) => new Date(iso).toISOString().slice(11, 19);
